"""What the viewer's facade shader (web/src/facade.ts) reads from the UVs of a building's walls.

`add_walls(facade=True)` keeps u running 0..bays and v running 0..floors, as always, but adds two whole
numbers the shader reads back and the repeating pattern does not notice: a per-wall seed in u and the
number of storeys in v.
"""

import json
import math
import struct

import numpy as np
import pytest
from shapely.geometry import Point, box

from boston_pipeline.features import Building, Scene
from boston_pipeline.mesh import (
    CURVE_MIN,
    FACADE_SEEDS,
    MAX_STOREYS,
    SEED_STRIDE,
    STOREY_STRIDE,
    MeshBuilder,
    add_walls,
    colour_bytes,
    facade_seeds,
)
from boston_pipeline.tiles import build


def _walls(polygon, z1=7.0, z0=0.0, facade=True, origin=(0.0, 0.0)):
    builder = MeshBuilder("walls", uvs=True, colours=True)
    add_walls(builder, polygon, z0, z1, np.asarray(origin), colour_bytes("#ffffff"), facade=facade)
    mesh = builder.finish()
    return mesh.uvs.reshape(-1, 4, 2)  # one row per wall: its four corners (bottom start, bottom end, top end, top start)


def _decode(uv):
    """What the shader recovers: (seed, storeys, u along the wall, v up the wall), as in facade.ts.

    The seed is 0..FACADE_SEEDS - 1; a raised wall adds FACADE_SEEDS (the shader subtracts it again and
    keeps it as a flag), which `_raised` reads.
    """
    seed = np.floor((uv[..., 0] + 0.5) / SEED_STRIDE)
    storeys = np.floor(uv[..., 1] / STOREY_STRIDE + 0.004)
    return seed % FACADE_SEEDS, storeys, uv[..., 0] - seed * SEED_STRIDE, uv[..., 1] - storeys * STOREY_STRIDE


def _raised(uv):
    return np.floor((uv[..., 0] + 0.5) / SEED_STRIDE) >= FACADE_SEEDS


def test_without_the_flag_the_uvs_are_plain_facade_cells():
    walls = _walls(box(0, 0, 9, 6), facade=False)
    assert walls[:, :, 0].min() == 0 and walls[:, :, 0].max() == 3  # 9 m is 3 bays
    assert walls[:, :, 1].min() == 0 and walls[:, :, 1].max() == 2  # 7 m is 2 storeys


def test_a_wall_carries_its_seed_and_its_storeys_and_the_pattern_is_unchanged():
    plain = _walls(box(0, 0, 9, 6), facade=False)
    walls = _walls(box(0, 0, 9, 6), facade=True)
    seed, storeys, u, v = _decode(walls)
    assert (storeys == 2).all()  # 7 m tall: two storeys
    assert ((seed >= 0) & (seed < FACADE_SEEDS)).all()
    assert (seed == np.floor(seed)).all()
    assert not _raised(walls).any()  # it starts on the ground
    # Take away the whole numbers and the cells are exactly what they were.
    assert np.allclose(u, plain[:, :, 0]) and np.allclose(v, plain[:, :, 1])
    # And they are whole numbers: the fractional part of u and v is the same with or without them.
    assert np.allclose(np.mod(walls, 1.0), np.mod(plain, 1.0))


def test_the_storey_count_survives_float32_for_the_tallest_wall_and_a_wall_with_no_bays():
    for floors in (1, 2, 17, 60, MAX_STOREYS):
        walls = _walls(box(0, 0, 9, 6), z1=floors * 3.5)
        assert (_decode(walls)[1] == floors).all(), floors
        # the top of the wall is exactly `floors` storeys above its bottom
        assert np.allclose(walls[:, 2, 1] - walls[:, 0, 1], floors)
    # More storeys than the encoding holds: no storey count, and v is the plain count.
    walls = _walls(box(0, 0, 9, 6), z1=(MAX_STOREYS + 5) * 3.5)
    assert (walls[:, 0, 1] == 0).all() and (walls[:, 2, 1] == MAX_STOREYS + 5).all()


def test_a_short_wall_stays_blank_and_has_u_of_exactly_zero():
    # A 1 m wall between two long ones: too short for a bay, and not part of a curve.
    from shapely.geometry import Polygon

    notch = Polygon([(0, 0), (12, 0), (12, 5), (11, 5), (11, 5.4), (0, 5.4)])
    builder = MeshBuilder("walls", uvs=True)
    add_walls(builder, notch, 0.0, 7.0, np.zeros(2), facade=True)
    walls = builder.finish().uvs.reshape(-1, 4, 2)
    blank = (walls[:, :, 0] == 0).all(axis=1)
    assert blank.sum() == 2  # the 0.4 m and the 1.0 m walls
    assert (walls[~blank][:, :, 0] > 0).any()
    # exactly zero: the shader tells a blank wall by u not changing at all
    assert (walls[blank][:, :, 0] == 0.0).all()


def test_the_seed_comes_from_where_a_wall_starts_and_is_the_same_each_time():
    starts = np.array([[0.0, 0.0], [10.0, 3.0], [-4.2, 7.7], [100.0, 100.0]])
    first, second = facade_seeds(starts), facade_seeds(starts.copy())
    assert (first == second).all()
    assert ((first >= 0) & (first < FACADE_SEEDS)).all()
    many = facade_seeds(np.random.default_rng(1).uniform(0, 256, size=(500, 2)))
    assert set(many.tolist()) == set(range(FACADE_SEEDS))  # all of them get used
    # the same wall, built twice, gets the same numbers
    assert np.array_equal(_walls(box(3, 4, 15, 9)), _walls(box(3, 4, 15, 9)))


def test_a_wall_that_starts_above_the_ground_is_marked_raised_and_keeps_its_pattern():
    low = _walls(box(0, 0, 9, 6), z0=0.0, z1=14.0)
    high = _walls(box(0, 0, 9, 6), z0=10.0, z1=24.0)  # the tower on a podium
    assert not _raised(low).any() and _raised(high).all()
    for walls in (low, high):
        seed, storeys, u, v = _decode(walls)
        assert (storeys == 4).all() and u.min() == 0 and u.max() == 3 and v.min() == 0 and v.max() == 4
    assert (_decode(low)[0] == _decode(high)[0]).all()  # the same seed, flagged
    # blank walls stay exactly blank whether or not they are raised
    from shapely.geometry import Polygon

    builder = MeshBuilder("walls", uvs=True)
    add_walls(builder, Polygon([(0, 0), (12, 0), (12, 5), (11, 5), (11, 5.4), (0, 5.4)]), 10.0, 20.0, np.zeros(2), facade=True)
    walls = builder.finish().uvs.reshape(-1, 4, 2)
    assert (walls[(walls[:, :, 0].max(axis=1) == 0)][:, :, 0] == 0).all()


def test_windows_carry_on_round_a_curved_front_made_of_short_facets():
    circle = Point(0, 0).buffer(10.0, quad_segs=16)  # 64 facets of about 1 m
    plain = _walls(circle, facade=False)
    assert (plain[:, :, 0] == 0).all()  # without the flag a curve is blank, as before
    walls = _walls(circle)
    seed, _, u, _ = _decode(walls)
    assert len(walls) == 64
    # u runs on from each facet into the next, so the cells flow round the curve
    for a, b in zip(walls[:-1], walls[1:]):
        assert a[1, 0] == pytest.approx(b[0, 0], abs=1e-4)
    perimeter = 2 * math.pi * 10.0
    assert walls[-1][1, 0] - walls[0][0, 0] == pytest.approx(perimeter / 3.0, rel=0.02)
    assert (seed == seed[0]).all()  # one seed for the whole curve


def test_a_few_short_facets_are_not_a_curve():
    from shapely.geometry import Polygon

    # A corner cut off by a 1 m diagonal: 1 m in all, under CURVE_MIN, so it stays blank.
    chamfered = Polygon([(0, 0), (12, 0), (12, 9.3), (11.3, 10.0), (0, 10)])
    builder = MeshBuilder("walls", uvs=True)
    add_walls(builder, chamfered, 0.0, 7.0, np.zeros(2), facade=True)
    walls = builder.finish().uvs.reshape(-1, 4, 2)
    lengths_blank = (walls[:, :, 0] == 0).all(axis=1)
    assert lengths_blank.sum() == 1
    assert CURVE_MIN > 1.0


def test_the_tile_writes_the_encoded_uvs_for_buildings(tmp_path):
    scene = Scene()
    scene.buildings.append(Building("w1", box(100, 100, 109, 106), 14.0, 0.0, "height", "#8a4b3c", "#5a5a5c", {"building": "yes"}))
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    data = (tmp_path / "tiles" / "0_0.glb").read_bytes()
    json_length, _ = struct.unpack_from("<I4s", data, 12)
    document = json.loads(data[20 : 20 + json_length])
    binary = data[28 + json_length :]
    walls = [m for m in document["meshes"] if m["name"] == "building_walls"][0]["primitives"][0]
    accessor = document["accessors"][walls["attributes"]["TEXCOORD_0"]]
    view = document["bufferViews"][accessor["bufferView"]]
    uv = np.frombuffer(binary, np.float32, accessor["count"] * 2, view["byteOffset"]).reshape(-1, 2)
    _, storeys, u, v = _decode(uv)
    assert (storeys == 4).all()  # 14 m: four storeys
    assert u.min() == 0 and u.max() == 3  # the 9 m walls are three bays, the 6 m walls two
    assert v.min() == 0 and v.max() == 4

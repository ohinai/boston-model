"""How the ground looks: lawns that vary, ponds that are not blue, beds that are not purple slabs, mulch rings."""

import json
import struct

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from boston_pipeline import groundlook
from boston_pipeline.features import Area, Point, Scene, water_cover
from boston_pipeline.groundlook import (
    BED_BLOOM,
    BED_FOLIAGE,
    add_lawn,
    bed_colour,
    bed_style,
    cut,
    lawn_factors,
    mulch_radius,
    value_noise,
)
from boston_pipeline.mesh import MeshBuilder, srgb_to_linear
from boston_pipeline.tiles import COVERS, build


def _glb(path):
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + length])
    start = 20 + length + 8

    def read(index, dtype, width):
        accessor = document["accessors"][index]
        view = document["bufferViews"][accessor["bufferView"]]
        return np.frombuffer(data, dtype, accessor["count"] * width, start + view["byteOffset"]).reshape(-1, width)

    return document, read


def _layers(path):
    document, read = _glb(path)
    return {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}, document, read


def _luminance(hex_colour):
    return float(np.dot(srgb_to_linear(hex_colour), [0.2126, 0.7152, 0.0722]))


def test_noise_is_repeatable_and_smooth():
    x, y = np.linspace(-500, 500, 400), np.linspace(300, -300, 400)
    first, second = value_noise(x, y, 25.0, seed=3), value_noise(x, y, 25.0, seed=3)
    assert np.array_equal(first, second)
    assert np.abs(first).max() <= 1.0
    assert not np.array_equal(first, value_noise(x, y, 25.0, seed=4))
    # Smooth: 0.1 m along a 25 m wave never moves it far.
    assert np.abs(value_noise(x + 0.1, y, 25.0, 3) - first).max() < 0.05


def test_lawn_colour_depends_only_on_position_and_averages_to_the_base():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-900, 900, 5000), rng.uniform(-900, 900, 5000)
    whole = lawn_factors(x, y)
    # The same place gets the same colour whatever else is asked for with it.
    assert np.allclose(whole[100:200], lawn_factors(x[100:200], y[100:200]))
    assert np.allclose(whole[::-1][::-1], whole)
    # The grass is not one colour, but it is the base colour on average (so the park does not get lighter or darker).
    assert whole[:, 1].std() > 0.08
    assert whole.mean(axis=0) == pytest.approx([1, 1, 1], abs=0.06)
    assert whole.min() >= groundlook.LAWN_FLOOR * 0.5


def test_cut_pieces_make_up_the_polygon_exactly():
    polygon = Polygon([(3, 2), (45, 4), (40, 33), (10, 30)], [[(15, 10), (15, 20), (28, 20), (28, 10)]])
    pieces = cut(polygon, 8.0)
    assert len(pieces) > 10
    assert sum(p.area for p in pieces) == pytest.approx(polygon.area, rel=1e-6)
    # No overlaps: the union has the same area as the sum.
    assert pieces[0].union(pieces[1]).area == pytest.approx(pieces[0].area + pieces[1].area, rel=1e-6)
    # A polygon within one cell is left whole.
    assert cut(box(1, 1, 3, 3), 8.0) == [box(1, 1, 3, 3)]


def test_lawn_pieces_agree_along_their_seams():
    """Two lawns side by side get the same colour where they meet, so a tile edge shows no seam."""
    left, right = MeshBuilder("green", colours=True), MeshBuilder("green", colours=True)
    origin = np.zeros(2)
    add_lawn(left, box(-30, 0, 16, 30), 0.0, origin, "#6d8c4c")
    add_lawn(right, box(16, 0, 60, 30), 0.0, origin, "#6d8c4c")

    def seam(builder):
        mesh = builder.finish()
        east, north = mesh.positions[:, 0], -mesh.positions[:, 2]
        on = np.isclose(east, 16.0)
        return {(round(float(e), 3), round(float(n), 3)): tuple(c) for e, n, c in zip(east[on], north[on], mesh.colours[on])}

    a, b = seam(left), seam(right)
    shared = set(a) & set(b)
    assert len(shared) >= 4
    assert all(a[key] == b[key] for key in shared)
    # And it really is patchy: more than a handful of different colours over the lawn.
    assert len({tuple(c) for c in left.finish().colours}) > 10


def test_lawn_triangle_count_stays_modest():
    builder = MeshBuilder("green", colours=True)
    add_lawn(builder, box(0, 0, 256, 256), 0.0, np.zeros(2), "#6d8c4c")
    # About two triangles per 8 m cell: a whole tile of lawn, a few thousand triangles.
    assert len(builder.finish().indices) // 3 <= 2 * (256 // 8) ** 2 + 200


def test_water_cover_tells_ponds_from_rivers():
    assert water_cover({"natural": "water", "water": "pond"}, 15000) == "pond"
    assert water_cover({"natural": "water", "water": "lagoon"}, 39000) == "pond"
    assert water_cover({"natural": "water", "water": "basin"}, 50) == "pond"
    assert water_cover({"natural": "water", "water": "river"}, 1700) == "open"
    assert water_cover({"natural": "water", "water": "canal"}, 130000) == "open"
    assert water_cover({"waterway": "dock"}, 2000) == "open"
    # Water of no stated kind is judged by its size.
    assert water_cover({"natural": "water"}, 80) == "pond"
    assert water_cover({"natural": "water", "water": "yes"}, 1_700_000) == "open"


def test_a_pond_is_olive_and_open_water_is_blue():
    pond, river = COVERS["water"]["pond"], COVERS["water"]["open"]
    r, g, b = srgb_to_linear(pond)
    assert g > b and g > r  # olive, not blue
    r, g, b = srgb_to_linear(river)
    assert b > g > r
    assert list(COVERS["water"])[-1] == "open"  # the default for water of any other kind


def test_asphalt_paths_are_not_lighter_than_pavement_and_not_blue():
    asphalt, concrete, paved = (COVERS["paths"][name] for name in ("asphalt", "concrete", "paved"))
    assert _luminance(asphalt) < min(_luminance(concrete), _luminance(paved))
    r, _, b = srgb_to_linear(asphalt)
    assert r >= b  # a neutral or warm grey: the sky tint in the shade does the rest


def test_beds_are_foliage_with_a_little_bloom_and_never_purple():
    for colour in BED_FOLIAGE:
        r, g, b = srgb_to_linear(colour)
        assert g > r and g > b
    for colour in BED_BLOOM:
        r, g, b = srgb_to_linear(colour)
        assert r >= b * 0.9 and not (b > g and b > r * 0.9)  # warm: gold, orange, cream, a dull red or pink
    assert bed_style("w1") == bed_style("w1")
    assert {bed_style(f"w{k}")[1] is None for k in range(200)} == {True, False}


def test_a_bed_is_mottled_not_one_slab():
    x, y = np.meshgrid(np.linspace(0, 12, 40), np.linspace(0, 12, 40))
    colours = bed_colour("w824367503")(x.ravel(), y.ravel())
    assert colours.dtype == np.uint8 and colours.shape == (1600, 4)
    assert len({tuple(c) for c in colours}) > 100
    assert np.array_equal(colours, bed_colour("w824367503")(x.ravel(), y.ravel()))
    # Mostly green: the average planting has more green than red or blue.
    mean = colours[:, :3].mean(axis=0)
    assert mean[1] > mean[0] and mean[1] > mean[2]


def test_mulch_ring_grows_with_the_trunk_within_limits():
    assert mulch_radius({}) == pytest.approx(0.9 + 1.2 * 0.12)
    assert mulch_radius({"circumference": "1.28"}) > mulch_radius({"circumference": "0.5"})
    assert mulch_radius({"circumference": "0.01"}) == pytest.approx(0.9, abs=0.01)
    assert mulch_radius({"circumference": "20"}) == 2.0


def _park_scene(extra_tree_tags=None):
    scene = Scene()
    park = box(0, 0, 200, 200)  # 40,000 m²: a large park
    scene.areas["green"].append(Area("w1", park, {"leisure": "park", "name": "Big Park"}, "lawn"))
    scene.areas["path"].append(Area("w2", box(0, 100, 200, 104), {"highway": "footway"}, "asphalt"))
    scene.trees = [
        Point("n1", 50.0, 50.0, tags={"natural": "tree", "circumference": "1.2", **(extra_tree_tags or {})}),  # on the lawn
        Point("n2", 100.0, 102.0, tags={"natural": "tree"}),  # on the path
        Point("n3", 240.0, 50.0, tags={"natural": "tree"}),  # beyond the park
    ]
    return scene


def _tile(tmp_path, scene):
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    return _layers(tmp_path / "tiles" / "0_0.glb")


def test_a_tree_on_the_lawn_of_a_large_park_stands_in_mulch(tmp_path):
    layers, document, read = _tile(tmp_path, _park_scene())
    # The scene has no flower beds, so a bed layer exists only because of the ring round the one tree on the lawn:
    # the tree on the path and the one beyond the park get none.
    primitive = layers["beds"]
    positions = read(primitive["attributes"]["POSITION"], np.float32, 3)
    # Everything is one ring, 3 cm above the lawn, round the tree at (50, 50): glTF x east, z -north.
    assert positions[:, 1] == pytest.approx(groundlook.MULCH_LIFT)
    east, north = positions[:, 0], -positions[:, 2]
    radius = mulch_radius({"circumference": "1.2"})
    assert np.hypot(east - 50, north - 50).max() == pytest.approx(radius, abs=0.05)
    assert "COLOR_0" in primitive["attributes"]


def test_a_tree_can_decline_its_mulch_ring(tmp_path):
    layers, _, _ = _tile(tmp_path, _park_scene({"tree_surround": "none"}))
    assert "beds" not in layers


def test_small_parks_do_not_get_mulch_rings(tmp_path):
    scene = _park_scene()
    scene.areas["green"][0] = Area("w1", box(0, 0, 100, 100), {"leisure": "park", "name": "Pocket Park"}, "lawn")  # 10,000 m²
    layers, _, _ = _tile(tmp_path, scene)
    assert "beds" not in layers

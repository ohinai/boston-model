"""Fences, walls, hedges, chains, kerbs, guard rails and bollards: reading them and drawing them."""

import json

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from boston_pipeline import world
from boston_pipeline.barriers import add_barrier
from boston_pipeline.features import BARRIERS, Area, Barrier, Building, Scene, Way, classify
from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.tiles import build

from test_parks import _mesh_data, _node, _read_glb


def _lonlat(points):
    return [{"lon": float(world.to_lonlat(x, y)[0]), "lat": float(world.to_lonlat(x, y)[1])} for x, y in points]


def _way(ident, tags, points):
    return {"type": "way", "id": ident, "tags": tags, "geometry": _lonlat(points)}


LINE = [(0, 0), (10, 0)]


@pytest.mark.parametrize(
    ("tags", "kind", "height", "width"),
    [
        ({"barrier": "fence"}, "fence", 1.2, 0.06),
        ({"barrier": "fence", "fence_type": "railing", "height": "1.5"}, "fence", 1.5, 0.06),
        ({"barrier": "chain"}, "chain", 0.6, 0.05),
        ({"barrier": "chain", "height": "0.5"}, "chain", 0.5, 0.05),
        ({"barrier": "wall"}, "wall", 1.0, 0.4),
        ({"barrier": "wall", "height": "2 m", "width": "0.3"}, "wall", 2.0, 0.3),
        ({"barrier": "city_wall"}, "wall", 1.0, 0.4),
        ({"barrier": "retaining_wall"}, "retaining_wall", 1.0, 0.4),
        ({"barrier": "hedge"}, "hedge", 1.2, 0.8),
        ({"barrier": "guard_rail"}, "guard_rail", 0.7, 0.1),
        ({"barrier": "kerb"}, "kerb", 0.15, 0.3),
        ({"barrier": "bollard"}, "bollards", 0.9, 0.0),
        ({"barrier": "fence", "height": "50"}, "fence", 1.2, 0.06),  # not a sensible height
        ({"barrier": "fence", "height": "tall"}, "fence", 1.2, 0.06),
    ],
)
def test_ways_become_barriers_with_defaults(tags, kind, height, width):
    (barrier,) = classify([_way(1, tags, LINE)]).barriers
    assert barrier.kind == kind
    assert barrier.height == pytest.approx(height, abs=1e-3) and barrier.width == pytest.approx(width, abs=1e-3)
    assert barrier.id == "w1" and barrier.line.length == pytest.approx(10, abs=0.05)
    assert kind in {entry[0] for entry in BARRIERS.values()}


def test_colour_and_material_are_read():
    (barrier,) = classify([_way(1, {"barrier": "fence", "colour": "black", "material": "metal;wood"}, LINE)]).barriers
    assert barrier.colour == "#2b2b2d" and barrier.material == "metal"
    (barrier,) = classify([_way(1, {"barrier": "wall", "wall": "dry_stone"}, LINE)]).barriers
    assert barrier.colour is None and barrier.material == "dry_stone"


def test_gates_underground_and_other_roles_are_not_barriers():
    scene = classify([
        _way(1, {"barrier": "gate", "fence_type": "iron"}, LINE),
        _way(2, {"barrier": "fence", "tunnel": "yes"}, LINE),
        _way(3, {"barrier": "fence", "layer": "-1"}, LINE),
        _way(4, {"barrier": "fence", "indoor": "yes"}, LINE),
        _way(5, {"barrier": "gate", "highway": "footway"}, LINE),
        _way(6, {"barrier": "fence", "highway": "service"}, LINE),
        _way(7, {"barrier": "unheard_of"}, LINE),
    ])
    assert scene.barriers == []


def test_closed_barrier_is_an_outline_not_an_area():
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    scene = classify([_way(1, {"barrier": "hedge", "leisure": "garden"}, ring), _way(2, {"barrier": "wall"}, ring)])
    assert [b.kind for b in scene.barriers] == ["hedge", "wall"]
    assert all(b.line.geom_type == "LineString" and b.line.is_ring for b in scene.barriers)
    assert len(scene.areas["green"]) == 1  # only the garden is an area, and only because of its own tag
    assert scene.areas["road"] == [] and scene.areas["path"] == []


def test_bollard_and_block_nodes_are_objects():
    scene = classify([
        _node(1, {"barrier": "bollard", "height": "1", "colour": "black"}),
        _node(2, {"barrier": "block", "material": "granite"}),
        _node(3, {"barrier": "gate"}),
    ])
    assert [(o.kind, o.height) for o in scene.objects] == [("bollard", 1.0), ("block", None)]
    assert scene.objects[0].colour == "#2b2b2d" and scene.objects[1].material == "granite"


def _water(ident, tags, outer, holes=()):
    members = [{"type": "way", "role": "outer", "geometry": _lonlat(outer)}]
    members += [{"type": "way", "role": "inner", "geometry": _lonlat(h)} for h in holes]
    return {"type": "relation", "id": ident, "tags": {"type": "multipolygon", "natural": "water", **tags}, "members": members}


def test_water_with_an_edge_gets_a_kerb_round_each_outer_ring_on_the_land_side():
    outer = [(0, 0), (40, 0), (40, 30), (0, 30), (0, 0)]
    island = [(15, 10), (25, 10), (25, 20), (15, 20), (15, 10)]
    elements = [
        _water(9, {"edge": "kerb", "edge:width": "0.5", "edge:height": "0.2"}, outer, [island]),
        _water(10, {"edge": "kerb"}, [(100, 0), (120, 0), (120, 20), (100, 20), (100, 0)]),
        _water(11, {}, [(200, 0), (210, 0), (210, 10), (200, 10), (200, 0)]),
    ]
    scene = classify(elements)
    (kerb,) = [b for b in scene.barriers if b.id == "r9"]  # one: the island is not edged
    assert (kerb.kind, kerb.height, kerb.width, kerb.material) == ("kerb", 0.2, 0.5, "granite")
    pond = next(a.polygon for a in scene.areas["water"] if a.id == "r9")
    # The line runs a quarter of a metre outside the shore, so the 0.5 m strip lies on the land.
    assert Polygon(kerb.line).contains(pond.exterior) and kerb.line.distance(pond.exterior) == pytest.approx(0.25, abs=0.02)
    assert not kerb.line.intersects(Polygon(island))
    (other,) = [b for b in scene.barriers if b.id == "r10"]
    assert (other.width, other.height) == (0.4, 0.15)
    assert not any(b.id == "r11" for b in scene.barriers)


# --- Geometry -----------------------------------------------------------------
LONG = np.array([(0.0, 0.0), (100.0, 0.0)])
BENT = np.array([(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (12.0, 8.0)])
RING = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)])


def _make(kind, line, height, width, colour=None, material=None, feature=3):
    builder = MeshBuilder("barriers", colours=True, features=True)
    add_barrier(builder, kind, line, height, width, colour, material, feature)
    return builder.finish()


def _enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


CASES = [
    ("fence", BENT, 1.2, 0.06),
    ("fence", BENT, 1.8, 0.06),
    ("chain", BENT, 0.6, 0.05),
    ("wall", BENT, 1.0, 0.4),
    ("retaining_wall", BENT, 1.4, 0.4),
    ("hedge", BENT, 1.2, 0.8),
    ("guard_rail", BENT, 0.7, 0.1),
    ("kerb", BENT, 0.15, 0.3),
    ("bollards", BENT, 0.9, 0.0),
    ("wall", RING, 1.0, 0.4),
    ("fence", RING, 1.2, 0.06),
    ("chain", RING, 0.6, 0.05),
]


@pytest.mark.parametrize("kind,line,height,width", CASES)
def test_barrier_shapes_are_sound(kind, line, height, width):
    mesh = _make(kind, line, height, width)
    assert len(mesh.indices) > 0 and len(mesh.indices) % 3 == 0
    up = _enu(mesh)[:, 2]
    assert up.min() >= -1e-6, "something below ground"
    assert up.max() == pytest.approx(height, abs=0.01), "the top is at the stated height"
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-9).all(), "degenerate triangle"
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()
    assert (mesh.features == 3).all() and mesh.colours.shape == (len(mesh.positions), 4)


@pytest.mark.parametrize("kind,height", [("fence", 1.2), ("fence", 1.8), ("chain", 0.6), ("guard_rail", 0.7)])
def test_a_long_run_stays_cheap(kind, height):
    # A straight 100 m run, and the same run with a vertex every 2.5 m.
    for line in (LONG, np.column_stack([np.arange(0, 100.1, 2.5), np.zeros(41)])):
        mesh = _make(kind, line, height, 0.06)
        assert len(mesh.indices) // 3 < 1500, (kind, len(mesh.indices) // 3)


def test_fence_has_posts_at_ends_and_bends_and_a_middle_rail_when_tall():
    low, tall = _make("fence", BENT, 1.2, 0.06), _make("fence", BENT, 1.4, 0.06)
    assert len(tall.indices) > len(low.indices)
    enu = _enu(low)
    for x, y in BENT:  # a post stands at each vertex
        near = enu[np.hypot(enu[:, 0] - x, enu[:, 1] - y) < 0.06]
        assert near[:, 2].max() == pytest.approx(1.2, abs=1e-3) and near[:, 2].min() == pytest.approx(0, abs=1e-3)


def test_chain_hangs_below_the_post_tops():
    enu = _enu(_make("chain", np.array([(0.0, 0.0), (2.4, 0.0)]), 0.6, 0.05))
    assert enu[:, 2].max() == pytest.approx(0.6, abs=1e-3)
    middle = enu[(enu[:, 0] > 0.3) & (enu[:, 0] < 2.1)]  # the chain between the posts
    assert middle[:, 2].max() < 0.6 - 0.08 + 1e-6, "hung below the post tops"
    assert middle[:, 2].min() < 0.6 - 0.08 - 0.10, "and sagging at the middle"


def test_solid_strips_have_the_stated_width_and_colours_by_material():
    line = np.array([(0.0, 0.0), (10.0, 0.0)])
    enu = _enu(_make("wall", line, 1.0, 0.4))
    assert enu[:, 1].max() - enu[:, 1].min() == pytest.approx(0.4, abs=1e-3)
    assert enu[:, 0].min() == pytest.approx(0, abs=1e-3) and enu[:, 0].max() == pytest.approx(10, abs=1e-3)
    brick, stone = _make("wall", line, 1.0, 0.4, material="brick"), _make("wall", line, 1.0, 0.4, material="granite")
    assert not np.array_equal(brick.colours[0], stone.colours[0])
    unknown, plain = _make("wall", line, 1.0, 0.4, material="unheard_of"), _make("wall", line, 1.0, 0.4, material="stone")
    assert np.array_equal(unknown.colours, plain.colours)


def test_a_colour_tag_overrides_fence_chain_rail_and_bollards():
    for kind in ("fence", "chain", "guard_rail", "bollards"):
        plain, red = _make(kind, BENT, 1.0, 0.1), _make(kind, BENT, 1.0, 0.1, colour="#9c3b30")
        assert (red.colours == red.colours[0]).all() and not np.array_equal(plain.colours[0], red.colours[0])


def test_bollards_stand_every_metre_and_a_half():
    mesh = _make("bollards", np.array([(0.0, 0.0), (6.0, 0.0)]), 0.9, 0.0)
    enu = _enu(mesh)
    tops = np.unique(np.round(enu[enu[:, 2] > 0.89][:, 0] / 1.5))  # which spacing each cap belongs to
    assert tops.tolist() == [0, 1, 2, 3, 4]


# --- Tiles ---------------------------------------------------------------------
def _barrier(ident, kind, points, height=1.2, width=0.06, **more):
    return Barrier(ident, LineString(points), kind, height, width, tags=more.pop("tags", {}), **more)


def _path(ident, points, width=2.2):
    return Way(ident, LineString(points), "footway", "path", width, None, {}, "paved")


def _tile(tmp_path, scene, name="0_0"):
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    path = tmp_path / "tiles" / f"{name}.glb"
    document = _read_glb(path)
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    semantics = json.loads((tmp_path / "tiles" / f"{name}.json").read_text(encoding="utf-8"))
    return path, document, layers, semantics


def _points(path, document, layers):
    positions, _ = _mesh_data(path, document, layers["barriers"])
    return np.column_stack([positions[:, 0], -positions[:, 2], positions[:, 1]])  # east, north, up


def test_a_path_across_a_fence_leaves_a_gap(tmp_path):
    scene = Scene()
    scene.barriers.append(_barrier("w1", "fence", [(20, 50), (120, 50)], tags={"name": "Railings"}))
    scene.ways.append(_path("w2", [(70, 0), (70, 100)]))
    path, document, layers, semantics = _tile(tmp_path, scene)
    assert "barriers" in layers and "COLOR_0" in layers["barriers"]["attributes"]
    east = _points(path, document, layers)[:, 0]
    # The strip is 2.2 m wide, and 0.4 m is cleared on each side: nothing between 68.5 and 71.5.
    assert not ((east > 68.5 + 0.1) & (east < 71.5 - 0.1)).any()
    assert (east < 68.5).any() and (east > 71.5).any()
    entries = [o for o in semantics["objects"] if o["kind"] == "fence"]
    assert len(entries) == 1 and entries[0]["id"] == "w1" and entries[0]["name"] == "Railings"
    assert 20 <= entries[0]["pos"][0] <= 120
    ids = layers["barriers"]["extensions"]["EXT_mesh_features"]["featureIds"][0]
    assert ids["featureCount"] == 1


def test_buildings_and_short_pieces_cut_barriers(tmp_path):
    scene = Scene()
    scene.buildings.append(Building("w9", box(40, 40, 60, 60), 10.0, 0.0, "default", "#aaaaaa", "#555555", {}))
    scene.barriers += [
        _barrier("w1", "wall", [(0, 50), (100, 50)], 1.0, 0.4),
        _barrier("w3", "fence", [(200, 200), (200.4, 200)]),  # shorter than half a metre
    ]
    path, document, layers, semantics = _tile(tmp_path, scene)
    points = _points(path, document, layers)
    inside = (points[:, 0] > 40.01) & (points[:, 0] < 59.99) & (points[:, 1] > 49) & (points[:, 1] < 51)
    assert not inside.any() and (points[:, 0] < 40.01).any() and (points[:, 0] > 59.99).any()
    assert [o["id"] for o in semantics["objects"] if o["kind"] in ("wall", "fence")] == ["w1"]


def test_a_barrier_crossing_two_tiles_is_cut_at_the_boundary(tmp_path):
    size = world.TILE_SIZE
    scene = Scene()
    scene.barriers.append(_barrier("w1", "wall", [(size - 30, 50), (size + 40, 50)], 1.0, 0.4))
    stats = build(scene, [(0, 0), (1, 0)], tmp_path, {"name": "Test"}, None)
    assert stats["barriers"] == 1
    seen = {}
    for name in ("0_0", "1_0"):
        file = tmp_path / "tiles" / f"{name}.glb"
        document = _read_glb(file)
        layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
        seen[name] = _points(file, document, layers)
        semantics = json.loads((tmp_path / "tiles" / f"{name}.json").read_text(encoding="utf-8"))
        assert [o["id"] for o in semantics["objects"] if o["kind"] == "wall"] == ["w1"]
    assert seen["0_0"][:, 0].max() == pytest.approx(size, abs=1e-3)
    assert seen["0_0"][:, 0].min() == pytest.approx(size - 30, abs=1e-3)
    assert seen["1_0"][:, 0].min() == pytest.approx(0, abs=1e-3) and seen["1_0"][:, 0].max() == pytest.approx(40, abs=1e-3)


def test_a_kerb_is_not_cut_by_a_path_beside_it_but_is_cut_by_a_dock(tmp_path):
    ring = [(60, 60), (160, 60), (160, 120), (60, 120), (60, 60)]
    scene = Scene()
    scene.barriers.append(_barrier("r1", "kerb", ring, 0.15, 0.4, material="granite"))
    scene.ways += [
        _path("w2", [(40, 60.5), (180, 60.5)]),  # runs along the near edge
        _path("w3", [(110, 40), (110, 140)]),  # crosses it
    ]
    scene.areas["path"].append(Area("w4", box(130, 110, 140, 130), {"man_made": "pier"}, "wood"))  # a dock on the far edge
    path, document, layers, _ = _tile(tmp_path, scene)
    points = _points(path, document, layers)
    along = points[(points[:, 1] < 61) & (points[:, 1] > 59.5)]
    assert along[:, 0].min() == pytest.approx(60, abs=0.3) and along[:, 0].max() == pytest.approx(160, abs=0.3)
    assert not ((along[:, 0] > 105) & (along[:, 0] < 115)).any(), "a path does not cut a kerb: no gap where w3 crosses"
    far = points[(points[:, 1] > 119) & (points[:, 1] < 121)]
    assert not ((far[:, 0] > 129.7) & (far[:, 0] < 140.3)).any(), "a dock does"
    assert (far[:, 0] < 129.7).any() and (far[:, 0] > 140.3).any()

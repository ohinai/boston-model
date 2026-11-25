"""The suspension footbridge, the open pavilion and the painted dock."""

import json
import struct

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from boston_pipeline.bridges import (
    GLOBE_RADIUS, IRONWORK, LIP, PLAIN, PYLON_TOP, RAIL_HEIGHT, SUSPENSION, SUSPENSION_DECK_DEPTH, _icosphere,
    add_bridge, bridge_style, deck_height,
)
from boston_pipeline.features import Area, Building, Object, Scene, Way
from boston_pipeline.mesh import MeshBuilder, colour_bytes
from boston_pipeline.pavilions import (
    DOCK_LIP, MIN_EAVE, OVERHANG, add_dock_edge, add_pavilion, painted_pier, roof_shape,
)
from boston_pipeline.tiles import build, prepare

DECK = np.array([200, 180, 150, 255], dtype=np.uint8)
STRUCTURE = np.array([90, 90, 95, 255], dtype=np.uint8)
RAIL = np.array([30, 30, 35, 255], dtype=np.uint8)
STRAIGHT = np.array([[10.0, 5.0], [50.0, 5.0]])  # 40 m east
RISE = 0.7
# One pylon each side at 12 m and at 28 m along the bridge, as the lamps beside the Garden's bridge stand.
TOWERS = [(12.0, 1, 4.9), (12.1, -1, 4.9), (28.0, 1, 4.9), (27.9, -1, 4.9)]


def _bridge(style=SUSPENSION, towers=TOWERS, line=STRAIGHT, width=4.0, railings=True):
    builder = MeshBuilder("bridges", colours=True, features=True)
    add_bridge(builder, line, width, rise=RISE, deck_colour=DECK, structure_colour=STRUCTURE, rail_colour=RAIL,
               railings=railings, feature=3, style=style, towers=towers, water_z=-0.6)
    return builder.finish()


def _up(mesh):
    return mesh.positions[:, 1]  # glTF y is up


def _east(mesh):
    return mesh.positions[:, 0]


def _north(mesh):
    return -mesh.positions[:, 2]


def test_only_a_bridge_tagged_suspension_is_drawn_as_one():
    assert bridge_style({"bridge:structure": "suspension"}) == SUSPENSION
    assert bridge_style({"bridge:structure": "beam"}) == PLAIN
    assert bridge_style({"highway": "footway", "bridge": "yes"}) == PLAIN


def test_the_plain_bridge_ignores_the_suspension_arguments():
    plain = _bridge(PLAIN, towers=())
    with_towers = _bridge(PLAIN, towers=TOWERS)
    assert np.array_equal(plain.positions, with_towers.positions) and np.array_equal(plain.indices, with_towers.indices)


def test_the_icosphere_faces_outward():
    verts, faces = _icosphere()
    assert len(faces) == 80 and np.allclose(np.linalg.norm(verts, axis=1), 1.0)
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.einsum("ij,ij->i", normal, tri.mean(axis=1)) > 0).all()


def test_suspension_normals_agree_with_winding_and_all_faces_belong_to_the_bridge():
    mesh = _bridge()
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    stored = mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]
    assert (np.einsum("ij,ij->i", face, stored) > 0).all()
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    assert (mesh.features == 3).all()


def test_the_deck_is_the_same_walking_surface_as_a_plain_bridge():
    plain, suspension = _bridge(PLAIN, towers=()), _bridge()

    def deck(mesh):
        on_deck = (mesh.colours == DECK).all(axis=1)
        return mesh.positions[on_deck]

    a, b = deck(plain), deck(suspension)
    assert len(a) == len(b) and np.allclose(a, b)


def test_a_walker_can_cross_between_the_pylons():
    mesh = _bridge(width=4.9)  # the Public Garden's bridge is this wide
    deck = np.array([deck_height(a, 40.0, RISE) for a in np.linspace(0, 40, 9)])
    assert deck.max() == pytest.approx(LIP + RISE, abs=0.01)
    # Nothing stands within 1.5 m of the centre line above the deck: pylons and railings keep to its edges.
    near_middle = np.abs(_north(mesh) - 5.0) < 1.5
    rail_top = np.array([deck_height(x - 10.0, 40.0, RISE) for x in _east(mesh)]) + 0.3
    assert not (near_middle & (_up(mesh) > rail_top)).any()


def test_pylons_stand_on_the_deck_edge_and_carry_a_globe_at_the_lamps_height():
    mesh = _bridge()
    top = _up(mesh).max()
    # The tallest thing is a lamp: its stated height above the deck where its pylon stands.
    assert top == pytest.approx(max(deck_height(a, 40.0, RISE) for a in (12.0, 28.0)) + 4.9, abs=0.03)
    white = (mesh.colours == colour_bytes("#f3f1ea")).all(axis=1)
    per_globe = 80 * 3  # 80 triangles, not welded until the tile is written
    assert white.sum() == 4 * per_globe
    centres = np.array([mesh.positions[white][i * per_globe:(i + 1) * per_globe].mean(axis=0) for i in range(4)])
    assert sorted(np.round(centres[:, 0] - 10.0, 1)) == pytest.approx([12.0, 12.0, 28.0, 28.0], abs=0.2)
    assert sorted(np.round(-centres[:, 2] - 5.0, 1)) == pytest.approx([-2.0, -2.0, 2.0, 2.0], abs=0.01)  # the deck's edges
    assert (centres[:, 1] > PYLON_TOP).all() and GLOBE_RADIUS < 0.5


def test_piers_go_down_to_the_water_and_the_abutments_stand_in_it():
    mesh = _bridge()
    assert _up(mesh).min() == pytest.approx(-0.6, abs=1e-3)
    low = _up(mesh) < -0.55
    # Stone at both ends of the span and under both pylon pairs; none in the middle.
    east = _east(mesh)[low] - 10.0
    assert east.min() < 0.1 and east.max() > 39.9
    assert not ((east > 15.0) & (east < 25.0)).any()


def test_the_railings_are_a_lattice_in_ironwork_and_the_triangle_budget_holds():
    mesh = _bridge()
    iron = (mesh.colours == colour_bytes(IRONWORK)).all(axis=1)
    assert iron.sum() > 1000
    assert len(mesh.indices) // 3 < 9000
    assert len(_bridge(railings=False).indices) // 3 < 300  # the deck alone
    # The girder is deeper than a plain bridge's slab: its underside is lower at the ends.
    assert _up(_bridge(railings=False)).min() == pytest.approx(LIP - SUSPENSION_DECK_DEPTH, abs=1e-4)


def test_a_suspension_bridge_without_towers_still_gets_lattice_railings():
    mesh = _bridge(towers=())
    assert (mesh.colours == colour_bytes(IRONWORK)).all(axis=1).sum() > 500
    assert _up(mesh).max() == pytest.approx(LIP + RISE + RAIL_HEIGHT, abs=0.1)


def _lagoon(extra_objects=()):
    scene = Scene()
    scene.areas["water"].append(Area("w10", box(50, 50, 150, 110), {"natural": "water"}))
    tags = {"highway": "footway", "bridge": "yes", "bridge:structure": "suspension"}
    scene.ways.append(Way("w11", LineString([(100, 40), (100, 120)]), "footway", "path", 4.0, "Pond Bridge", tags, "asphalt", True))
    scene.objects += [
        Object("n1", 102.0, 63.0, "lamp", height=4.9), Object("n2", 98.1, 63.4, "lamp", height=4.9),
        Object("n3", 102.0, 97.0, "lamp", height=4.9), Object("n4", 98.0, 96.8, "lamp", height=4.9),
        Object("n5", 200.0, 200.0, "lamp", height=4.9),  # on the ground, far from the bridge
        *extra_objects,
    ]
    return scene


def test_towers_are_the_lamps_standing_in_the_water_beside_the_deck():
    scene = _lagoon()
    ctx = prepare(scene, [(0, 0)])
    (found,) = ctx.towers.values()
    assert sorted(side for _, side, _ in found) == [-1, -1, 1, 1]
    assert all(height == 4.9 for _, _, height in found)
    alongs = sorted(round(a) for a, _, _ in found)
    assert alongs == [round(a) for a in sorted([63 - 49.7, 63.4 - 49.7, 97 - 49.7, 96.8 - 49.7])]
    # A bridge not tagged suspension has none.
    scene.ways[0].tags.pop("bridge:structure")
    assert prepare(scene, [(0, 0)]).towers == {}


def test_the_tile_draws_the_pylons_with_the_bridge_and_lists_the_lamps(tmp_path):
    scene = _lagoon()
    stats = build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    assert stats["bridges"] == 1
    document = _glb(tmp_path / "tiles" / "0_0.glb")
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    top = document["accessors"][layers["bridges"]["attributes"]["POSITION"]]["max"][1]
    assert top > 4.5  # pylons and their globe lamps
    # The far lamp on the ground is still a lamp in the furniture layer; the four beside the bridge are not.
    assert "furniture" in layers
    assert document["accessors"][layers["furniture"]["attributes"]["POSITION"]]["max"][1] == pytest.approx(4.9, abs=0.05)
    semantics = json.loads((tmp_path / "tiles" / "0_0.json").read_text(encoding="utf-8"))
    assert {entry["id"] for entry in semantics["objects"] if entry["kind"] == "lamp"} == {"n1", "n2", "n3", "n4", "n5"}
    plain = _lagoon()
    plain.ways[0].tags.pop("bridge:structure")
    build(plain, [(0, 0)], tmp_path / "plain", {"name": "Test"}, None)
    plain_doc = _glb(tmp_path / "plain" / "tiles" / "0_0.glb")
    plain_layers = {mesh["name"]: mesh["primitives"][0] for mesh in plain_doc["meshes"]}
    assert plain_doc["accessors"][plain_layers["bridges"]["attributes"]["POSITION"]]["max"][1] < 4.0  # today's look


def _glb(path):
    data = path.read_bytes()
    (json_length,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + json_length])


# --- the pavilion -----------------------------------------------------------------------------------------------


def _roof(tags=None, polygon=None, height=3.0):
    tags = {"building": "roof", "roof:shape": "hipped", **(tags or {})}
    polygon = polygon or Polygon([(0, 0), (8, 0), (8, 6), (0, 6)])
    return Building("w1", polygon, height, max(0.0, height - 0.4), "height", "#8a4b3c", "#8f8f8f", tags)


def _pavilion(**kwargs):
    walls = MeshBuilder("building_walls", uvs=True, colours=True, features=True)
    roofs = MeshBuilder("building_roofs", colours=True, features=True)
    add_pavilion(walls, roofs, _roof(**kwargs), np.zeros(2), feature=5)
    return walls.finish(), roofs.finish()


def test_only_a_shaped_open_roof_without_a_base_is_a_pavilion():
    assert roof_shape({"building": "roof", "roof:shape": "hipped"}) == "hip"
    assert roof_shape({"building": "roof", "roof:shape": "gabled"}) == "gable"
    assert roof_shape({"building": "roof", "roof:shape": "pyramidal"}) == "hip"
    assert roof_shape({"building": "roof"}) is None  # a flat slab, as before
    assert roof_shape({"building": "roof", "roof:shape": "flat"}) is None
    assert roof_shape({"building": "roof", "roof:shape": "hipped", "min_height": "3"}) is None  # it stands on something
    assert roof_shape({"building": "roof", "roof:shape": "hipped", "building:min_level": "1"}) is None
    assert roof_shape({"building": "yes", "roof:shape": "hipped"}) is None  # a house with a hipped roof keeps its walls


def test_a_hipped_pavilion_is_posts_under_a_roof_open_on_every_side():
    walls, roofs = _pavilion()
    assert (walls.features == 5).all() and (roofs.features == 5).all()
    # The roof's ridge is the building's height; the eaves are lower and overhang the footprint.
    assert _up(roofs).max() == pytest.approx(3.0, abs=1e-4)
    eave = _up(roofs)[_up(roofs) < 2.9].max()
    assert MIN_EAVE <= eave < 3.0
    assert _east(roofs).min() == pytest.approx(-OVERHANG, abs=0.01) and _east(roofs).max() == pytest.approx(8 + OVERHANG, abs=0.01)
    # Posts reach from the ground to the roof, and there are no walls between them.
    assert _up(walls).min() == pytest.approx(0.0) and _up(walls).max() < eave
    tri = walls.positions[walls.indices.reshape(-1, 3)]
    wide = np.maximum(np.ptp(tri[:, :, 0], axis=1), np.ptp(tri[:, :, 2], axis=1))
    tall = np.ptp(tri[:, :, 1], axis=1)
    assert wide[tall > 0.25].max() < 0.2  # only posts are taller than the beam: no wall between them
    assert len(walls.indices) // 3 < 700 and len(roofs.indices) // 3 < 100
    # The roof has both its top and an underside, both with normals agreeing with their winding.
    for mesh in (walls, roofs):
        tri = mesh.positions[mesh.indices.reshape(-1, 3)]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()
    roof_normals = roofs.normals[roofs.indices.reshape(-1, 3)[:, 0]]
    assert (roof_normals[:, 1] > 0.3).any() and (roof_normals[:, 1] < -0.3).any()
    assert (walls.colours == colour_bytes("#7a5a3a")).all(axis=1).any()  # bare timber unless the data names a colour


def test_a_pavilion_takes_its_post_colour_roof_height_and_orientation_from_the_data():
    walls, roofs = _pavilion(tags={"colour": "green", "roof:height": "1.0"}, polygon=Polygon([(0, 0), (6, 6), (3, 9), (-3, 3)]))
    assert (walls.colours == colour_bytes("#557a55")).all(axis=1).any()
    assert _up(roofs).max() == pytest.approx(3.0, abs=1e-4) and _up(roofs)[_up(roofs) < 2.9].max() == pytest.approx(2.0, abs=1e-3)
    # A footprint at 45 degrees (a 4.24 m by 8.49 m rectangle): the roof turns with it and overhangs it all round.
    eaves = roofs.positions[np.abs(_up(roofs) - 2.0) < 1e-3]
    outline = Polygon(np.column_stack([eaves[:, 0], -eaves[:, 2]])).minimum_rotated_rectangle
    assert outline.area == pytest.approx((4.2426 + 2 * OVERHANG) * (8.4853 + 2 * OVERHANG), rel=0.01)


def test_a_gabled_roof_has_vertical_ends_and_a_ridge_the_length_of_the_roof():
    def upright_above_the_eaves(mesh, eave):
        tri = mesh.positions[mesh.indices.reshape(-1, 3)]
        normals = mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]
        return ((np.abs(normals[:, 1]) < 1e-6) & (tri[:, :, 1].max(axis=1) > eave + 0.1)).any()

    _, roofs = _pavilion(tags={"roof:shape": "gabled"})
    eave = _up(roofs)[_up(roofs) < 2.9].max()
    assert upright_above_the_eaves(roofs, eave)  # the gable ends
    ridge = roofs.positions[_up(roofs) > 2.99]
    assert np.ptp(ridge[:, 0]) == pytest.approx(8 + 2 * OVERHANG, abs=0.01)
    _, hipped = _pavilion()
    assert not upright_above_the_eaves(hipped, eave)  # a hip has only slopes above the eaves


def test_a_square_pavilion_comes_to_a_point():
    _, roofs = _pavilion(polygon=box(0, 0, 5, 5))
    peak = roofs.positions[_up(roofs) > 2.99]
    assert np.ptp(peak[:, 0]) < 1e-6 and np.ptp(peak[:, 2]) < 1e-6
    assert np.isfinite(roofs.positions).all()


def test_the_tile_draws_the_pavilion_in_place_of_the_box(tmp_path):
    scene = Scene()
    scene.buildings.append(_roof(polygon=box(100, 100, 108, 106)))
    scene.buildings.append(Building("w2", box(150, 100, 160, 110), 8.0, 0.0, "height", "#8a4b3c", "#5a5a5c", {"building": "yes"}))
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    document = _glb(tmp_path / "tiles" / "0_0.glb")
    walls = [m for m in document["meshes"] if m["name"] == "building_walls"][0]["primitives"][0]
    count = document["accessors"][walls["attributes"]["POSITION"]]["count"]
    assert count > 16 * 2  # posts and a beam as well as the other building's four walls
    semantics = json.loads((tmp_path / "tiles" / "0_0.json").read_text(encoding="utf-8"))
    assert [b["id"] for b in semantics["buildings"]] == ["w1", "w2"]  # both are still listed, in order


# --- the dock ---------------------------------------------------------------------------------------------------


def test_only_a_pier_with_a_known_colour_is_painted():
    assert painted_pier({"man_made": "pier", "colour": "green"})
    assert painted_pier({"man_made": "pier", "colour": "#123456"})
    assert not painted_pier({"man_made": "pier"})
    assert not painted_pier({"man_made": "pier", "colour": "mauve-ish"})
    assert not painted_pier({"colour": "green"})


def test_the_painted_side_follows_the_water_beside_the_dock_and_faces_the_water():
    dock = box(40, 40, 60, 46)
    lake = box(0, 0, 100, 100).difference(dock)
    pond = box(200, 200, 220, 220)  # water elsewhere is not painted
    builder = MeshBuilder("bridges", colours=True, features=True)
    add_dock_edge(builder, [lake, pond], dock, "#557a55", -0.6, np.zeros(2), (0, 0, 256, 256), feature=2)
    mesh = builder.finish()
    assert (mesh.features == 2).all() and (mesh.colours == colour_bytes("#557a55")).all()
    assert _up(mesh).min() == pytest.approx(-0.6) and _up(mesh).max() == pytest.approx(DOCK_LIP)
    east, north = _east(mesh), _north(mesh)
    assert east.min() > 39.9 and east.max() < 60.1 and north.min() > 39.9 and north.max() < 46.1  # round the dock only
    # All four sides, each facing away from the dock.
    index = mesh.indices.reshape(-1, 3)
    centre = np.array([50.0, 43.0])
    for tri in index:
        corner = mesh.positions[tri]
        face = np.cross(corner[1] - corner[0], corner[2] - corner[0])
        middle = np.array([corner[:, 0].mean(), -corner[:, 2].mean()])
        assert face[0] * (middle[0] - centre[0]) + (-face[2]) * (middle[1] - centre[1]) > 0
    assert len(mesh.indices) // 3 == 8  # a quad each side
    # Edges on the tile boundary are skipped.
    builder = MeshBuilder("bridges", colours=True, features=True)
    add_dock_edge(builder, [lake], dock, "#557a55", -0.6, np.zeros(2), (40, 0, 256, 256), feature=2)
    assert len(builder.finish().indices) // 3 == 6


def test_a_painted_dock_is_painted_in_the_tile_and_listed(tmp_path):
    scene = Scene()
    scene.areas["water"].append(Area("w10", box(0, 0, 200, 200), {"natural": "water"}))
    scene.areas["path"].append(Area("w20", box(90, 90, 110, 96), {"man_made": "pier", "colour": "green"}, "wood"))
    scene.areas["path"].append(Area("w21", box(90, 150, 110, 156), {"man_made": "pier"}, "wood"))
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    document = _glb(tmp_path / "tiles" / "0_0.glb")
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    count = document["accessors"][layers["bridges"]["attributes"]["POSITION"]]["count"]
    assert count == 16  # four sides of the painted dock only
    semantics = json.loads((tmp_path / "tiles" / "0_0.json").read_text(encoding="utf-8"))
    assert [(o["id"], o["kind"]) for o in semantics["objects"]] == [("w20", "dock")]

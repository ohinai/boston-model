"""Street furniture the data already has: traffic signals, stop signs, post boxes, bicycle racks, bus stops, lamps on arms.

Shapes (streetfurniture.SHAPES) are checked on their own for what they face and how high they stand; placement
(street_objects) is checked on small scenes for which side of the road a pole is on and which way it looks.
"""

import json
import math

import numpy as np
import pytest
from shapely.geometry import LineString, box
from test_parks import _mesh_data, _read_glb

from boston_pipeline import streetfurniture as sf
from boston_pipeline import world
from boston_pipeline.features import Building, Object, Point, Scene, Way, building_height, classify
from boston_pipeline.furniture import add_object
from boston_pipeline.mesh import MeshBuilder, colour_bytes
from boston_pipeline.tiles import build


def _build(kind, x=10.0, y=20.0, **options):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, kind, x, y, **options)
    return builder.finish()


def _enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


def _toward(mesh, colour, origin=(10.0, 20.0)):
    """The mean horizontal position, relative to `origin`, of the vertices painted `colour` (a hex string)."""
    wanted = colour_bytes(colour)
    rows = (mesh.colours == wanted).all(axis=1)
    assert rows.any(), f"nothing is painted {colour}"
    return _enu(mesh)[rows][:, :2].mean(axis=0) - np.array(origin)


def _unit(direction):
    t = math.radians(direction)
    return np.array([math.sin(t), math.cos(t)])


def _normal(mesh, colour):
    rows = (mesh.colours == colour_bytes(colour)).all(axis=1)
    n = mesh.normals[rows].mean(axis=0)  # glTF axes: east, up, south
    return np.array([n[0], -n[2], n[1]])


STREET = ("signal", "stop_sign", "post_box", "bicycle_parking", "bus_stop")
VARIANTS = [
    ("signal", {"variant": "post"}), ("signal", {"variant": "mast", "size": 5.0}), ("signal", {"variant": "mast", "size": 11.0}),
    ("signal", {"variant": "pedestrian"}), ("signal", {"variant": "post", "direction": 217, "height": 4.2}),
    ("stop_sign", {}), ("stop_sign", {"direction": 300}), ("post_box", {}), ("post_box", {"variant": "ups", "direction": 80}),
    ("bicycle_parking", {}), ("bicycle_parking", {"size": 9}), ("bicycle_parking", {"variant": "bollard", "size": 4}),
    ("bicycle_parking", {"variant": "shed"}), ("bus_stop", {}), ("bus_stop", {"variant": "shelter+bench", "size": 1.2, "direction": 135}),
    ("bus_stop", {"variant": "shelter+flip", "direction": 20}),
]
BUDGETS = {"signal": 105, "stop_sign": 40, "post_box": 40, "bicycle_parking": 200, "bus_stop": 100}  # triangles; a rack of six hoops is the most


@pytest.mark.parametrize("kind,options", VARIANTS)
def test_faces_wind_counter_clockwise_and_normals_are_unit(kind, options):
    mesh = _build(kind, **options)
    assert len(mesh.indices) > 0 and len(mesh.indices) % 3 == 0
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-9).all(), "degenerate triangle"
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()
    normals = mesh.normals[mesh.indices.reshape(-1, 3)]
    assert np.allclose(normals[:, 0], normals[:, 1]) and np.allclose(normals[:, 0], normals[:, 2])


@pytest.mark.parametrize("kind,options", VARIANTS)
def test_stands_on_the_ground_where_it_is_put_and_carries_its_feature(kind, options):
    mesh = _build(kind, x=10.0, y=20.0, feature=4, **options)
    east, north, up = _enu(mesh).T
    assert up.min() >= -1e-6
    assert east.min() <= 10.0 <= east.max() and north.min() <= 20.0 <= north.max()
    assert np.hypot(east - 10.0, north - 20.0).max() < 14.0
    assert (mesh.features == 4).all() and (mesh.colours[:, 3] == 255).all()


@pytest.mark.parametrize("kind,options", VARIANTS)
def test_triangle_budgets_and_determinism(kind, options):
    a, b = _build(kind, seed=7, **options), _build(kind, seed=7, **options)
    assert len(a.indices) // 3 < BUDGETS[kind]
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)


def test_every_street_kind_is_known_to_add_object():
    assert set(sf.STREET_KINDS) == set(STREET)
    with pytest.raises(ValueError):
        _build("gazebo")


# --- shapes face the right way ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", [0, 90, 180, 270, 37])
@pytest.mark.parametrize("variant", ["post", "mast"])
def test_signal_lenses_face_the_direction(direction, variant):
    mesh = _build("signal", direction=direction, variant=variant, size=5.0)
    for lamp in (sf.LAMP_RED, sf.LAMP_AMBER, sf.LAMP_GREEN):
        ahead = _toward(mesh, lamp)
        assert ahead @ _unit(direction) > 0.2  # in front of the pole
        assert _normal(mesh, lamp) @ np.array([*_unit(direction), 0.0]) > 0.99
    # red above amber above green
    heights = [_enu(mesh)[(mesh.colours == colour_bytes(c)).all(axis=1)][:, 2].mean() for c in (sf.LAMP_RED, sf.LAMP_AMBER, sf.LAMP_GREEN)]
    assert heights[0] > heights[1] > heights[2]


@pytest.mark.parametrize("direction", [0, 90, 215])
def test_a_mast_arm_runs_out_to_the_right_of_the_way_the_heads_face_and_stays_out_of_reach(direction):
    mesh = _build("signal", direction=direction, variant="mast", size=6.0)
    xyz = _enu(mesh)
    relative = xyz[:, :2] - np.array([10.0, 20.0])
    right = np.array([math.cos(math.radians(direction)), -math.sin(math.radians(direction))])
    assert (relative @ right).max() == pytest.approx(6.0, abs=0.3)  # the arm reaches `size` out toward the road
    assert (relative @ right).min() > -0.2  # and nothing sticks out the other way
    away = np.hypot(*relative.T) > 0.35  # everything but the pole is overhead: 4.5 m clears a walker with room to spare
    assert xyz[away, 2].min() >= 4.5
    assert xyz[:, 2].max() < 9.0


def test_a_post_signal_head_is_above_head_height_and_a_pole_is_a_pole():
    xyz = _enu(_build("signal", variant="post"))
    away = np.hypot(xyz[:, 0] - 10.0, xyz[:, 1] - 20.0) > 0.12
    assert xyz[away, 2].min() >= 2.4  # a head: nobody walking into one
    assert xyz[:, 2].max() == pytest.approx(sf.SIGNAL_POST_HEIGHT, abs=0.01)


@pytest.mark.parametrize("direction", [0, 90, 215])
def test_pedestrian_head_looks_across_the_street(direction):
    mesh = _build("signal", variant="pedestrian", direction=direction)
    for colour in (sf.PED_HAND, sf.PED_WALKER):
        assert _toward(mesh, colour) @ _unit(direction) > 0.1
        assert _normal(mesh, colour) @ np.array([*_unit(direction), 0.0]) > 0.99


@pytest.mark.parametrize("direction", [0, 135, 300])
def test_stop_sign_face_is_red_and_looks_at_the_traffic(direction):
    mesh = _build("stop_sign", direction=direction)
    assert _normal(mesh, sf.STOP_RED) @ np.array([*_unit(direction), 0.0]) > 0.99
    assert _normal(mesh, sf.STOP_BACK) @ np.array([*_unit(direction), 0.0]) < -0.99  # from behind it is grey
    xyz = _enu(mesh)
    red = xyz[(mesh.colours == colour_bytes(sf.STOP_RED)).all(axis=1)]
    assert red[:, 2].min() >= 2.2  # a stop sign hangs above a pedestrian's head
    assert np.ptp(red[:, 2]) == pytest.approx(2 * sf.STOP_RADIUS * 0.88 * math.cos(math.pi / 8), abs=0.02)


@pytest.mark.parametrize("direction", [0, 90, 250])
def test_post_box_door_faces_the_direction(direction):
    mesh = _build("post_box", direction=direction)
    assert _normal(mesh, sf.POST_BOX_DOOR) @ np.array([*_unit(direction), 0.0]) > 0.99
    assert _toward(mesh, sf.POST_BOX_DOOR) @ _unit(direction) > 0.2
    assert _enu(mesh)[:, 2].max() == pytest.approx(sf.POST_BOX_BODY + sf.POST_BOX_DOME, abs=0.02)
    body = lambda **o: _build("post_box", **o).colours[0].astype(int)  # noqa: E731  the first thing drawn is the body
    assert np.abs(body() - colour_bytes(sf.POST_BOX_BLUE)).max() <= 3  # the default, give or take a few percent of shade
    assert np.abs(body(variant="ups") - colour_bytes(sf.POST_BOX_BROWN)).max() <= 3
    assert body(colour="#aa2222").tolist() == colour_bytes("#aa2222").tolist()  # a stated colour is exact


def test_a_hoop_stands_across_its_direction_and_hoops_follow_one_another_along_it():
    for direction in (0, 90):
        one = _enu(_build("bicycle_parking", direction=direction, size=2))
        along = _unit(direction)
        across = np.array([along[1], -along[0]])
        relative = one[:, :2] - np.array([10.0, 20.0])
        assert np.ptp(relative @ across) == pytest.approx(sf.HOOP_WIDTH, abs=0.12)  # a hoop is a flat U across
        assert np.ptp(relative @ along) < 0.1
        assert one[:, 2].max() == pytest.approx(sf.HOOP_HEIGHT, abs=0.05)
        six = _enu(_build("bicycle_parking", direction=direction, size=12))
        assert np.ptp((six[:, :2] - np.array([10.0, 20.0])) @ along) == pytest.approx(5 * sf.HOOP_SPACING, abs=0.12)
    # more bicycles than six hoops can hold stops at six
    assert len(_build("bicycle_parking", size=40).indices) == len(_build("bicycle_parking", size=12).indices)


def test_a_hoop_costs_30_triangles():
    assert [len(_build("bicycle_parking", size=2 * n).indices) // 3 for n in range(1, 7)] == [30 * n for n in range(1, 7)]


def test_bollard_racks_and_sheds_are_other_shapes():
    post = _enu(_build("bicycle_parking", variant="bollard", size=2))
    assert post[:, 2].max() == pytest.approx(sf.BOLLARD_POST + 2 * sf.BOLLARD_RING, abs=0.05)
    shed = _enu(_build("bicycle_parking", variant="shed"))
    assert shed[:, 2].max() == pytest.approx(1.3)


@pytest.mark.parametrize("direction", [0, 90, 200])
def test_bus_stop_sign_faces_left_of_the_way_to_the_road_unless_flipped(direction):
    for flags, side in (("", -1), ("flip", 1)):
        mesh = _build("bus_stop", direction=direction, variant=flags, colour=sf.BUS_BLUE)
        left = np.array([-_unit(direction)[1], _unit(direction)[0]])  # 90 degrees counter-clockwise of `direction`
        assert _normal(mesh, sf.BUS_BLUE)[:2] @ left == pytest.approx(-side, abs=0.01)
        xyz = _enu(mesh)
        assert xyz[:, 2].max() == pytest.approx(sf.BUS_POLE, abs=0.01)
        blue = xyz[(mesh.colours == colour_bytes(sf.BUS_BLUE)).all(axis=1)]
        assert blue[:, 2].min() >= 1.9  # the lower edge of the plate clears a walker's head


@pytest.mark.parametrize("direction", [0, 90, 250])
def test_shelter_stands_behind_the_pole_open_toward_the_road(direction):
    plain = _enu(_build("bus_stop", direction=direction))
    shelter = _build("bus_stop", direction=direction, variant="shelter+bench", size=1.2)
    assert len(shelter.indices) > len(_build("bus_stop", direction=direction, variant="shelter").indices) > len(_build("bus_stop").indices)
    xyz = _enu(shelter)
    roof = xyz[(shelter.colours == colour_bytes(sf.SHELTER_ROOF)).all(axis=1)]
    behind = (roof[:, :2] - np.array([10.0, 20.0])) @ _unit(direction)
    assert behind.max() < 0 and behind.min() == pytest.approx(-(0.25 + 1.2 + 0.1), abs=0.01)  # behind the pole, as deep as it was asked to be
    assert np.ptp(roof[:, 2]) <= 0.11
    glass = xyz[(shelter.colours == colour_bytes(sf.SHELTER_GLASS)).all(axis=1)]
    assert ((glass[:, :2] - np.array([10.0, 20.0])) @ _unit(direction)).max() <= -0.2  # nothing across the front
    assert plain[:, 2].max() == pytest.approx(sf.BUS_POLE, abs=0.01)


# --- placement --------------------------------------------------------------------------------------------------------
def _road(ident, *points, width=7.0, kind="residential", **tags):
    return Way(ident, LineString(points), kind, "road", width, None, dict(tags))


def _signal_node(ident, x, y, **tags):
    return Point(ident, x, y, tags={"highway": "traffic_signals", **tags})


def _scene(*ways, signals=(), objects=(), buildings=()):
    scene = Scene()
    scene.ways.extend(ways)
    scene.signals.extend(signals)
    scene.objects.extend(objects)
    scene.buildings.extend(buildings)
    return scene


def _of(things, kind):
    return [t for t in things if t.kind == kind]


NORTH = _road("w1", (0, 0), (0, 100))  # a road running north, two-way, 7 m wide: its edge is 3.5 m from the middle


@pytest.mark.parametrize(("tag", "faces", "east_of_road"), [("forward", 180.0, True), ("backward", 0.0, False)])
def test_a_directed_signal_stands_on_the_right_of_its_traffic_and_looks_at_it(tag, faces, east_of_road):
    scene = _scene(NORTH, signals=[_signal_node("n1", 0.0, 50.0, **{"traffic_signals:direction": tag})])
    (signal,) = _of(sf.street_objects(scene), "signal")
    assert signal.id == "n1" and signal.variant == "post"
    assert signal.direction == pytest.approx(faces, abs=0.5)  # `forward` traffic comes from the south, so the head looks south
    assert signal.x == pytest.approx((1 if east_of_road else -1) * (3.5 + sf.KERB), abs=0.01)  # a northbound driver's right is east
    assert signal.y == pytest.approx(50.0, abs=0.01)


def test_wide_roads_get_mast_arms_that_reach_the_middle_of_a_two_way_road_and_all_of_a_one_way_one():
    wide = _road("w2", (0, 0), (0, 100), width=12.0, kind="primary")
    scene = _scene(wide, signals=[_signal_node("n1", 0.0, 50.0, **{"traffic_signals:direction": "forward"})])
    (signal,) = _of(sf.street_objects(scene), "signal")
    assert signal.variant == "mast" and signal.size == pytest.approx(6.0 + sf.KERB, abs=0.01)
    one_way = _road("w2", (0, 0), (0, 100), width=12.0, kind="primary", oneway="yes")
    (signal,) = _of(sf.street_objects(_scene(one_way, signals=[_signal_node("n1", 0.0, 50.0, **{"traffic_signals:direction": "forward"})])), "signal")
    assert signal.size == pytest.approx(12.0 + sf.KERB, abs=0.01)


def _cross(**east_west):
    """Two streets meeting at the origin, each as two ways ending there (as OpenStreetMap has them)."""
    ways = [
        _road("n", (0, 0), (0, 100)), _road("s", (0, -100), (0, 0)),
        _road("e", (0, 0), (100, 0), **east_west), _road("w", (-100, 0), (0, 0), **east_west),
    ]
    return ways


def test_an_undirected_signal_at_a_junction_serves_every_approach():
    scene = _scene(*_cross(), signals=[_signal_node("n1", 0.0, 0.0)])
    signals = _of(sf.street_objects(scene), "signal")
    assert sorted(round(s.direction) for s in signals) == [0, 90, 180, 270]
    for s in signals:
        ahead = _unit(s.direction)  # the way the head looks: out along the road the traffic comes from
        driver_right = np.array([-ahead[1], ahead[0]])  # the driver heads along -ahead; clockwise of that is their right
        along, right = np.array([s.x, s.y]) @ ahead, np.array([s.x, s.y]) @ driver_right
        assert along == pytest.approx(0.5 * 7.0 + sf.STOP_LINE, abs=0.01)  # beyond the cross street's kerb
        assert right == pytest.approx(3.5 + sf.KERB, abs=0.01)  # on the verge to the driver's right


def test_a_one_way_street_gets_no_head_on_the_arm_that_leaves_the_junction():
    scene = _scene(*_cross(oneway="yes"), signals=[_signal_node("n1", 0.0, 0.0)])
    signals = _of(sf.street_objects(scene), "signal")
    assert sorted(round(s.direction) for s in signals) == [0, 180, 270]  # the one-way street runs east: nothing comes in from the east


def test_a_signal_on_no_road_has_no_geometry():
    scene = _scene(NORTH, signals=[_signal_node("n1", 40.0, 50.0), _signal_node("n2", 40.0, 50.0, **{"traffic_signals:direction": "forward"})])
    assert _of(sf.street_objects(scene), "signal") == []
    cycleway = Way("c", LineString([(0, 0), (0, 100)]), "cycleway", "path", 2.0, None, {})
    assert _of(sf.street_objects(_scene(cycleway, signals=[_signal_node("n3", 0.0, 50.0)])), "signal") == []
    narrow = _road("w3", (0, 0), (0, 100), width=3.5, kind="service")
    assert _of(sf.street_objects(_scene(narrow, signals=[_signal_node("n4", 0.0, 50.0)])), "signal") == []
    stub = _road("w4", (0, 0), (0, 5))
    assert _of(sf.street_objects(_scene(stub, signals=[_signal_node("n5", 0.0, 2.5)])), "signal") == []  # a stub, not an approach


def test_a_pole_is_never_put_inside_a_building_or_on_the_road():
    tags = {"traffic_signals:direction": "forward"}
    wall = Building("b", box(3.9, 0, 30, 100), 10.0, 0.0, "default", "#aaaaaa", "#555555", {})
    scene = _scene(NORTH, signals=[_signal_node("n1", 0.0, 50.0, **tags)], buildings=[wall])
    assert _of(sf.street_objects(scene), "signal") == []  # nowhere clear beside the kerb
    far = Building("b", box(6.0, 0, 30, 100), 10.0, 0.0, "default", "#aaaaaa", "#555555", {})
    (signal,) = _of(sf.street_objects(_scene(NORTH, signals=[_signal_node("n1", 0.0, 50.0, **tags)], buildings=[far])), "signal")
    assert signal.x > 3.5 + 0.3  # off the carriageway


def test_stop_signs_face_their_traffic_from_its_right_and_need_a_direction_and_a_road():
    def sign(ident, y, **tags):
        return Object(ident, 0.0, y, "stop_sign", None, None, None, None, None, None, None, tags)

    scene = _scene(NORTH, objects=[sign("a", 20.0, direction="forward"), sign("b", 30.0, direction="backward"), sign("c", 40.0), sign("d", 50.0, direction="forward")])
    scene.objects[3].x = 40.0  # on no road
    placed = {t.id: t for t in _of(sf.street_objects(scene), "stop_sign")}
    assert set(placed) == {"a", "b"}  # no direction stated, no road: no sign
    assert placed["a"].direction == pytest.approx(180.0, abs=0.5) and placed["a"].x == pytest.approx(3.5 + sf.KERB)
    assert placed["b"].direction == pytest.approx(0.0, abs=0.5) and placed["b"].x == pytest.approx(-(3.5 + sf.KERB))


def test_bus_stops_face_the_road_and_the_sign_looks_at_the_bus():
    def stop(ident, x, y, variant=None):
        return Object(ident, x, y, "bus_stop", "A stop", None, None, None, None, None, variant, {})

    scene = _scene(NORTH, objects=[stop("a", 6.0, 50.0), stop("b", -6.0, 50.0), stop("far", 60.0, 50.0)])
    placed = {t.id: t for t in _of(sf.street_objects(scene), "bus_stop")}
    assert set(placed) == {"a", "b"}  # a sign with no road to face has no meaning
    assert placed["a"].direction == pytest.approx(270.0, abs=0.5) and placed["b"].direction == pytest.approx(90.0, abs=0.5)
    assert placed["a"].name == "A stop" and "flip" not in (placed["a"].variant or "")
    # on a one-way street the sign looks at the buses that actually come: northbound they come from the south
    for oneway, flips in (("yes", False), ("-1", True)):
        road = _road("w1", (0, 0), (0, 100), oneway=oneway)
        (one,) = _of(sf.street_objects(_scene(road, objects=[stop("a", 6.0, 50.0)])), "bus_stop")
        assert ("flip" in (one.variant or "")) == flips
        mesh = _build("bus_stop", direction=one.direction, variant=one.variant, colour=sf.BUS_BLUE)
        want = np.array([0.0, -1.0] if oneway == "yes" else [0.0, 1.0])  # toward the oncoming bus
        assert _normal(mesh, sf.BUS_BLUE)[:2] @ want == pytest.approx(1.0, abs=0.01)


def test_a_shelter_is_made_shallower_or_left_out_where_a_building_stands_behind_it():
    def stop(variant="shelter+bench"):
        return Object("a", 6.0, 50.0, "bus_stop", None, None, None, None, None, None, variant, {})

    def wall(x):
        return Building("b", box(x, 0, x + 20, 100), 10.0, 0.0, "default", "#aaaaaa", "#555555", {})

    def placed(*buildings):
        (one,) = _of(sf.street_objects(_scene(NORTH, objects=[stop()], buildings=list(buildings))), "bus_stop")
        return one

    assert placed().size == 1.5 and placed().variant.startswith("shelter+bench")
    assert placed(wall(8.0)).size == 1.5  # the roof's rear edge is at 6 + 0.25 + 1.5 + 0.1 = 7.85
    assert placed(wall(7.7)).size == 1.2 and placed(wall(7.4)).size == 0.9
    nothing = placed(wall(6.9))
    assert not (nothing.variant or "").startswith("shelter") and nothing.size is None  # the sign stays, the shelter is dropped


def test_a_shelter_mapped_as_a_building_is_not_drawn_twice_and_is_not_a_tall_block():
    stop = Object("a", 6.0, 50.0, "bus_stop", None, None, None, None, None, None, "shelter+bench", {})
    mapped = Building("b", box(7.0, 47.0, 11.0, 53.0), 3.0, 0.0, "default", "#aaaaaa", "#555555", {"amenity": "shelter"})
    (one,) = _of(sf.street_objects(_scene(NORTH, objects=[stop], buildings=[mapped])), "bus_stop")
    assert one.variant == "" and one.size is None  # the sign stays; the building is the shelter
    assert building_height({"building": "yes", "amenity": "shelter", "shelter_type": "public_transport"}) == (3.0, 0.0, "default")
    assert building_height({"building": "yes", "amenity": "shelter", "height": "2.4"})[0] == 2.4
    assert building_height({"building": "yes"})[0] == 12.0  # other unmarked buildings are as they were


def test_post_boxes_look_away_from_the_road_and_racks_line_up_with_it():
    box_ = Object("p", 5.0, 50.0, "post_box", None, None, None, None, None, None, None, {})
    boxes = [box_, Object("q", 5.0, 60.0, "post_box", None, 33.0, None, None, None, None, None, {}), Object("r", 80.0, 60.0, "post_box", None, None, None, None, None, None, None, {})]
    racks = [Object("s", -5.0, 50.0, "bicycle_parking", None, None, None, 2.0, None, None, "stands", {}), Object("t", -5.0, 70.0, "bicycle_parking", None, None, None, 10.0, None, None, "stands", {})]
    placed = {t.id: t for t in sf.street_objects(_scene(NORTH, objects=[*boxes, *racks]))}
    assert placed["p"].direction == pytest.approx(90.0, abs=0.5)  # the door faces the pavement
    assert placed["q"].direction == 33.0  # the data's direction wins
    assert placed["r"].direction is None  # no road near: left as it was
    assert placed["s"].direction == pytest.approx(90.0, abs=0.5)  # hoops are in a row away from the kerb ... one hoop lies across `direction`, so its bicycles lie along the kerb
    assert placed["t"].direction == pytest.approx(0.0, abs=0.5)  # a long rack runs along the kerb
    assert all((t.x, t.y) in {(5.0, 50.0), (5.0, 60.0), (80.0, 60.0), (-5.0, 50.0), (-5.0, 70.0)} for t in placed.values())  # not moved


def test_things_inside_buildings_are_left_out():
    inside = Building("b", box(4.0, 0, 30, 100), 10.0, 0.0, "default", "#aaaaaa", "#555555", {})
    things = [Object("p", 5.0, 50.0, "post_box", None, None, None, None, None, None, None, {}), Object("s", 5.0, 40.0, "bicycle_parking", None, None, None, 2.0, None, None, None, {})]
    assert sf.street_objects(_scene(NORTH, objects=things, buildings=[inside])) == []


def test_a_lamp_on_an_arm_reaches_over_the_road_unless_it_is_told_otherwise():
    lamps = [
        Object("a", 5.0, 50.0, "lamp", None, None, 7.0, None, None, None, "arm", {}),
        Object("b", 5.0, 60.0, "lamp", None, 45.0, 7.0, None, None, None, "arm", {}),
        Object("c", 5.0, 70.0, "lamp", None, None, 5.0, None, None, None, "post", {}),
        Object("d", 90.0, 70.0, "lamp", None, None, 7.0, None, None, None, "arm", {}),
    ]
    placed = {t.id: t for t in sf.street_objects(_scene(NORTH, objects=lamps))}
    assert placed["a"].direction == pytest.approx(270.0, abs=0.5)
    assert placed["b"].direction == 45.0
    assert placed["c"].direction is None  # a post lamp has no arm to turn
    assert placed["d"].direction is None  # no road within reach


def test_signalised_crossings_get_a_head_at_each_end_looking_across():
    def crossing(ident, a, b, **tags):
        return Way(ident, LineString([a, b]), "footway", "path", 2.2, None, {"footway": "crossing", **tags})

    ways = [NORTH, crossing("c1", (-10.0, 50.0), (10.0, 50.0), crossing="traffic_signals"), crossing("c2", (-10.0, 30.0), (10.0, 30.0), crossing="marked"),
            crossing("c3", (-1.0, 10.0), (1.0, 10.0), crossing="traffic_signals"), crossing("c4", (-10.0, 70.0), (10.0, 70.0), **{"crossing:signals": "yes"})]
    heads = _of(sf.street_objects(_scene(*ways)), "signal")
    assert {h.id for h in heads} == {"c1", "c4"}  # a marked crossing has no signals, a 2 m one is a ramp
    for ident in ("c1", "c4"):
        mine = [h for h in heads if h.id == ident]
        assert sorted(round(h.direction) for h in mine) == [90, 270] and all(h.variant == "pedestrian" for h in mine)
        for h in mine:  # standing beyond its end of the crossing, looking toward the other end
            assert (h.x < -10.0) == (round(h.direction) == 90)
            assert abs(h.x) > 10.0 and abs(abs(h.y - float(ident == "c1") * 50 - float(ident == "c4") * 70)) <= sf.CROSSING_SIDE + 1e-6


def test_a_pedestrian_head_is_not_put_on_top_of_a_vehicle_pole():
    crossing = Way("c1", LineString([(-10.0, 50.0), (10.0, 50.0)]), "footway", "path", 2.2, None, {"footway": "crossing", "crossing": "traffic_signals"})
    (head, other) = _of(sf.street_objects(_scene(NORTH, crossing)), "signal")
    scene = _scene(NORTH, crossing, signals=[_signal_node("n1", head.x, head.y)])  # a signal node right at one head
    things = _of(sf.street_objects(scene), "signal")
    assert not any(t.variant == "pedestrian" and math.hypot(t.x - u.x, t.y - u.y) < sf.PED_BESIDE_POLE for t in things for u in things if u.variant != "pedestrian")
    assert len(sf.street_objects(_scene(NORTH, crossing))) == 2


def test_placement_is_deterministic_and_survives_an_empty_scene():
    scene = _scene(*_cross(), signals=[_signal_node("n1", 0.0, 0.0), _signal_node("n2", 0.0, 50.0, **{"traffic_signals:direction": "forward"})],
                   objects=[Object("p", 5.0, 20.0, "post_box", None, None, None, None, None, None, None, {})])
    key = lambda things: [(t.id, t.kind, round(t.x, 6), round(t.y, 6), t.direction, t.variant, t.size) for t in things]  # noqa: E731
    assert key(sf.street_objects(scene)) == key(sf.street_objects(scene)) and len(sf.street_objects(scene)) > 3
    assert sf.street_objects(Scene()) == []
    assert [t.id for t in sf.street_objects(_scene(objects=[Object("p", 5.0, 20.0, "post_box", None, None, None, None, None, None, None, {})]))] == ["p"]


def test_odd_geometry_draws_nothing_rather_than_something_wrong():
    zero = Way("z", LineString([(0, 0), (0, 0.2)]), "residential", "road", 7.0, None, {})  # shorter than the minimum
    scene = _scene(zero, signals=[_signal_node("n1", 0.0, 0.1)], objects=[Object("s", 1.0, 0.0, "stop_sign", None, None, None, None, None, None, None, {"direction": "forward"})])
    assert sf.street_objects(scene) == []
    ring = Way("r", LineString([(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]), "residential", "road", 7.0, None, {"junction": "roundabout"})
    sf.street_objects(_scene(ring, signals=[_signal_node("n1", 30.0, 15.0)]))  # must not raise
    degenerate = Way("d", LineString([(0, 0), (0, 0)]), "residential", "road", 7.0, None, {})
    sf.street_objects(_scene(degenerate, signals=[_signal_node("n1", 0.0, 0.0)]))


# --- from tags to tile ------------------------------------------------------------------------------------------------
def _elements(**nodes):
    out = []
    for i, (name, (x, y, tags)) in enumerate(nodes.items()):
        lon, lat = world.to_lonlat(x, y)
        out.append({"type": "node", "id": 1000 + i, "lat": lat, "lon": lon, "tags": tags})
    return out


def test_tags_become_the_right_objects():
    scene = classify(_elements(
        stop=(0, 0, {"highway": "stop", "direction": "forward"}), ups=(5, 0, {"amenity": "post_box", "brand": "UPS"}),
        usps=(10, 0, {"amenity": "post_box", "colour": "blue", "operator": "United States Postal Service"}),
        red=(15, 0, {"amenity": "post_box", "colour": "red"}),
        racks=(20, 0, {"amenity": "bicycle_parking", "bicycle_parking": "wall_loops", "capacity": "21"}),
        shelter=(25, 0, {"highway": "bus_stop", "shelter": "yes", "bench": "yes", "name": "Main St"}),
        plain=(30, 0, {"highway": "bus_stop", "bench": "yes"}),
        tall=(35, 0, {"highway": "street_lamp", "lamp_mount": "straight_mast", "height": "7"}),
        park=(40, 0, {"highway": "street_lamp", "lamp_mount": "straight_mast", "height": "5"}),
        bent=(45, 0, {"highway": "street_lamp", "lamp_mount": "bent_mast"}),
    ))
    by_id = {o.id: o for o in scene.objects}
    n = lambda i: by_id[f"n{1000 + i}"]  # noqa: E731
    assert (n(0).kind, n(0).direction) == ("stop_sign", None)  # `forward` is not a compass direction
    assert (n(1).kind, n(1).variant) == ("post_box", "ups")
    assert (n(2).kind, n(2).variant, n(2).colour) == ("post_box", None, None)  # blue is the default
    assert n(3).colour is not None
    assert (n(4).kind, n(4).variant, n(4).size) == ("bicycle_parking", "wall_loops", 21.0)
    assert (n(5).kind, n(5).variant, n(5).name) == ("bus_stop", "shelter+bench", "Main St")
    assert n(6).variant is None  # a bench is drawn only inside a shelter
    assert n(7).variant == "arm" and n(8).variant == "post" and n(9).variant == "arm"


def test_a_tile_lists_and_draws_the_street_furniture(tmp_path):
    scene = _scene(
        _road("w1", (150.0, 100.0), (150.0, 300.0), width=12.0, kind="primary"),
        signals=[_signal_node("n77", 150.0, 200.0, **{"traffic_signals:direction": "forward"})],
        objects=[Object("n78", 160.0, 200.0, "post_box", None, None, None, None, None, None, None, {}),
                 Object("n79", 140.0, 220.0, "bus_stop", "Stop", None, None, None, None, None, "shelter", {})],
    )
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    data = json.loads((tmp_path / "tiles" / "0_0.json").read_text())
    kinds = {o["id"]: o["kind"] for o in data["objects"]}
    assert kinds == {"n77": "signal", "n78": "post_box", "n79": "bus_stop"}
    assert data["signals"] == [[150.0, 200.0]]  # the signal node's own position is still listed for clients
    pole = next(o for o in data["objects"] if o["id"] == "n77")
    assert pole["pos"][0] > 156.0  # a pole beside the carriageway, not on its middle line
    glb = tmp_path / "tiles" / "0_0.glb"
    document = _read_glb(glb)
    (furniture,) = [mesh["primitives"][0] for mesh in document["meshes"] if mesh["name"] == "furniture"]
    positions, _ = _mesh_data(glb, document, furniture)
    assert positions[:, 1].max() > 6.0  # the mast
    assert len(positions) > 100


def test_every_pole_of_a_junction_is_registered_in_its_tile(tmp_path):
    scene = _scene(*_cross(), signals=[_signal_node("n1", 150.0, 150.0)])
    for way in scene.ways:
        way.line = LineString([(x + 150.0, y + 150.0) for x, y in way.line.coords])
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    data = json.loads((tmp_path / "tiles" / "0_0.json").read_text())
    assert [o["kind"] for o in data["objects"]] == ["signal"] * 4
    assert len({tuple(o["pos"]) for o in data["objects"]}) == 4


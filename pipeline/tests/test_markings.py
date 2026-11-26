"""Road paint: crosswalks, centre lines, lane lines, stop lines, and the asphalt under them."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point as ShapelyPoint, box
from shapely.ops import unary_union

from boston_pipeline import markings, world
from boston_pipeline.features import Point, Scene, Way, classify
from boston_pipeline.markings import MARK_LIFT, PATCH_DARK, PATCH_LIGHT, WHITE, YELLOW, crossing_style
from boston_pipeline.tiles import ROAD_Z, PolygonIndex, build

from test_parks import _mesh_data, _read_glb


def _road(ident, points, kind="secondary", width=7.0, **tags):
    return Way(ident, LineString(points), kind, "road", width, None, {"highway": kind, **tags})


def _crossing(ident, points, markings_tag="ladder", crossing="marked", **tags):
    tags = {"highway": "footway", "footway": "crossing", "crossing": crossing, **tags}
    if markings_tag is not None:
        tags["crossing:markings"] = markings_tag
    return Way(ident, LineString(points), "footway", "path", 2.2, None, tags)


def _plan(*ways, nodes=()):
    scene = Scene()
    scene.ways = list(ways)
    scene.roadnodes = list(nodes)
    polygons = [w.line.buffer(w.width / 2, quad_segs=3) for w in ways if w.surface == "road"]
    return scene, markings.plan(scene, PolygonIndex(polygons))


def _of(plan, colour):
    return [p for p, c in zip(plan.polygons, plan.colours) if c == colour]


def _paint(plan):
    """Everything that is lane paint or crosswalk paint, not asphalt patches."""
    return [p for p, c in zip(plan.polygons, plan.colours) if c in (WHITE, YELLOW)]


def _spread(polygons):
    return unary_union(polygons)


# A way 120 m long, going east, with the junction in its middle.
EAST = [(-60.0, 0.0), (60.0, 0.0)]
ACROSS = [(0.0, -9.0), (0.0, 9.0)]


# --- crossings ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tags", "style"),
    [
        ({"crossing": "marked", "crossing:markings": "ladder"}, "ladder"),
        ({"crossing": "traffic_signals", "crossing:markings": "ladder:skewed"}, "ladder"),
        ({"crossing": "marked", "crossing:markings": "zebra"}, "zebra"),
        ({"crossing": "zebra"}, "zebra"),
        ({"crossing": "marked", "crossing:markings": "lines"}, "lines"),
        ({"crossing": "marked", "crossing:markings": "dashes"}, "lines"),
        ({"crossing": "marked", "crossing:markings": "yes"}, "lines"),
        ({"crossing": "marked"}, "lines"),
        ({"crossing": "traffic_signals"}, "lines"),
        ({"crossing": "uncontrolled"}, "lines"),
        ({"crossing": "marked", "crossing:markings": "ladder;surface"}, "ladder"),
        ({"crossing": "unmarked", "crossing:markings": "no"}, None),
        ({"crossing": "unmarked"}, None),
        ({"crossing": "informal"}, None),
        ({"crossing": "marked", "crossing:markings": "no"}, None),
        ({"crossing": "uncontrolled", "crossing:markings": "surface"}, None),
        ({"crossing": "marked", "crossing:markings": "dots"}, None),
        ({"crossing": "unmarked", "crossing:markings": "ladder"}, None),  # the data contradicts itself: draw nothing
        ({}, None),
    ],
)
def test_crossing_style_from_tags(tags, style):
    assert crossing_style(tags) == style


def test_a_ladder_is_painted_across_the_carriageway_and_no_further():
    road = _road("w1", EAST, "residential", 7.0, lanes="1", oneway="yes")  # a street with no lines of its own
    _, plan = _plan(road, _crossing("w2", ACROSS, "ladder"))
    paint = _spread(_paint(plan))
    assert _paint(plan) and set(plan.colours) <= {WHITE, PATCH_DARK, PATCH_LIGHT}  # white paint (and maybe a repair patch)
    # Cut to the carriageway (y within 3.5 m of the road's centre line), not the pavement beyond it.
    assert paint.bounds[1] >= -3.5 - 1e-6 and paint.bounds[3] <= 3.5 + 1e-6
    assert paint.bounds[1] == pytest.approx(-3.5, abs=0.3) and paint.bounds[3] == pytest.approx(3.5, abs=0.3)
    # It is 3 m wide, along the road: x within 1.5 m of the crossing.
    assert paint.bounds[0] == pytest.approx(-1.5, abs=0.05) and paint.bounds[2] == pytest.approx(1.5, abs=0.05)
    # Two edge lines and rungs between them: more than two pieces.
    assert len(_paint(plan)) > 6


def test_styles_differ_in_how_much_paint_they_use():
    road = _road("w1", EAST, "residential", 7.0, lanes="1", oneway="yes")
    counts = {}
    for style in ("ladder", "zebra", "lines"):
        _, plan = _plan(road, _crossing("w2", ACROSS, style))
        counts[style] = len(_paint(plan))
    assert counts["lines"] == 2  # just the two edge lines
    assert counts["zebra"] >= 6 and counts["ladder"] >= 8  # bars across a 7 m street; the ladder adds its two edge lines
    assert counts["ladder"] > counts["lines"] + 4
    # A zebra's bars reach across the whole width; a ladder's stop short of its edge lines.
    _, zebra = _plan(road, _crossing("w2", ACROSS, "zebra"))
    assert _spread(_paint(zebra)).bounds[0] == pytest.approx(-1.5, abs=0.05)


@pytest.mark.parametrize(
    "tags",
    [
        {"crossing": "unmarked", "crossing:markings": "no"},
        {"crossing": "marked", "crossing:markings": "no"},
        {"crossing": "informal"},
        {"crossing": "uncontrolled", "crossing:markings": "surface"},
    ],
)
def test_nothing_is_painted_on_an_unmarked_crossing(tags):
    road = _road("w1", EAST, "residential", 7.0, lanes="1", oneway="yes")
    crossing = Way("w2", LineString(ACROSS), "footway", "path", 2.2, None, {"highway": "footway", "footway": "crossing", **tags})
    _, plan = _plan(road, crossing)
    assert _paint(plan) == []


def test_a_crossing_over_a_service_road_or_no_road_gets_nothing():
    service = Way("w1", LineString(EAST), "service", "road", 4.0, None, {"highway": "service"})
    _, plan = _plan(service, _crossing("w2", ACROSS))
    assert plan.polygons == []
    _, plan = _plan(_road("w1", EAST, "residential", 7.0, lanes="1", oneway="yes"), _crossing("w2", [(0.0, 50.0), (0.0, 70.0)]))  # far from the road
    assert _paint(plan) == []


def test_a_crossing_that_runs_along_the_road_is_not_painted():
    road = _road("w1", EAST, "residential", 9.0, lanes="1", oneway="yes")
    _, plan = _plan(road, _crossing("w2", [(-20.0, 0.5), (20.0, 0.5)]))
    assert _paint(plan) == []


def test_a_crossing_node_with_no_way_is_drawn_across_the_road():
    road = _road("w1", EAST, "residential", 7.0, lanes="1", oneway="yes")
    node = Point("n1", 10.0, 0.0, kind="crossing", tags={"highway": "crossing", "crossing": "marked", "crossing:markings": "zebra"})
    _, plan = _plan(road, nodes=[node])
    paint = _spread(_paint(plan))
    assert paint.bounds[1] == pytest.approx(-3.5, abs=0.3) and paint.bounds[3] == pytest.approx(3.5, abs=0.3)
    assert paint.bounds[0] == pytest.approx(8.5, abs=0.1) and paint.bounds[2] == pytest.approx(11.5, abs=0.1)
    # ...but a node that belongs to a crossing way is not drawn a second time.
    _, again = _plan(road, _crossing("w2", [(10.0, -9.0), (10.0, 9.0)], "zebra"), nodes=[node])
    _, once = _plan(road, _crossing("w2", [(10.0, -9.0), (10.0, 9.0)], "zebra"))
    assert len(again.polygons) == len(once.polygons)
    # An unmarked node is nothing.
    quiet = Point("n2", 10.0, 0.0, kind="crossing", tags={"highway": "crossing", "crossing": "unmarked"})
    assert _paint(_plan(road, nodes=[quiet])[1]) == []


def test_a_way_takes_its_style_from_its_node_when_it_has_none():
    road = _road("w1", EAST, "residential", 7.0, lanes="1", oneway="yes")
    bare = Way("w2", LineString(ACROSS), "footway", "path", 2.2, None, {"highway": "footway", "footway": "crossing"})
    node = Point("n1", 0.0, 0.0, kind="crossing", tags={"crossing": "marked", "crossing:markings": "ladder"})
    assert len(_paint(_plan(road, bare, nodes=[node])[1])) > 6  # a ladder, from the node
    assert _paint(_plan(road, bare)[1]) == []  # no say, no paint


# --- stop lines -----------------------------------------------------------------------------------------------
def test_a_signalised_crossing_on_a_one_way_street_has_a_stop_line_before_it():
    road = _road("w1", EAST, "primary", 7.0, lanes="2", oneway="yes")
    _, plan = _plan(road, _crossing("w2", ACROSS, "ladder", crossing="traffic_signals"))
    stops = [p for p in _paint(plan) if p.bounds[2] < -2.5 and abs(p.bounds[3] - p.bounds[1]) > 5]
    assert len(stops) == 1  # one way: one stop line, on the side traffic comes from
    stop = stops[0]
    # 3 m crosswalk, 1.2 m clear, then the line: x from -3.2 to -2.7. It spans the two lanes.
    assert stop.bounds[0] == pytest.approx(-3.2, abs=0.05) and stop.bounds[2] == pytest.approx(-2.7, abs=0.05)
    assert stop.bounds[1] == pytest.approx(-3.2 + 0.12, abs=0.2) and stop.bounds[3] == pytest.approx(3.2 - 0.12, abs=0.2)
    assert not [p for p in _paint(plan) if p.bounds[0] > 2.0 and abs(p.bounds[3] - p.bounds[1]) > 5]  # none on the far side


def test_a_two_way_street_has_a_stop_line_on_each_side_across_its_own_lanes():
    road = _road("w1", EAST, "secondary", 7.0, lanes="2")
    _, plan = _plan(road, _crossing("w2", ACROSS, "ladder", crossing="traffic_signals"))
    long = [p for p in _paint(plan) if abs(p.centroid.x) > 2.5 and (p.bounds[3] - p.bounds[1]) > 2.5]
    assert len(long) == 2
    west = next(p for p in long if p.centroid.x < 0)  # eastbound traffic: on the south (right) half
    east = next(p for p in long if p.centroid.x > 0)  # westbound traffic: on the north half
    assert west.centroid.y < 0 < east.centroid.y
    assert west.bounds[3] < 0.05 and east.bounds[1] > -0.05  # neither crosses the centre line


def test_an_unsignalised_crossing_has_no_stop_line_but_a_stop_sign_gives_one():
    road = _road("w1", EAST, "primary", 7.0, lanes="2", oneway="yes")
    crossing = _crossing("w2", ACROSS, "ladder")
    _, plan = _plan(road, crossing)
    assert not [p for p in _paint(plan) if p.bounds[2] < -2.5 and abs(p.bounds[3] - p.bounds[1]) > 5]
    stop = Point("n1", -9.0, -1.5, kind="stop", tags={"highway": "stop"})  # on the road, before the crosswalk
    _, with_sign = _plan(road, crossing, nodes=[stop])
    assert len([p for p in _paint(with_sign) if p.bounds[2] < -2.5 and abs(p.bounds[3] - p.bounds[1]) > 5]) == 1
    behind = Point("n2", 9.0, -1.5, kind="stop", tags={"highway": "stop"})  # after it: stops the other way, not this one
    _, after = _plan(road, crossing, nodes=[behind])
    assert not [p for p in _paint(after) if p.bounds[2] < -2.5 and abs(p.bounds[3] - p.bounds[1]) > 5]


# --- lines along the road -------------------------------------------------------------------------------------
def test_a_two_way_street_has_a_double_yellow_centre_line():
    _, plan = _plan(_road("w1", EAST, "secondary", 7.0, lanes="2"))
    yellow = _of(plan, YELLOW)
    assert len(yellow) == 2  # two lines, 0.15 m wide with 0.13 m between
    assert sorted(round(p.centroid.y, 2) for p in yellow) == [-0.14, 0.14]
    assert all(p.bounds[2] - p.bounds[0] > 110 for p in yellow)  # unbroken along the street
    assert _of(plan, WHITE) == []  # one lane each way: no lane lines


def test_a_one_way_street_with_three_lanes_has_two_lines_of_dashes():
    _, plan = _plan(_road("w1", EAST, "primary", 10.2, lanes="3", oneway="yes"))
    assert _of(plan, YELLOW) == []
    dashes = _of(plan, WHITE)
    assert 18 <= len(dashes) <= 28
    assert {round(p.centroid.y, 1) for p in dashes} == {-1.6, 1.6}  # a lane (3.2 m) apart, symmetric
    assert all(2.9 <= p.bounds[2] - p.bounds[0] <= 3.1 for p in dashes)  # 3 m dashes
    # The pattern is 3 m of paint and 6 m of road.
    row = sorted(p.bounds[0] for p in dashes if p.centroid.y > 0)
    assert np.diff(row) == pytest.approx(9.0, abs=0.01)


def test_the_lines_of_a_four_lane_street_are_a_centre_line_and_dashes_each_side():
    _, plan = _plan(_road("w1", EAST, "primary", 13.4, lanes="4"))
    assert len(_of(plan, YELLOW)) == 2
    assert {round(p.centroid.y, 1) for p in _of(plan, WHITE)} == {-3.2, 3.2}


def test_the_centre_line_lies_in_the_travel_lanes_when_there_is_parking_on_one_side():
    road = _road("w1", EAST, "secondary", 9.2, lanes="2", **{"parking:right": "parallel"})
    _, plan = _plan(road)
    middle = {round(p.centroid.y, 1) for p in _of(plan, YELLOW)}
    assert middle == {round(1.1 - 0.14, 1), round(1.1 + 0.14, 1)}  # moved to the side without parking


def test_nothing_on_roads_that_are_not_marked():
    for road in (
        Way("w1", LineString(EAST), "service", "road", 4.0, None, {"highway": "service", "lanes": "2"}),
        _road("w2", EAST, "secondary", 4.5, lanes="2"),  # under 5 m wide
        _road("w3", EAST, "secondary", 7.0, lanes="3"),  # an odd number of lanes: which way the odd one goes is a guess
        _road("w4", EAST, "secondary", 7.0, lanes="2", surface="gravel"),
        _road("w5", EAST, "secondary", 7.0, lanes="2", junction="roundabout", oneway="yes"),
        _road("w6", EAST, "residential", 7.0),  # a side street that does not say it has two lanes
        _road("w7", EAST, "residential", 6.0, lanes="1", oneway="yes"),
        _road("w8", EAST, "primary", 3.8, lanes="1", oneway="yes"),
        _road("w9", EAST, "tertiary", 3.8, lanes="1", oneway="yes"),
    ):
        assert _paint(_plan(road)[1]) == [], road.id
    # A residential street that says it has two lanes does get a centre line.
    assert len(_of(_plan(_road("w1", EAST, "residential", 9.2, lanes="2"))[1], YELLOW)) == 2


def test_lines_stop_at_a_junction_and_the_through_road_keeps_its_own():
    # A T junction at the origin: the through road is two ways meeting there, the side street ends on it.
    west = _road("w1", [(-100.0, 0.0), (0.0, 0.0)], "secondary", 7.0, lanes="2")
    east = _road("w2", [(0.0, 0.0), (100.0, 0.0)], "secondary", 7.0, lanes="2")
    side = _road("w3", [(0.0, 0.0), (0.0, 100.0)], "secondary", 7.0, lanes="2")
    _, plan = _plan(west, east, side)
    yellow = _of(plan, YELLOW)
    through = [p for p in yellow if abs(p.centroid.y) < 1 and abs(p.bounds[3] - p.bounds[1]) < 0.2]
    along_side = [p for p in yellow if p.centroid.y > 1]
    assert through and along_side
    assert unary_union(through).distance(ShapelyPoint(0.0, 0.14)) < 1e-6  # unbroken across the junction
    assert min(p.bounds[1] for p in along_side) == pytest.approx(3.5 + 1.0, abs=0.01)  # stops short of the edge of the through road


def test_where_two_roads_of_the_same_class_cross_all_four_arms_stop():
    arms = [
        _road("w1", [(-100.0, 0.0), (0.0, 0.0)], "secondary", 7.0, lanes="2"),
        _road("w2", [(0.0, 0.0), (100.0, 0.0)], "secondary", 7.0, lanes="2"),
        _road("w3", [(0.0, -100.0), (0.0, 0.0)], "secondary", 7.0, lanes="2"),
        _road("w4", [(0.0, 0.0), (0.0, 100.0)], "secondary", 7.0, lanes="2"),
    ]
    _, plan = _plan(*arms)
    near = box(-4.4, -4.4, 4.4, 4.4)
    assert not any(p.intersects(near) for p in _of(plan, YELLOW))
    # A bend (two arms only) is no junction.
    _, bend = _plan(_road("w1", [(-100.0, 0.0), (0.0, 0.0)], "secondary", 7.0, lanes="2"), _road("w2", [(0.0, 0.0), (0.0, 100.0)], "secondary", 7.0, lanes="2"))
    assert any(p.intersects(box(-1, -1, 1, 1)) for p in _of(bend, YELLOW))


def test_lines_stop_short_of_a_crosswalk():
    _, plan = _plan(_road("w1", EAST, "secondary", 7.0, lanes="2"), _crossing("w2", ACROSS, "ladder"))
    assert not any(p.intersects(box(-2.4, -3.5, 2.4, 3.5)) for p in _of(plan, YELLOW))
    assert any(p.bounds[2] < -2.4 for p in _of(plan, YELLOW)) and any(p.bounds[0] > 2.4 for p in _of(plan, YELLOW))


# --- the whole plan ---------------------------------------------------------------------------------------------
def test_repair_patches_are_on_main_streets_only_and_stay_inside_them():
    main = _road("w1", [(0.0, 0.0), (900.0, 0.0)], "primary", 7.0, lanes="2", oneway="yes")
    side = _road("w2", [(0.0, 50.0), (900.0, 50.0)], "residential", 7.0, lanes="2")
    _, plan = _plan(main, side)
    patches = [p for p, c in zip(plan.polygons, plan.colours) if c in (PATCH_DARK, PATCH_LIGHT)]
    assert 8 <= len(patches) <= 16  # a site every 55 m, 40 of 100 with a patch
    strip = main.line.buffer(3.5 + 1e-6)
    assert all(strip.contains(p) for p in patches)  # none on the side street, none hanging over the kerb
    # A patch is a few metres across at most; a trench runs across the street.
    assert all(max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) <= 7.01 for p in patches)


def test_the_plan_is_repeatable():
    ways = (
        _road("w1", EAST, "primary", 10.2, lanes="3", oneway="yes"),
        _road("w2", [(0.0, -80.0), (0.0, 80.0)], "secondary", 7.0, lanes="2"),
        _crossing("w3", [(30.0, -9.0), (30.0, 9.0)], "ladder", crossing="traffic_signals"),
    )
    a, b = _plan(*ways)[1], _plan(*ways)[1]
    assert [p.wkb for p in a.polygons] == [p.wkb for p in b.polygons] and a.colours == b.colours
    assert a.polygons  # and it did draw something
    patches = [p.wkb for p, c in zip(a.polygons, a.colours) if c in (PATCH_DARK, PATCH_LIGHT)]
    assert patches == [p.wkb for p, c in zip(b.polygons, b.colours) if c in (PATCH_DARK, PATCH_LIGHT)]


def test_odd_geometry_draws_nothing_and_does_not_fail():
    empty = markings.plan(Scene(), PolygonIndex([]))
    assert empty.polygons == [] and empty.tree is None
    road = _road("w1", EAST, "secondary", 7.0, lanes="2")
    odd = [
        _road("w2", [(5.0, 5.0), (5.0, 5.0)], "primary", 7.0, lanes="2"),  # no length
        _road("w3", [(0.0, 0.0), (0.0, 0.2)], "primary", 7.0, lanes="2"),  # a stub
        _crossing("w4", [(3.0, 3.0), (3.0, 3.0)]),
        _crossing("w5", [(0.0, 1.0), (0.0, 1.5)]),  # a crossing too short to be one
        _crossing("w6", [(0.0, -9.0), (0.0, 0.0), (0.0, -9.0)]),  # doubles back on itself
        Way("w7", LineString([(0.0, 0.0), (1.0, 1.0)]), "footway", "path", 2.2, None, {"footway": "crossing"}),
    ]
    _, plan = _plan(road, *odd)
    assert all(p.is_valid and not p.is_empty for p in plan.polygons)
    # The road's own lines are untouched by the junk.
    assert len(_of(plan, YELLOW)) >= 2


def test_every_mark_lies_on_the_road_it_belongs_to():
    road = _road("w1", EAST, "secondary", 7.0, lanes="2")
    _, plan = _plan(road, _crossing("w2", ACROSS, "ladder", crossing="traffic_signals"))
    strip = road.line.buffer(3.5 + 1e-6)
    assert all(strip.contains(p) for p in _paint(plan))


def test_classify_keeps_crossing_and_stop_nodes():
    lon, lat = (float(v) for v in world.to_lonlat(5.0, 5.0))
    scene = classify([
        {"type": "node", "id": 1, "lon": lon, "lat": lat, "tags": {"highway": "crossing", "crossing": "marked"}},
        {"type": "node", "id": 2, "lon": lon, "lat": lat, "tags": {"highway": "stop"}},
        {"type": "node", "id": 3, "lon": lon, "lat": lat, "tags": {"highway": "traffic_signals"}},
    ])
    assert sorted(p.kind for p in scene.roadnodes) == ["crossing", "stop"]
    assert len(scene.signals) == 1  # a signal is still a signal


# --- the asphalt ------------------------------------------------------------------------------------------------
def test_asphalt_is_dark_warm_and_a_function_of_position():
    x, y = np.meshgrid(np.linspace(-500, 500, 41), np.linspace(-500, 500, 41))
    x, y = x.ravel(), y.ravel()
    colours = markings.asphalt_at(x, y)
    assert colours.dtype == np.uint8 and colours.shape == (len(x), 4) and (colours[:, 3] == 255).all()
    mean = colours[:, :3].astype(float).mean(axis=0)
    assert mean[0] > mean[2]  # warmer than blue
    assert mean.max() < 80  # linear RGBA bytes: dark (a mid grey would be about 120 here)
    assert np.array_equal(markings.asphalt_at(x, y), colours)
    # It varies, but gently: the lightest place is under twice as light as the darkest.
    luminance = colours[:, :3].astype(float).sum(axis=1)
    assert 1.1 < luminance.max() / luminance.min() < 2.0
    # The same place gets the same colour whichever tile asks.
    assert np.array_equal(markings.asphalt_at(np.array([256.0]), np.array([12.5])), markings.asphalt_at(np.array([256.0]), np.array([12.5])))


# --- in a tile ---------------------------------------------------------------------------------------------------
def test_a_built_tile_has_a_paint_layer_just_above_the_road(tmp_path):
    scene = Scene()
    scene.ways = [
        _road("w1", [(10.0, 128.0), (250.0, 128.0)], "secondary", 7.0, lanes="2"),
        _crossing("w2", [(100.0, 118.0), (100.0, 138.0)], "ladder", crossing="traffic_signals"),
    ]
    stats = build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    assert stats["triangles"] > 0
    document = _read_glb(tmp_path / "tiles" / "0_0.glb")
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    assert {"roads", "markings"} <= set(layers)
    for name in ("roads", "markings"):
        assert "COLOR_0" in layers[name]["attributes"] and "NORMAL" not in layers[name]["attributes"]
    positions, triangles = _mesh_data(tmp_path / "tiles" / "0_0.glb", document, layers["markings"])
    heights = positions[:, 1]  # glTF y is up
    assert heights.max() == pytest.approx(ROAD_Z + MARK_LIFT, abs=1e-4)  # the paint
    assert heights.min() >= ROAD_Z + 0.01  # and nothing sunk into the road
    # glTF z is minus north: all of it within the 7 m carriageway.
    assert (-positions[:, 2]).min() >= 128.0 - 3.5 - 1e-3 and (-positions[:, 2]).max() <= 128.0 + 3.5 + 1e-3
    # Paint is facing up, so the viewer shows its top.
    corners = positions[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert (normals[:, 1] > 0).all()
    # The road itself is coloured per vertex and ends the same height as before.
    road, _ = _mesh_data(tmp_path / "tiles" / "0_0.glb", document, layers["roads"])
    assert road[:, 1].max() == pytest.approx(ROAD_Z, abs=1e-4)
    # A tile whose only road is an alley has no paint, and no empty paint layer.
    bare = Scene()
    bare.ways = [Way("w1", LineString([(10.0, 128.0), (250.0, 128.0)]), "service", "road", 3.5, None, {"highway": "service"})]
    build(bare, [(0, 0)], tmp_path / "bare", {"name": "Test"}, None)
    names = {mesh["name"] for mesh in _read_glb(tmp_path / "bare" / "tiles" / "0_0.glb")["meshes"]}
    assert "roads" in names and "markings" not in names

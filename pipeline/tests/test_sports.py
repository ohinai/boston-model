"""Pitches: lines, goals, hoops, nets, bases and backstops (sports.py)."""

import json
import math

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from boston_pipeline import extras, sports, world
from boston_pipeline.features import Area, Barrier, Building, Scene, Way, green_cover
from boston_pipeline.furniture import add_object
from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.sports import DIRT, LINE_COLOUR, LINE_LIFT
from boston_pipeline.tiles import build

from test_parks import _mesh_data, _read_glb


def _pitch(ident, polygon, sport, cover="pitch", **tags):
    return Area(ident, polygon, {"leisure": "pitch", "sport": sport, **tags}, cover)


def _court(cx, cy, length, width, angle=0.0):
    """A rectangle `length` along x and `width` along y, centred on (cx, cy), turned `angle` degrees counter-clockwise."""
    return rotate(box(cx - length / 2, cy - width / 2, cx + length / 2, cy + width / 2), angle, origin=(cx, cy))


def _plan(*areas, **extra):
    scene = Scene()
    scene.areas["green"].extend(areas)
    for name, value in extra.items():
        setattr(scene, name, value)
    return scene, sports.plan(scene)


def _paint(plan):
    return unary_union(plan.polygons) if plan.polygons else Polygon()


def _kinds(plan, kind):
    return [o for o in plan.objects if o.kind == kind]


def _dims(geom):
    rect = geom.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    a, b = math.hypot(xs[1] - xs[0], ys[1] - ys[0]), math.hypot(xs[2] - xs[1], ys[2] - ys[1])
    return max(a, b), min(a, b)


def _fan(home, heading, radius=70.0, half=45.0):
    """A sector of a circle round `home` (the outline of a ball field's outfield), turned to `heading` degrees clockwise from north."""
    pts = [home]
    for step in range(0, 61):
        a = math.radians(heading - half + 2 * half * step / 60)
        pts.append((home[0] + radius * math.sin(a), home[1] + radius * math.cos(a)))
    return Polygon(pts)


def _ll(x, y):
    lon, lat = world.to_lonlat(x, y)
    return f"{lat:.7f},{lon:.7f}"


# --- what is a pitch ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tag", "expected"),
    [("tennis", ["tennis"]), ("soccer;lacrosse", ["soccer", "lacrosse"]), (" Soccer ; softball ", ["soccer", "softball"]),
     ("multi", []), ("multi;baseball", []), ("a;b;c;d", []), ("", []), (None, [])],
)
def test_the_sports_of_a_pitch(tag, expected):
    assert sports.pitch_sports({} if tag is None else {"sport": tag}) == expected


def test_the_modules_hooks_are_found_by_extras():
    assert extras.shape("hoop") is sports.SHAPES["hoop"]
    assert {"hoop", "goal", "goalpost", "net", "backstop"} <= set(extras.kinds())
    scene, plan = _plan(_pitch("w1", _court(100, 100, 23.77, 10.97), "tennis"))
    assert extras.objects(scene) == plan.objects and plan.objects


@pytest.mark.parametrize(
    "tags",
    [{"sport": "multi"}, {"covered": "yes"}, {"location": "rooftop"}, {"sport": "a;b;c;d"}, {"sport": "tennis", "leisure": "park"}, {"description": "bullpen"}],
)
def test_a_pitch_that_is_not_for_one_game_or_is_covered_is_bare(tags):
    area = Area("w1", _court(100, 100, 23.77, 10.97), {"leisure": "pitch", "sport": "tennis", **tags}, "pitch")
    _, plan = _plan(area)
    assert not plan.polygons and not plan.objects and not plan.nets


# --- tennis ------------------------------------------------------------------------------------------------------
def test_a_tennis_court_gets_lines_a_net_and_two_posts():
    polygon = _court(100, 100, 23.77, 10.97)
    _, plan = _plan(_pitch("w1", polygon, "tennis"))
    paint = _paint(plan)
    assert set(plan.colours) == {LINE_COLOUR} and set(plan.lifts) == {LINE_LIFT}
    assert polygon.buffer(0.01).contains(paint)  # inside the polygon
    assert paint.bounds == pytest.approx(polygon.bounds, abs=0.02)  # the baselines and doubles sidelines are its edge
    # Singles sidelines, two service lines, the centre service line and the centre marks make a recognisable court.
    assert 8 < len(plan.polygons) or paint.area > 6.0
    assert plan.polygons and paint.area == pytest.approx(0.08 * (2 * (23.77 + 10.97) + 2 * 23.77 + 2 * 8.23 + 12.8 + 0.4), rel=0.15)
    (net,) = plan.nets
    assert math.hypot(net[2] - net[0], net[3] - net[1]) == pytest.approx(10.97 + 2 * 0.914, abs=0.01)
    assert net[4] == pytest.approx(0.914)
    assert (net[0] + net[2]) / 2 == pytest.approx(100.0) and (net[1] + net[3]) / 2 == pytest.approx(100.0)  # at the centre line
    (posts,) = _kinds(plan, "net")
    assert (posts.x, posts.y) == pytest.approx((100.0, 100.0)) and posts.size == pytest.approx(12.8, abs=0.01)
    assert posts.height == pytest.approx(1.07) and posts.direction == pytest.approx(90.0)  # facing along the court's long side


def test_a_turned_court_is_turned_and_its_net_runs_across_it():
    polygon = _court(100, 100, 23.77, 10.97, angle=30.0)
    _, plan = _plan(_pitch("w1", polygon, "tennis"))
    length, width = _dims(_paint(plan))
    assert (length, width) == pytest.approx((23.77, 10.97), abs=0.1)
    (net,) = plan.nets
    along = math.degrees(math.atan2(net[3] - net[1], net[2] - net[0])) % 180
    assert along == pytest.approx(120.0, abs=0.5)  # across a court whose long side points at 30 degrees


def test_several_courts_in_one_polygon_and_far_too_small_a_polygon_are_bare():
    assert not _plan(_pitch("w1", _court(100, 100, 48, 24), "tennis"))[1].polygons
    assert not _plan(_pitch("w1", _court(100, 100, 12, 5.5), "tennis"))[1].polygons
    small = _plan(_pitch("w1", _court(100, 100, 18.5, 8.6), "tennis"))[1]  # about 78%: scaled down, not left bare
    assert small.polygons and _court(100, 100, 18.5, 8.6).buffer(0.01).contains(_paint(small))


# --- basketball --------------------------------------------------------------------------------------------------
def test_a_full_basketball_court_has_a_hoop_at_each_end_facing_in():
    polygon = _court(100, 100, 25.6, 15.3)
    _, plan = _plan(_pitch("w1", polygon, "basketball", "court", hoops="2"))
    hoops = sorted(_kinds(plan, "hoop"), key=lambda o: o.x)
    assert len(hoops) == 2
    assert [round(h.direction) for h in hoops] == [90, 270]  # west end faces east, east end faces west
    assert hoops[0].x == pytest.approx(100 - 12.8 + 1.22, abs=0.01) and hoops[1].x == pytest.approx(100 + 12.8 - 1.22, abs=0.01)
    assert all(h.y == pytest.approx(100.0, abs=0.01) for h in hoops)
    paint = _paint(plan)
    assert polygon.buffer(0.01).contains(paint)
    # The centre circle is a ring on the halfway line, and each end has a key with a free-throw circle.
    assert any(len(p.interiors) for p in plan.polygons)
    assert 12.0 < paint.area < 40.0


def test_a_half_court_has_one_hoop_and_one_hoop_means_one():
    _, plan = _plan(_pitch("w1", _court(100, 100, 16.8, 10.5), "basketball", hoops="1"))
    assert len(_kinds(plan, "hoop")) == 1 and plan.polygons
    # Which end it is is a hash of the id: the same each time, and the other id may take the other end.
    ends = {_plan(_pitch(f"w{k}", _court(100, 100, 16.8, 10.5), "basketball"))[1].objects[0].direction for k in range(1, 40)}
    assert ends == {90.0, 270.0}
    assert _plan(_pitch("w1", _court(100, 100, 16.8, 10.5), "basketball"))[1].objects == _plan(_pitch("w1", _court(100, 100, 16.8, 10.5), "basketball"))[1].objects


def test_a_court_with_run_off_at_each_end_is_one_court_centred():
    _, plan = _plan(_pitch("w1", _court(100, 100, 34.3, 15.4), "basketball", "court"))
    hoops = sorted(_kinds(plan, "hoop"), key=lambda o: o.x)
    assert len(hoops) == 2 and hoops[1].x - hoops[0].x == pytest.approx(28.65 - 2 * 1.22, abs=0.05)  # a 94-foot court, not stretched


def test_a_basketball_polygon_for_more_than_a_court_is_bare():
    assert not _plan(_pitch("w1", _court(100, 100, 40, 25), "basketball"))[1].polygons
    assert not _plan(_pitch("w1", _court(100, 100, 68, 20.6), "basketball"))[1].polygons


# --- soccer and football -----------------------------------------------------------------------------------------
def test_a_soccer_pitch_has_a_halfway_line_a_centre_circle_boxes_and_goals():
    polygon = _court(500, 500, 105, 65)
    _, plan = _plan(_pitch("w1", polygon, "soccer"))
    paint = _paint(plan)
    assert polygon.buffer(0.01).contains(paint)
    assert paint.contains(Point(500, 500 + 0.0 + 0.05))  # the halfway line through the centre spot
    assert any(len(p.interiors) >= 3 for p in plan.polygons)  # the outline joined to the halfway line, boxes and circle
    assert Point(500, 500).buffer(9.15, quad_segs=32).boundary.intersection(paint).length > 55.0  # the centre circle: 57.5 m round
    goals = sorted(_kinds(plan, "goal"), key=lambda o: o.x)
    assert [round(g.direction) for g in goals] == [90, 270]
    assert goals[0].x == pytest.approx(500 - 52.5, abs=0.05) and goals[1].x == pytest.approx(500 + 52.5, abs=0.05)
    assert all(g.size == pytest.approx(7.32) and g.height == pytest.approx(2.44) for g in goals)


def test_a_small_pitch_gets_a_small_goal_and_a_square_field_nothing():
    _, plan = _plan(_pitch("w1", _court(500, 500, 42, 28), "soccer"))
    goals = _kinds(plan, "goal")
    assert len(goals) == 2 and all(g.size < 4.0 for g in goals)
    assert not _plan(_pitch("w1", _court(500, 500, 70, 66), "soccer"))[1].polygons


def test_an_american_football_field_has_yard_lines_and_goalposts():
    polygon = _court(500, 500, 117, 64)
    _, plan = _plan(_pitch("w1", polygon, "american_football", "turf"))
    paint = _paint(plan)
    goal = 91.44 / 2
    for n in range(-9, 10):  # a line every five yards, across the field
        assert paint.contains(Point(500 + n * 4.572, 500)), n
    assert paint.contains(Point(500 - goal, 505)) and paint.contains(Point(500 + goal, 495))  # the goal lines
    assert not paint.contains(Point(500 + 2.0, 500.0))  # nothing between the lines
    posts = sorted(_kinds(plan, "goalpost"), key=lambda o: o.x)
    assert [round(p.direction) for p in posts] == [90, 270] and posts[1].x - posts[0].x == pytest.approx(109.73, abs=0.1)
    assert not _plan(_pitch("w1", _court(500, 500, 94, 86), "american_football"))[1].polygons  # not long enough for a football field
    assert _plan(_pitch("w1", polygon, "soccer;american_football"))[1].counts == {"soccer": 1}  # the first sport it fits


def test_a_pitch_listed_for_several_sports_takes_the_first_it_fits():
    _, plan = _plan(_pitch("w1", _court(500, 500, 60, 33), "soccer;multi;american_football"))
    assert plan.counts == {"soccer": 1}
    _, plan = _plan(_pitch("w1", _court(500, 500, 60, 33), "american_football;soccer"))  # too small for football: the soccer pitch
    assert plan.counts == {"soccer": 1}


# --- volleyball --------------------------------------------------------------------------------------------------
def test_a_volleyball_court_has_a_net_and_posts():
    _, plan = _plan(_pitch("w1", _court(100, 100, 19, 11), "volleyball", "dirt"), _pitch("w2", _court(200, 100, 25.1, 12.9), "beachvolleyball", "dirt"))
    assert len(plan.nets) == 2 and len(_kinds(plan, "net")) == 2
    assert all(n[4] == pytest.approx(2.43) for n in plan.nets)
    assert set(plan.counts) == {"beachvolleyball"}  # the 19 by 11 sand court is a beach court too: 16 by 8


# --- nesting, hiding ---------------------------------------------------------------------------------------------
def test_where_pitches_overlap_the_smaller_shows_and_lines_stop_at_it():
    soccer, tennis = _court(500, 500, 105, 65), _court(500, 500, 23.77, 10.97)
    both = _plan(_pitch("w1", soccer, "soccer"), _pitch("w2", tennis, "tennis"))[1]
    only_tennis = _plan(_pitch("w2", tennis, "tennis"))[1]
    inside = _paint(both).intersection(tennis).area
    assert inside == pytest.approx(_paint(only_tennis).intersection(tennis).area, rel=0.001)  # no soccer line on the court


def test_paint_stays_out_of_roads_paths_water_and_buildings():
    polygon = _court(100, 100, 23.77, 10.97)
    road = box(95, 90, 105, 110)
    scene = Scene()
    scene.areas["green"].append(_pitch("w1", polygon, "tennis"))
    scene.areas["road"].append(Area("w2", road, {"highway": "residential"}))
    plan = sports.plan(scene)
    assert plan.polygons and _paint(plan).intersection(road).area < 1e-6
    # A way with a width hides the ground it covers too.
    scene = Scene()
    scene.areas["green"].append(_pitch("w1", polygon, "tennis"))
    path = Way("w3", LineString([(100, 90), (100, 110)]), "footway", "path", 2.0, None, {})
    scene.ways.append(path)
    plan = sports.plan(scene)
    assert plan.polygons and _paint(plan).intersection(path.line.buffer(0.99)).area < 1e-6
    # A pitch mostly under a building is left bare.
    scene = Scene()
    scene.areas["green"].append(_pitch("w1", polygon, "tennis"))
    scene.buildings.append(Building("w4", box(90, 90, 110, 110), 12.0, 0.0, "default", "#aaaaaa", "#444444", {"building": "yes"}))
    assert not sports.plan(scene).polygons


# --- baseball and softball ---------------------------------------------------------------------------------------
HOME = (300.0, 300.0)


def _diamond_pitch(heading, sport="baseball", cover="pitch", **tags):
    t = math.radians(heading)
    tags = {"home_plate": _ll(*HOME), "direction": str(heading), **tags}
    return _pitch("w1", _fan(HOME, heading), sport, cover, **tags)


def test_the_polygons_outline_alone_does_not_make_a_diamond():
    assert not _plan(_pitch("w1", _fan(HOME, 20.0), "baseball"))[1].polygons


def test_the_fan_seed_finds_the_apex_and_axis_of_a_fan_shaped_outline():
    seed = sports.fan_seed(_fan(HOME, 20.0))
    home, axis, reach, foul = seed
    assert home == pytest.approx(HOME, abs=0.5) and axis == pytest.approx((math.sin(math.radians(20)), math.cos(math.radians(20))), abs=0.02)
    assert reach == pytest.approx(70.0, abs=1.0) and foul == pytest.approx(70.0, abs=1.0)
    assert sports.fan_seed(box(0, 0, 30, 14)) is None and sports.fan_seed(Point(0, 0).buffer(40)) is None  # a court, a circle


def test_a_diamond_from_tags_is_placed_where_the_tags_say():
    pitch = _diamond_pitch(35.0)
    _, plan = _plan(pitch)
    assert plan.counts == {"baseball": 1}
    d = sports.fit_diamond(sports.Pitch("w1", 0, pitch.polygon, pitch.tags, "pitch", (0, 0), 1, 0, 70, 70), "baseball")
    assert d.how == "tags" and d.home == pytest.approx(HOME, abs=0.02)
    assert d.axis == pytest.approx((math.sin(math.radians(35)), math.cos(math.radians(35))), abs=1e-6)
    assert d.base == pytest.approx(60 * 0.3048) and d.pitch == pytest.approx(46 * 0.3048) and not d.skin
    big = sports.fit_diamond(sports.Pitch("w1", 0, pitch.polygon, {**pitch.tags, "base_distance": "90 ft", "infield": "dirt"}, "pitch", (0, 0), 1, 0, 70, 70), "baseball")
    assert big.base == pytest.approx(90 * 0.3048) and big.pitch == pytest.approx(60.5 * 0.3048) and big.skin
    soft = sports.fit_diamond(sports.Pitch("w1", 0, pitch.polygon, pitch.tags, "pitch", (0, 0), 1, 0, 70, 70), "softball")
    assert soft.pitch == pytest.approx(43 * 0.3048)


def test_a_diamond_has_foul_lines_plate_bases_dirt_and_a_backstop():
    heading = 35.0
    _, plan = _plan(_diamond_pitch(heading))
    ey = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
    ex = np.array([ey[1], -ey[0]])
    home, b = np.array(HOME), 60 * 0.3048
    lines = unary_union([p for p, c, l in zip(plan.polygons, plan.colours, plan.lifts) if c == LINE_COLOUR and l == LINE_LIFT])
    # Foul lines run out along both base paths and well past the bases.
    for side in (-1, 1):
        for distance in (b, 2 * b):
            along = home + (ey + side * ex) / math.sqrt(2) * distance
            assert lines.distance(Point(*along)) < 0.06
    bases = [p for p, c, l in zip(plan.polygons, plan.colours, plan.lifts) if c == LINE_COLOUR and l == sports.BASE_LIFT]
    bases = unary_union(bases)
    for where in (home + ey * b * math.sqrt(2), home + (ey + ex) * b / math.sqrt(2)):  # second base; first (shifted in by 0.19)
        assert bases.distance(Point(*where)) < 0.4
    assert bases.distance(Point(*(home + ey * 0.2))) < 0.05  # home plate
    assert bases.distance(Point(*(home + ey * 14.02))) < 0.4  # the rubber, 46 feet from the back of the plate
    dirt = unary_union([p for p, c in zip(plan.polygons, plan.colours) if c == DIRT])
    assert dirt.contains(Point(*(home + ey * 14.02)))  # the mound
    assert dirt.contains(Point(*(home + (ey + ex) * b / math.sqrt(2) + ey * 0.5))) and not dirt.contains(Point(*(home + ey * 8.0)))  # grass inside the paths
    assert dirt.area == pytest.approx(480.0, abs=160.0)
    (stop,) = _kinds(plan, "backstop")
    assert (stop.x, stop.y) == pytest.approx(HOME, abs=0.05) and stop.direction == pytest.approx(heading) and 7.0 <= stop.size <= 11.0


def test_a_full_skin_infield_has_no_grass_and_a_diamond_on_dirt_has_no_dirt_drawn():
    _, plan = _plan(_diamond_pitch(0.0, infield="dirt"))
    dirt = unary_union([p for p, c in zip(plan.polygons, plan.colours) if c == DIRT])
    assert dirt.contains(Point(HOME[0], HOME[1] + 8.0))
    _, plan = _plan(_diamond_pitch(0.0, cover="dirt"))
    assert DIRT not in plan.colours and plan.polygons  # lines and bases, no patch of dirt on dirt
    _, plan = _plan(_diamond_pitch(0.0, cover="turf"))
    assert sports.DIRT_TURF in plan.colours


def test_an_infield_mapped_inside_the_pitch_gives_the_diamond():
    heading, b = 70.0, 90 * 0.3048
    ey = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
    ex = np.array([ey[1], -ey[0]])
    home = np.array(HOME)
    centre, half = home + ey * b / math.sqrt(2), 26.6 / math.sqrt(2)  # the grass square inside the base paths, 26.6 m a side
    square = Polygon([centre + half * ey, centre + half * ex, centre - half * ey, centre - half * ex])
    outer = _pitch("w1", _fan(HOME, heading, 100.0), "baseball")
    inner = _pitch("w2", square, "baseball", "dirt")
    _, plan = _plan(outer, inner)
    assert plan.counts == {"baseball": 1}  # one diamond: the infield polygon is part of it, not a pitch of its own
    (stop,) = _kinds(plan, "backstop")
    assert (stop.x, stop.y) == pytest.approx(HOME, abs=0.6) and stop.direction == pytest.approx(heading, abs=0.5)
    # The lines and dirt go on the infield polygon too.
    assert _paint(plan).intersection(square).area > 5.0
    area = sports.fit_diamond(sports.Pitch("w1", 0, outer.polygon, outer.tags, "pitch", (0, 0), 1, 0, 70, 70, children=[square]), "baseball")
    assert area.how == "infield" and area.base == pytest.approx(b)


def test_a_backstop_is_left_out_where_a_fence_is_already_mapped_behind_home_plate():
    heading = 0.0
    fence = Barrier("w9", LineString([(HOME[0] - 12, HOME[1] - 6), (HOME[0], HOME[1] - 8), (HOME[0] + 12, HOME[1] - 6)]), "fence", 3.0, 0.06)
    _, plan = _plan(_diamond_pitch(heading), barriers=[fence])
    assert plan.polygons and not _kinds(plan, "backstop") and plan.skipped["backstop (a fence is mapped there)"] == 1


# --- the shapes --------------------------------------------------------------------------------------------------
def _mesh(kind, x=10.0, y=20.0, **kw):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, kind, x, y, **kw)
    mesh = builder.finish()
    # glTF axes are x east, y up, z south: back to x, y north, z up.
    return np.column_stack([mesh.positions[:, 0], -mesh.positions[:, 2], mesh.positions[:, 1]]), mesh.indices.reshape(-1, 3)


def test_a_hoop_has_a_backboard_a_rim_at_305_cm_and_a_pole_behind():
    points, triangles = _mesh("hoop", direction=90.0)  # faces east
    assert len(triangles) <= 100
    assert points[:, 2].max() == pytest.approx(3.97) and points[:, 2].min() == pytest.approx(0.0)
    rim = points[np.isclose(points[:, 2], 3.08)]
    assert rim[:, 0].mean() == pytest.approx(10.0 + 0.39, abs=0.01) and rim[:, 1].mean() == pytest.approx(20.0, abs=0.01)  # in front of the board
    assert points[:, 0].min() == pytest.approx(10.0 - 1.75, abs=0.01)  # the pole is behind it, 1.75 m
    board = points[(points[:, 2] > 2.8) & (np.abs(points[:, 0] - 10.0) < 0.06)]
    assert np.ptp(board[:, 1]) == pytest.approx(1.83, abs=0.01)  # 1.83 m wide


def test_a_goal_stands_on_the_goal_line_with_a_net_behind():
    points, triangles = _mesh("goal", direction=0.0, size=7.32, height=2.44)  # faces north
    assert 100 <= len(triangles) <= 180
    assert points[:, 2].max() == pytest.approx(2.44) and points[:, 0].min() == pytest.approx(10 - 3.66 - 0.06, abs=0.01)
    assert points[:, 0].max() == pytest.approx(10 + 3.66 + 0.06, abs=0.01)
    assert points[:, 1].max() == pytest.approx(20.06, abs=0.01) and points[:, 1].min() == pytest.approx(20 - 2.0, abs=0.03)  # the net goes back 2 m


def test_a_goalpost_has_a_crossbar_at_305_and_uprights_to_91():
    points, triangles = _mesh("goalpost", direction=270.0, size=5.64, height=9.1)  # faces west
    assert len(triangles) <= 70
    assert points[:, 2].max() == pytest.approx(9.1)
    bar = points[np.isclose(points[:, 2], 3.05)]
    assert bar.size and np.ptp(bar[:, 1]) == pytest.approx(5.64 + 0.08, abs=0.01)  # across the north-south direction


def test_net_posts_stand_a_net_apart_and_are_slim():
    points, triangles = _mesh("net", direction=90.0, size=12.8, height=1.07)
    assert len(triangles) <= 30
    assert np.ptp(points[:, 1]) == pytest.approx(12.8 + 0.1, abs=0.01) and np.ptp(points[:, 0]) <= 0.11 and points[:, 2].max() == pytest.approx(1.07)


def test_a_backstop_curves_behind_home_plate_and_runs_on_in_wings():
    points, triangles = _mesh("backstop", direction=0.0, size=8.0, height=4.6)  # faces north
    assert 100 <= len(triangles) <= 400
    assert points[:, 2].max() == pytest.approx(4.6)
    assert points[:, 1].min() == pytest.approx(20 - 8.0 - 0.06, abs=0.01)  # 8 m behind home plate
    assert points[:, 1].max() > 20.0 - 1.0  # the wings come forward to level with it
    assert np.ptp(points[:, 0]) > 16.0
    line = sports._backstop_line(10.0, 20.0, 0.0, 8.0)
    assert line.bounds[1] == pytest.approx(20 - 8.0, abs=0.01) and line.length > 25.0


# --- paint in a tile ---------------------------------------------------------------------------------------------
def test_marks_are_cut_to_the_tile_and_a_net_is_drawn_whole_in_the_tile_of_its_middle():
    polygon = _court(256.0, 100, 23.77, 10.97)  # straddles the edge between tile (0, 0) and tile (1, 0)
    _, plan = _plan(_pitch("w1", polygon, "tennis"))
    left, right = MeshBuilder("markings", colours=True), MeshBuilder("markings", colours=True)
    sports.add_marks(left, plan, box(0, 0, 256, 256), np.array([0.0, 0.0]))
    sports.add_marks(right, plan, box(256, 0, 512, 256), np.array([256.0, 0.0]))
    xs = [m.positions[:, 0] for m in (left.finish(), right.finish())]
    assert xs[0].max() <= 256.0 + 1e-3 and xs[1].min() >= -1e-3 and xs[0].max() > 250.0 and xs[1].max() > 5.0
    # The net's middle is at x = 256: it is in the second tile.
    (net,) = plan.nets
    assert (net[0] + net[2]) / 2 == pytest.approx(256.0)
    heights = [m.positions[:, 1].max() for m in (left.finish(), right.finish())]
    assert heights[0] == pytest.approx(LINE_LIFT, abs=1e-4)  # only paint
    assert heights[1] == pytest.approx(0.914, abs=1e-3)  # the net, 91 cm


def test_a_net_is_seen_from_both_sides():
    builder = MeshBuilder("markings", colours=True)
    sports._add_net(builder, (10.0, 5.0, 10.0, 15.0, 0.914), np.array([0.0, 0.0]))
    mesh = builder.finish()
    corners = mesh.positions[mesh.indices.reshape(-1, 3)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert len(corners) >= 16 and (normals[:, 0] > 0).sum() == (normals[:, 0] < 0).sum() == len(corners) // 2
    assert len(corners) <= 80  # a tennis net is a few dozen triangles
    z = mesh.positions[:, 1]
    assert z.max() == pytest.approx(0.914) and z.min() >= 0.05 - 1e-6  # the tape is at the top, the lowest band off the ground
    tape = mesh.positions[mesh.positions[:, 1] > 0.9]
    assert np.ptp(tape[:, 1]) <= 0.08  # the tape is a narrow band across the top


def test_a_built_tile_has_the_lines_in_markings_and_the_equipment_in_furniture(tmp_path):
    scene = Scene()
    scene.areas["green"].append(_pitch("w1", _court(128, 128, 25.6, 15.3), "basketball", "court", hoops="2"))
    scene.areas["green"].append(_pitch("w2", _court(60, 60, 23.77, 10.97), "tennis", "court"))
    stats = build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    assert stats["triangles"] > 0
    path = tmp_path / "tiles" / "0_0.glb"
    document = _read_glb(path)
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    assert {"markings", "furniture"} <= set(layers)
    positions, triangles = _mesh_data(path, document, layers["markings"])
    assert positions[:, 1].min() >= 0.0 and positions[:, 1].max() == pytest.approx(0.914, abs=1e-3)  # paint above the ground, the net at 91 cm
    flat = positions[triangles][np.ptp(positions[triangles][:, :, 1], axis=1) < 1e-6]  # the paint is flat and faces up
    normals = np.cross(flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 0])
    assert len(flat) and (normals[:, 1] > 0).all() and np.isclose(flat[:, :, 1], LINE_LIFT).all()
    data = json.loads((tmp_path / "tiles" / "0_0.json").read_text())
    kinds = sorted(o["kind"] for o in data["objects"])
    assert kinds == ["hoop", "hoop", "net"]


# --- the surface of a court --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tags", "cover"),
    [
        ({"leisure": "pitch", "sport": "tennis"}, "court"),
        ({"leisure": "pitch", "sport": "basketball"}, "court"),
        ({"leisure": "pitch", "sport": "skateboard"}, "court"),
        ({"leisure": "pitch", "sport": "tennis", "surface": "clay"}, "dirt"),
        ({"leisure": "pitch", "sport": "tennis", "surface": "artificial_turf"}, "turf"),
        ({"leisure": "pitch", "sport": "basketball", "surface": "asphalt"}, "court"),
        ({"leisure": "pitch", "sport": "soccer"}, "pitch"),
        ({"leisure": "pitch", "sport": "baseball"}, "pitch"),
        ({"leisure": "pitch", "sport": "tennis;soccer"}, "pitch"),
        ({"leisure": "pitch"}, "pitch"),
        ({"leisure": "playground", "sport": "basketball"}, "playground"),
    ],
)
def test_a_court_with_no_surface_stated_is_a_hard_court(tags, cover):
    assert green_cover(tags) == ("green", cover)

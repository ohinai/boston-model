"""Playgrounds, allotment plots, bandstands and gazebos, and the small park things (playgrounds.py).

Shapes are checked on their own for winding, size and budget; the layout of a playground and of a field of plots on small
scenes for what they keep clear of; the open structures on buildings of a few shapes; and a tile is built end to end.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
from test_parks import _mesh_data, _read_glb

from boston_pipeline import extras, playgrounds, world
from boston_pipeline.features import Area, Barrier, Building, Object, Scene, Way, classify, green_cover
from boston_pipeline.furniture import add_object
from boston_pipeline.mesh import MeshBuilder, colour_bytes
from boston_pipeline.tiles import COVERS, build


def _build(kind, x=10.0, y=20.0, **options):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, kind, x, y, **options)
    return builder.finish()


def _enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


def _node(ident, tags, x=5.0, y=5.0):
    lon, lat = (float(v) for v in world.to_lonlat(x, y))
    return {"type": "node", "id": ident, "lon": lon, "lat": lat, "tags": tags}


def _playground(ident="w1", polygon=None, **tags):
    return Area(ident, polygon or box(20, 20, 60, 50), {"leisure": "playground", **tags}, "playground")


def _scene(*playgrounds, **lists):
    scene = Scene()
    scene.areas["green"].extend(playgrounds)
    for name, values in lists.items():
        getattr(scene, name).extend(values)
    return scene


def _thing(ident, kind, x, y, **more):
    return Object(ident, x, y, kind, None, more.get("direction"), more.get("height"), more.get("size"), None, None, more.get("variant"), more.get("tags", {}))


def _footprint(thing):
    return playgrounds._footprint(thing.kind, thing.x, thing.y, thing.direction, thing.size, thing.height)


EQUIPMENT = [
    ("play_swing", {}), ("play_swing", {"size": 4, "direction": 70}), ("play_slide", {}), ("play_slide", {"height": 2.5, "direction": 200}),
    ("play_slide", {"variant": "steel", "material": "stainless_steel"}), ("play_frame", {}), ("play_frame", {"direction": 123}),
    ("play_seesaw", {}), ("play_spring", {"direction": 300}), ("play_sandpit", {}), ("play_sandpit", {"size": 4.0, "direction": 45}),
]
SMALL_THINGS = [
    ("bbq", {}), ("bbq", {"variant": "gas", "direction": 90}), ("picnic_table", {}), ("picnic_table", {"size": 2.4, "direction": 33}),
    ("chess_table", {}), ("chess_table", {"size": 0.9, "colour": "#2b2b2d", "direction": 250}), ("toilets", {}), ("toilets", {"variant": "steel"}),
    ("open_gazebo", {}), ("open_gazebo", {"size": 8.0, "direction": 140}),
]
BUDGETS = {  # triangles
    "play_swing": 140, "play_slide": 160, "play_frame": 170, "play_seesaw": 60, "play_spring": 90, "play_sandpit": 50,
    "bbq": 50, "picnic_table": 80, "chess_table": 130, "toilets": 40, "open_gazebo": 210,
}
HEIGHTS = {  # the tallest point of each, within a little
    "play_swing": 2.5, "play_slide": 2.2, "play_frame": 3.3, "play_seesaw": 1.4, "play_spring": 0.9, "play_sandpit": 0.3,
    "bbq": 0.95, "picnic_table": 0.8, "chess_table": 0.8, "toilets": 2.9,
}


# --- shapes -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,options", EQUIPMENT + SMALL_THINGS)
def test_faces_wind_counter_clockwise_and_normals_are_unit(kind, options):
    mesh = _build(kind, **options)
    assert len(mesh.indices) > 0 and len(mesh.indices) % 3 == 0
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-9).all(), "degenerate triangle"
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()


@pytest.mark.parametrize("kind,options", EQUIPMENT + SMALL_THINGS)
def test_stands_on_the_ground_where_it_is_put_and_carries_its_feature(kind, options):
    mesh = _build(kind, x=10.0, y=20.0, feature=4, **options)
    east, north, up = _enu(mesh).T
    assert up.min() >= -1e-6
    assert east.min() <= 10.0 + 1e-6 and east.max() >= 10.0 - 1e-6 and north.min() <= 20.0 + 1e-6 and north.max() >= 20.0 - 1e-6
    assert np.hypot(east - 10.0, north - 20.0).max() < 7.0
    assert (mesh.features == 4).all() and (mesh.colours[:, 3] == 255).all()


@pytest.mark.parametrize("kind,options", EQUIPMENT + SMALL_THINGS)
def test_triangle_budgets_and_determinism(kind, options):
    a, b = _build(kind, seed=7, **options), _build(kind, seed=7, **options)
    assert len(a.indices) // 3 <= BUDGETS[kind]
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)


@pytest.mark.parametrize("kind", sorted(HEIGHTS))
def test_each_piece_is_the_height_of_its_kind(kind):
    top = _enu(_build(kind))[:, 2].max()
    assert HEIGHTS[kind] - 0.35 <= top <= HEIGHTS[kind] + 0.1, f"{kind} is {top:.2f} m tall"


def test_a_slide_deck_is_as_high_as_it_is_told_and_the_chute_comes_down_to_the_ground():
    low, high = _enu(_build("play_slide", height=1.0)), _enu(_build("play_slide", height=2.0))
    assert high[:, 2].max() > low[:, 2].max() + 0.9
    assert high[:, 1].max() - high[:, 1].min() > low[:, 1].max() - low[:, 1].min()  # a taller slide runs out farther
    assert high[:, 2][np.argmax(high[:, 1])] < 0.5  # its far end is near the ground


def test_a_slide_runs_out_along_its_direction():
    for direction in (0, 90, 180, 270):
        xyz = _enu(_build("play_slide", direction=direction))
        ahead = np.array([math.sin(math.radians(direction)), math.cos(math.radians(direction))])
        reach = (xyz[:, :2] - np.array([10.0, 20.0])) @ ahead
        assert reach.max() > 2.0 > -reach.min()  # the chute is ahead, the ladder behind it


def test_a_swings_seats_follow_its_size_and_a_stainless_slide_is_steel():
    seats = lambda n: len(_build("play_swing", size=n, seed=5).indices)  # noqa: E731
    assert seats(1) < seats(2) < seats(4) < seats(6) == seats(9)
    steel = _build("play_slide", variant="steel", seed=3)
    assert (steel.colours == colour_bytes(playgrounds.STAINLESS)).all(axis=1).any()
    assert not (_build("play_slide", seed=3).colours == colour_bytes(playgrounds.STAINLESS)).all(axis=1).any()


def test_a_playgrounds_pieces_share_its_theme():
    def painted(kind, theme):
        return {tuple(row) for row in _build(kind, variant=str(theme), seed=1).colours}

    for theme in range(len(playgrounds.THEMES)):
        for kind in ("play_frame", "play_slide"):
            assert tuple(colour_bytes(playgrounds.THEMES[theme][0])) in painted(kind, theme)  # the platform is the theme's first colour
    orange = tuple(colour_bytes(playgrounds.THEMES[3][0]))
    assert orange not in painted("play_frame", 0) and orange in painted("play_frame", 3)


def test_a_toilet_with_a_building_over_it_draws_nothing_but_is_still_a_toilet():
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, "toilets", 10.0, 20.0, variant="indoor")
    assert builder.empty
    assert len(_build("toilets").indices) > 0


def test_a_chess_table_has_a_board_and_a_stool_each_end():
    xyz = _enu(_build("chess_table", direction=0))
    assert xyz[:, 1].max() - 20.0 > 0.8 and 20.0 - xyz[:, 1].min() > 0.8  # a stool each end of the table's length (along north)
    assert xyz[:, 0].max() - xyz[:, 0].min() < 0.65  # and nothing to the sides
    mesh = _build("chess_table")
    light = (mesh.colours == colour_bytes(playgrounds.CHESS_LIGHT)).all(axis=1)
    assert light.sum() == 32 * 4  # 32 squares of 4 vertices


def test_a_picnic_table_is_as_long_as_it_is_told():
    for length in (1.2, 1.8, 2.4):
        xyz = _enu(_build("picnic_table", size=length, direction=0))
        assert xyz[:, 1].max() - xyz[:, 1].min() == pytest.approx(length, abs=0.05)


# --- reading points -----------------------------------------------------------------------------------------------------
def test_interpret_reads_the_tags_of_each_kind():
    read = playgrounds.interpret
    assert read({"playground": "slide", "material": "stainless_steel"}) == ("play_slide", "steel", None)
    assert read({"playground": "swing", "seats": "4"}) == ("play_swing", None, 4.0)
    assert read({"playground": "basketswing"})[0] == "play_swing"
    assert read({"playground": "springy"})[0] == "play_spring" and read({"playground": "climbingframe"})[0] == "play_frame"
    assert read({"playground": "sandpit"})[0] == "play_sandpit" and read({"playground": "seesaw"})[0] == "play_seesaw"
    assert read({"playground": "roundabout"}) is None and read({"playground": "slide;swing"})[0] == "play_slide"
    assert read({"amenity": "bbq", "fuel": "charcoal"}) == ("bbq", None, None)
    assert read({"amenity": "bbq", "fuel": "gas"}) == ("bbq", "gas", None)
    assert read({"leisure": "picnic_table", "length": "2.4"}) == ("picnic_table", None, pytest.approx(2.4))
    assert read({"amenity": "picnic_table"})[0] == "picnic_table"
    assert read({"amenity": "table", "sport": "chess", "length": "0.9"}) == ("chess_table", None, pytest.approx(0.9))
    assert read({"amenity": "table"}) is None and read({"sport": "chess"}) is None  # a table with no chess, chess with no table
    assert read({"amenity": "toilets"}) == ("toilets", None, None)
    assert read({"amenity": "toilets", "name": "The Portland Loo"})[1] == "steel"
    assert read({"amenity": "shelter", "shelter_type": "gazebo"})[0] == "open_gazebo"
    assert read({"amenity": "shelter", "shelter_type": "weather_shelter"}) is None and read({"amenity": "bench"}) is None and read({}) is None


def test_every_interpreted_kind_has_a_shape_and_the_hook_finds_them():
    extras.loaded.cache_clear()
    assert playgrounds in extras.loaded()
    for kind in playgrounds.SHAPES:
        assert extras.shape(kind) is playgrounds.SHAPES[kind]
    assert extras.interpret({"amenity": "bbq"}) == ("bbq", None, None)


def test_classify_turns_the_points_into_objects():
    scene = classify([
        _node(1, {"playground": "slide", "name": "Cop slide", "material": "stainless_steel"}),
        _node(2, {"amenity": "bbq", "fuel": "gas", "access": "private"}),
        _node(3, {"amenity": "table", "sport": "chess", "length": "0.9", "colour": "black"}),
        _node(4, {"amenity": "toilets"}),
        _node(5, {"amenity": "bench"}),
        _node(6, {"amenity": "shelter", "shelter_type": "gazebo"}),
        _node(7, {"amenity": "shelter", "public_transport": "yes"}),
    ])
    things = {thing.id: thing for thing in scene.objects}
    assert set(things) == {"n1", "n2", "n3", "n4", "n5", "n6"}
    assert (things["n1"].kind, things["n1"].variant, things["n1"].name) == ("play_slide", "steel", "Cop slide")
    assert (things["n2"].kind, things["n2"].variant) == ("bbq", "gas")
    assert (things["n3"].kind, things["n3"].size, things["n3"].colour) == ("chess_table", pytest.approx(0.9), "#2b2b2d")
    assert things["n6"].kind == "open_gazebo"


# --- generic equipment ----------------------------------------------------------------------------------------------------
def test_the_number_of_pieces_follows_the_area():
    count = playgrounds.equipment_count
    assert [count(a) for a in (10, 44)] == [0, 0]
    assert count(60) == 1
    assert 2 <= count(100) <= count(199) <= 3  # "two or three" under 200
    assert count(500) < count(1000) < count(1500) == 8  # "eight or so" at 1,500
    assert count(1e6) == playgrounds.EQUIPMENT_COUNT["max"]
    assert all(count(a) <= count(a + 50) for a in range(45, 5000, 50))


def test_a_playground_gets_equipment_that_stays_inside_and_apart():
    scene = _scene(_playground())
    made = playgrounds.objects(scene)
    area = scene.areas["green"][0].polygon
    assert len(made) == playgrounds.equipment_count(area.area) and len(made) >= 6
    inside = area.buffer(-playgrounds.EDGE_MARGIN + 1e-6)
    for thing in made:
        assert thing.kind in playgrounds.SHAPES and thing.kind.startswith("play_")
        assert inside.contains(_footprint(thing)), f"{thing.id} is closer than {playgrounds.EDGE_MARGIN} m to the edge"
        assert thing.tags["generic"] == "yes" and thing.tags["playground"] and "source" in thing.tags
        assert thing.id.startswith("pg:w1:") and thing.name is None
    for i, a in enumerate(made):
        for b in made[i + 1 :]:
            assert _footprint(a).distance(_footprint(b)) >= playgrounds.SEPARATION - 1e-6


def test_the_layout_is_the_same_every_time_and_differs_between_playgrounds():
    first = playgrounds.objects(_scene(_playground()))
    again = playgrounds.objects(_scene(_playground()))
    assert [(t.id, t.kind, t.x, t.y, t.direction, t.variant) for t in first] == [(t.id, t.kind, t.x, t.y, t.direction, t.variant) for t in again]
    other = playgrounds.objects(_scene(_playground("w2")))
    assert [(t.x, t.y) for t in first] != [(t.x, t.y) for t in other]
    assert len({t.variant for t in first}) == 1  # one theme in a playground


def test_a_mix_of_kinds_and_a_big_structure_and_swings_in_a_big_playground():
    made = playgrounds.objects(_scene(_playground(polygon=box(0, 0, 60, 40))))
    kinds = [t.kind for t in made]
    assert len(set(kinds)) >= 4 and "play_swing" in kinds and ("play_frame" in kinds or "play_slide" in kinds)


def test_pieces_keep_clear_of_what_is_on_the_ground():
    area = _playground(polygon=box(0, 0, 40, 30))
    building = Building("b1", box(10, 10, 18, 18), 10.0, 0.0, "default", "#a", "#b", {"building": "yes"})
    path = Way("w9", LineString([(0, 22), (40, 22)]), "footway", "path", 2.0, None, {})
    scene = _scene(
        area, buildings=[building], ways=[path], objects=[_thing("n1", "bench", 30.0, 8.0)], trees=[SimpleNamespace(x=30.0, y=15.0)],
        barriers=[Barrier("f1", LineString([(25, 0), (25, 14)]), "fence", 1.2, 0.06)],
    )
    scene.areas["water"].append(Area("w5", box(32, 2, 38, 6), {"natural": "water"}))
    scene.areas["bed"].append(Area("w6", box(2, 2, 7, 7), {"leisure": "garden", "garden:type": "flowerbed"}, "flowers"))
    made = playgrounds.objects(scene)
    assert len(made) >= 4
    keep_out = [
        building.polygon.buffer(0.5), path.line.buffer(1.0 + 0.3), Point(30, 8).buffer(0.8), Point(30, 15).buffer(0.9),
        LineString([(25, 0), (25, 14)]).buffer(0.8), box(32, 2, 38, 6).buffer(0.3), box(2, 2, 7, 7).buffer(0.3),
    ]
    for thing in made:
        for blocked in keep_out:
            assert not _footprint(thing).intersects(blocked), f"{thing.id} ({thing.kind}) is on top of something"


def test_a_pitch_inside_a_playground_is_left_alone():
    scene = _scene(_playground(polygon=box(0, 0, 40, 30)))
    scene.areas["green"].append(Area("w7", box(0, 0, 25, 30), {"leisure": "pitch"}, "turf"))
    for thing in playgrounds.objects(scene):
        assert thing.x > 25.0


def test_a_smaller_lawn_on_a_playground_is_where_the_lawn_shows_so_no_equipment_goes_there():
    scene = _scene(_playground(polygon=box(0, 0, 40, 30)))
    scene.areas["green"].append(Area("w8", box(0, 0, 40, 18), {"leisure": "park", "name": "Little Park"}, "lawn"))  # the smaller one shows
    scene.areas["green"].append(Area("w9", box(0, 0, 400, 300), {"leisure": "park"}, "lawn"))  # a bigger park does not
    made = playgrounds.objects(scene)
    assert made and all(thing.y > 18.0 for thing in made)


def test_a_surveyed_piece_stops_a_generic_one_of_its_kind():
    surveyed = _thing("n1", "play_slide", 40.0, 35.0, tags={"playground": "slide"})
    for with_it in (False, True):
        scene = _scene(_playground(polygon=box(20, 20, 60, 50)), objects=[surveyed] if with_it else [])
        kinds = {t.kind for t in playgrounds.objects(scene)}
        assert ("play_slide" in kinds) is (not with_it)
    # A slide in one playground does not stop slides in another.
    scene = _scene(_playground("w1", box(20, 20, 60, 50)), _playground("w2", box(300, 20, 340, 50)), objects=[surveyed])
    by_playground = {t.id.split(":")[1]: set() for t in playgrounds.objects(scene)}
    for t in playgrounds.objects(scene):
        by_playground[t.id.split(":")[1]].add(t.kind)
    assert "play_slide" not in by_playground["w1"] and "play_slide" in by_playground["w2"]


def test_small_and_awkward_outlines_get_what_fits():
    assert playgrounds.objects(_scene(_playground(polygon=box(0, 0, 5, 5)))) == []  # 25 m2
    assert playgrounds.objects(_scene(_playground(polygon=box(0, 0, 3.5, 60)))) == []  # a strip too narrow once the edge is kept clear
    tot = playgrounds.objects(_scene(_playground(polygon=box(0, 0, 12, 12))))
    assert 1 <= len(tot) <= 3
    assert playgrounds.objects(_scene(_playground(indoor="yes"))) == []


def test_a_huge_playground_gets_ten_pieces_in_a_cluster():
    made = playgrounds.objects(_scene(_playground(polygon=box(0, 0, 120, 60))))
    assert len(made) == playgrounds.EQUIPMENT_COUNT["max"]
    xs, ys = np.array([t.x for t in made]), np.array([t.y for t in made])
    assert max(np.ptp(xs), np.ptp(ys)) < 70.0  # not strewn over 7,200 m2


def test_equipment_lines_up_with_the_playground():
    made = playgrounds.objects(_scene(_playground(polygon=Polygon([(0, 0), (40, 0), (40, 20), (0, 20)]))))
    for thing in made:
        assert min(thing.direction % 90, 90 - thing.direction % 90) < 7.0  # within the little jitter of a side of the outline


def test_only_playgrounds_get_equipment_and_a_non_scene_gets_none():
    park = Area("w3", box(0, 0, 60, 60), {"leisure": "park"}, "lawn")
    assert playgrounds.objects(_scene(park)) == []
    assert playgrounds.objects("not a scene") == []


# --- toilets ----------------------------------------------------------------------------------------------------------------
def test_a_toilet_inside_a_building_is_left_to_the_building_and_one_outside_faces_the_path():
    building = Building("b1", box(0, 0, 20, 20), 8.0, 0.0, "default", "#a", "#b", {"building": "yes"})
    path = Way("w9", LineString([(40, -50), (40, 50)]), "footway", "path", 2.0, None, {})
    inside, outside = _thing("n1", "toilets", 10.0, 10.0), _thing("n2", "toilets", 30.0, 30.0)
    stated = _thing("n3", "toilets", 30.0, 35.0, direction=200.0)
    scene = _scene(buildings=[building], ways=[path], objects=[inside, outside, stated])
    playgrounds.objects(scene)
    assert inside.variant == "indoor" and outside.variant is None
    assert outside.direction == pytest.approx(90.0, abs=1.0)  # the path is due east
    assert stated.direction == 200.0


# --- allotments ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tags,expected",
    [
        ({"landuse": "allotments", "leisure": "garden"}, True),
        ({"leisure": "garden", "garden:type": "community"}, True),
        ({"leisure": "garden", "garden:type": "allotment"}, True),
        ({"leisure": "garden", "name": "Penniman Road Community Garden"}, True),
        ({"leisure": "garden", "garden:type": "community", "name": "Minot Rose Garden"}, False),
        ({"leisure": "garden", "garden:type": "community", "name": "Rain Garden", "landuse": "grass"}, False),
        ({"leisure": "garden", "garden:type": "residential", "name": "Westgate/Tang Community Gardens"}, False),
        ({"leisure": "garden", "garden:type": "flowerbed"}, False),
        ({"leisure": "garden", "name": "Public Garden"}, False),
        ({"leisure": "park", "garden:type": "community"}, False),
    ],
)
def test_which_gardens_are_fields_of_plots(tags, expected):
    assert playgrounds.is_allotment(tags) is expected


def _garden(ident="w10", polygon=None, **tags):
    return Area(ident, polygon or box(20, 20, 80, 60), {"leisure": "garden", "garden:type": "community", **tags}, "garden")


def _plots(scene):
    playgrounds._cache.clear()
    return playgrounds.plots(scene)


def _rect(plot):
    t = math.radians(plot.heading)
    u, v = np.array([math.cos(t), -math.sin(t)]), np.array([math.sin(t), math.cos(t)])
    c = np.array([plot.x, plot.y])
    return Polygon([c + u * a * plot.width / 2 + v * b * plot.length / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def test_a_community_garden_is_laid_out_in_plots_about_the_standard_size():
    scene = _scene(_garden())
    found = _plots(scene)
    garden = scene.areas["green"][0].polygon
    assert 25 <= len(found) <= 60  # 60 x 40 m: about 4 x 6 m each with paths between
    for plot in found:
        assert 3.5 <= plot.width <= 4.5 and 5.4 <= plot.length <= 6.6
        assert garden.buffer(-playgrounds.PLOT_EDGE + 1e-6).contains(_rect(plot))
    rects = [_rect(p) for p in found]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert a.distance(b) >= playgrounds.PLOT_GAP - 0.05  # a path between any two plots
    assert _plots(scene) == found and playgrounds.plots(scene) == found  # the same every time


def test_plots_keep_off_buildings_paths_trees_water_and_mapped_beds():
    building = Building("b1", box(40, 30, 50, 40), 6.0, 0.0, "default", "#a", "#b", {"building": "yes"})
    path = Way("w9", LineString([(20, 45), (80, 45)]), "footway", "path", 2.0, None, {})
    scene = _scene(_garden(), buildings=[building], ways=[path], trees=[SimpleNamespace(x=65.0, y=30.0)])
    scene.areas["water"].append(Area("w5", box(25, 22, 30, 28), {"natural": "water"}))
    scene.areas["bed"].append(Area("w6", box(55, 50, 60, 58), {"leisure": "garden", "garden:type": "flowerbed"}, "flowers"))
    free = _plots(_scene(_garden()))
    found = _plots(scene)
    assert 0 < len(found) < len(free)
    blocked = [building.polygon, path.line.buffer(1.0), Point(65, 30).buffer(0.9), box(25, 22, 30, 28), box(55, 50, 60, 58)]
    for plot in found:
        for thing in blocked:
            assert not _rect(plot).intersects(thing)


def test_small_gardens_and_plain_green_get_no_plots():
    assert _plots(_scene(_garden(polygon=box(0, 0, 8, 8)))) == []  # 64 m2
    assert _plots(_scene(Area("w11", box(0, 0, 80, 60), {"leisure": "garden", "name": "Public Garden"}, "garden"))) == []


def _triangles(builder):
    return 0 if builder.empty else len(builder.finish().indices) // 3


def test_every_plot_is_drawn_once_in_the_tile_its_middle_is_in():
    scene = _scene(_garden(polygon=box(200, 20, 320, 70)))  # straddles the edge between tile (0, 0) and tile (1, 0)
    found = _plots(scene)
    assert any(p.x < 256 for p in found) and any(p.x >= 256 for p in found)
    drawn = []
    for tile_box in (box(0, 0, 256, 256), box(256, 0, 512, 256)):
        builder = MeshBuilder("beds", colours=True)
        playgrounds.add_plots(scene, tile_box, np.array([tile_box.bounds[0], tile_box.bounds[1]]), builder)
        drawn.append(_triangles(builder))
    everything = MeshBuilder("beds", colours=True)
    playgrounds.add_plots(scene, box(0, 0, 512, 256), np.array([0.0, 0.0]), everything)
    assert all(drawn) and sum(drawn) == _triangles(everything)
    assert 30 * len(found) < sum(drawn) < 130 * len(found)  # "about 50 triangles" a plot, more with a shed


def test_plots_are_low_and_their_planting_stays_in_the_plot():
    scene = _scene(_garden())
    builder = MeshBuilder("beds", colours=True)
    playgrounds.add_plots(scene, box(0, 0, 256, 256), np.array([0.0, 0.0]), builder)
    xyz = _enu(builder.finish())
    assert xyz[:, 2].min() >= -1e-6 and xyz[:, 2].max() < 2.2  # a shed is the tallest thing
    assert 20 <= xyz[:, 0].min() and xyz[:, 0].max() <= 80 and 20 <= xyz[:, 1].min() and xyz[:, 1].max() <= 60


# --- bandstands and gazebos -------------------------------------------------------------------------------------------------
def _round_building(radius=4.0, centre=(100.0, 100.0), sides=8, **tags):
    ring = [(centre[0] + radius * math.cos(2 * math.pi * k / sides + 0.2), centre[1] + radius * math.sin(2 * math.pi * k / sides + 0.2)) for k in range(sides)]
    tags = {"leisure": "bandstand", "building": "yes", **tags}
    return Building("w50", Polygon(ring), 12.0, 0.0, tags.pop("source", "default"), "#aaaaaa", "#bbbbbb", tags)


def _ctx(*ways):
    scene = Scene(ways=list(ways))
    return SimpleNamespace(scene=scene, ways_tree=STRtree([w.line for w in scene.ways]))


def _draw(building, ctx=None):
    walls, roofs = MeshBuilder("building_walls", uvs=True, colours=True, features=True), MeshBuilder("building_roofs", colours=True, features=True)
    handled = playgrounds.add_open_structure(ctx or _ctx(), walls, roofs, building, np.array([0.0, 0.0]), feature=3)
    return handled, walls, roofs


def test_which_buildings_are_open_structures():
    assert playgrounds.open_structure(_round_building()) == "bandstand"
    assert playgrounds.open_structure(_round_building(leisure=None, amenity="shelter", shelter_type="gazebo")) == "gazebo"
    assert playgrounds.open_structure(_round_building(leisure=None, amenity="shelter", shelter_type="pavilion")) == "gazebo"
    assert playgrounds.open_structure(_round_building(leisure=None, amenity="shelter", shelter_type="weather_shelter")) is None
    assert playgrounds.open_structure(_round_building(radius=10.0)) is None  # 300 m2: a stage building (the Hatch Shell is 676)
    assert playgrounds.open_structure(_round_building(radius=1.0)) is None  # too small to be one
    assert playgrounds.open_structure(_round_building(leisure=None)) is None


def test_a_round_bandstand_is_posts_a_half_wall_and_a_roof_over_a_raised_floor():
    handled, walls, roofs = _draw(_round_building(radius=4.2))
    assert handled and not walls.empty and not roofs.empty
    w, r = _enu(walls.finish()), _enu(roofs.finish())
    style = playgrounds.STRUCTURES["bandstand"]
    assert w[:, 2].min() >= -1e-6
    assert style["floor"] < w[:, 2].max() <= style["eave"] + 1e-6  # the posts reach the eaves
    assert r[:, 2].min() >= style["eave"] - 0.25 and r[:, 2].max() > style["eave"] + 0.9  # the roof starts at the eaves and rises
    assert 6.0 < r[:, 2].max() < 8.5  # a bandstand's total height
    centre = np.array([100.0, 100.0])
    assert np.hypot(*(w[:, :2] - centre).T).max() < 4.2 + 2.0 and np.hypot(*(r[:, :2] - centre).T).max() < 4.2 + 1.0
    assert (walls.finish().features == 3).all() and (roofs.finish().features == 3).all()
    assert len(walls.finish().indices) // 3 + len(roofs.finish().indices) // 3 < 320


def test_a_stated_height_is_the_total_height_to_the_top_of_the_finial():
    for stated in (5.5, 9.0):
        building = _round_building(radius=3.0)
        building.height, building.height_source = stated, "height"
        _, _, roofs = _draw(building)
        assert _enu(roofs.finish())[:, 2].max() == pytest.approx(stated, abs=0.05)


def _path_toward(heading, distance=30.0):
    """A footway across the way `heading` points, `distance` metres from the middle of a building at (100, 100)."""
    t = math.radians(heading)
    ahead, across = np.array([math.sin(t), math.cos(t)]), np.array([math.cos(t), -math.sin(t)])
    mid = np.array([100.0, 100.0]) + ahead * distance
    return Way("w9", LineString([tuple(mid - across * 40), tuple(mid + across * 40)]), "footway", "path", 2.0, None, {}), ahead


def test_the_steps_face_the_nearest_path():
    for heading in (0.0, 90.0, 180.0, 270.0, 40.0):
        path, ahead = _path_toward(heading)
        _, walls, _ = _draw(_round_building(radius=4.0), _ctx(path))
        xyz = _enu(walls.finish())
        low = xyz[xyz[:, 2] < 0.1][:, :2] - np.array([100.0, 100.0])  # the foot of the floor and of the steps
        assert (low @ ahead).max() > 4.0 + 0.9  # the steps stick out toward the path, past the floor's corners
        assert (low @ -ahead).max() < 4.5  # and nothing sticks out behind


def test_with_no_path_near_the_steps_face_south():
    _, walls, _ = _draw(_round_building(radius=4.0))
    low = _enu(walls.finish())
    low = low[low[:, 2] < 0.1][:, :2] - np.array([100.0, 100.0])
    assert -low[:, 1].min() > 4.0 + 0.9


def test_a_rectangular_gazebo_is_a_pavilion_and_a_stage_building_is_left_a_building():
    rectangle = Building("w51", box(100, 100, 108, 104), 3.0, 0.0, "default", "#aaaaaa", "#bbbbbb", {"amenity": "shelter", "shelter_type": "gazebo", "building": "yes"})
    handled, walls, roofs = _draw(rectangle)
    assert handled and not walls.empty and not roofs.empty
    assert _enu(roofs.finish())[:, 2].max() == pytest.approx(3.0, abs=0.05)
    shell = Building("w52", Polygon([(0, 0), (30, 0), (30, 22), (0, 22)]), 12.0, 0.0, "default", "#aaaaaa", "#bbbbbb", {"leisure": "bandstand", "building": "yes"})
    handled, walls, roofs = _draw(shell)
    assert not handled and walls.empty and roofs.empty


# --- surfaces ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tags,cover",
    [
        ({"leisure": "playground"}, "playground"), ({"leisure": "playground", "surface": "paved"}, "playground_hard"),
        ({"leisure": "playground", "surface": "asphalt"}, "playground_hard"), ({"leisure": "playground", "surface": "concrete"}, "playground_hard"),
        ({"leisure": "playground", "surface": "sand"}, "playground_sand"), ({"leisure": "playground", "surface": "grass"}, "grass"),
        ({"leisure": "playground", "surface": "rubber"}, "court"), ({"leisure": "playground", "surface": "artificial_turf"}, "turf"),
        ({"leisure": "playground", "surface": "dirt"}, "dirt"), ({"leisure": "pitch", "surface": "paved"}, "court"),
        ({"leisure": "pitch", "surface": "sand"}, "dirt"),
    ],
)
def test_a_playgrounds_surface_decides_its_ground(tags, cover):
    assert green_cover(tags) == ("green", cover)
    assert cover in COVERS["green"]


# --- a tile, end to end ------------------------------------------------------------------------------------------------------
def test_a_tile_draws_equipment_plots_and_a_bandstand(tmp_path):
    scene = _scene(_playground("w1", box(20, 20, 70, 60)), _garden("w2", box(100, 20, 170, 70)))
    scene.objects.append(_thing("n7", "bbq", 90.0, 90.0, variant="gas"))
    scene.objects.append(_thing("n8", "toilets", 200.0, 200.0))
    scene.buildings.append(_round_building(radius=4.0, centre=(200.0, 100.0)))
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    data = json.loads((tmp_path / "tiles" / "0_0.json").read_text())
    pieces = [o for o in data["objects"] if o["id"].startswith("pg:w1:")]
    assert len(pieces) >= 6 and {o["kind"] for o in pieces} <= set(playgrounds.SHAPES)
    assert {o["id"] for o in data["objects"]} >= {"n7", "n8"}
    assert [b["id"] for b in data["buildings"]] == ["w50"]  # the bandstand is still a building in the data
    glb = tmp_path / "tiles" / "0_0.glb"
    document = _read_glb(glb)
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    beds, _ = _mesh_data(glb, document, layers["beds"])
    assert len(beds) > 500 and beds[:, 1].max() < 2.2  # the plots
    furniture, _ = _mesh_data(glb, document, layers["furniture"])
    assert furniture[:, 1].max() > 2.5  # a swing frame or a tower
    walls, _ = _mesh_data(glb, document, layers["building_walls"])
    assert walls[:, 1].max() < 4.3  # the bandstand's posts are 4.6 m up from the ground and the roof is in its own layer; no 12 m box
    roofs, _ = _mesh_data(glb, document, layers["building_roofs"])
    assert 6.0 < roofs[:, 1].max() < 8.5


def test_the_toilet_nodes_inside_a_building_are_not_drawn(tmp_path):
    inside = _thing("n8", "toilets", 150.0, 150.0)
    scene = _scene(objects=[inside], buildings=[Building("b9", box(140, 140, 160, 160), 8.0, 0.0, "default", "#aaaaaa", "#bbbbbb", {"building": "yes"})])
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    data = json.loads((tmp_path / "tiles" / "0_0.json").read_text())
    assert [o["id"] for o in data["objects"]] == ["n8"]
    glb = tmp_path / "tiles" / "0_0.glb"
    layers = {mesh["name"] for mesh in _read_glb(glb)["meshes"]}
    assert "furniture" not in layers  # nothing was drawn for it

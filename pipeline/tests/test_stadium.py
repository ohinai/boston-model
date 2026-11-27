"""Ballparks and stadiums: stands, field walls, the walls of a stadium's parts, foul poles, light towers and landmark signs."""

import numpy as np
import pytest
from shapely.geometry import Point, box

from boston_pipeline import extras, furniture, stadium
from boston_pipeline.features import Area, Building, Scene, _object, classify
from boston_pipeline.mesh import FACADE_STYLE_STRIDE, SEED_STRIDE, MeshBuilder, facade_u

FIELD = box(30, 30, 70, 62)  # the playing surface, 40 by 32 m


def _building(ident, polygon, height, tags, part_of=None, part=True):
    return Building(ident, polygon, height, 0.0, "levels", "#86442f", "#9fc4cc", tags, part, part_of)


def _stadium(parts, with_field=True, outline_tags=None):
    """A stadium outline round a field, holding the given parts, as classify leaves it."""
    scene = Scene()
    if with_field:
        scene.areas["green"].append(Area("w1", FIELD, {"leisure": "pitch", "sport": "baseball"}, "pitch"))
    ring = box(0, 0, 100, 90).difference(box(28, 28, 72, 62))
    outline = Building("w9", ring, 12.0, 0.0, "default", "#8a4b3c", "#5a5a5c", {"building": "stadium", **(outline_tags or {})}, False)
    scene.outlines.append(outline)
    scene.buildings.extend(parts)
    return scene, outline


def _deck(ident="w10", polygon=None, height=18.0, tags=None):
    return _building(ident, polygon or box(0, 0, 100, 10), height, {"building:part": "yes", **(tags or {})}, "w9")


@pytest.fixture(autouse=True)
def _clean_plans():
    stadium.PLANS.clear()
    yield
    stadium.PLANS.clear()


def terraces(scene):
    return [b for b in scene.buildings if b.tags.get("stand") == "terrace"]


def test_a_stadium_with_a_field_gets_stands_where_no_part_stands():
    deck = _deck()
    scene, _ = _stadium([deck])
    stadium.augment(scene)
    made = terraces(scene)
    assert made, "the ring round the field, less the deck, is a stand"
    union = made[0].polygon
    for stand in made[1:]:
        union = union.union(stand.polygon)
    assert union.intersection(FIELD).area == pytest.approx(0, abs=0.01)  # no stand on the pitch
    assert union.intersection(deck.polygon).area == pytest.approx(0, abs=0.5)  # none where the part stands
    for stand in made:
        plan = stadium.part_plan(stand)
        assert plan.role == "terrace" and plan.rows >= 1
        assert stadium.FRONT_WALL < stand.height <= stadium.CONCOURSE_MAX
        assert plan.front.distance(FIELD) < 12  # the rows run along the edge nearest the field
    assert stadium.part_plan(deck).role == "deck"
    assert deck in scene.buildings  # a deck is still a building, drawn by add_part


def test_the_stand_rises_no_higher_than_half_the_structure_behind_it():
    scene, _ = _stadium([_deck(height=12.0)])
    stadium.augment(scene)
    behind = [b for b in terraces(scene) if b.polygon.distance(box(0, 0, 100, 10)) < 1.0]
    assert behind and max(b.height for b in behind) <= 6.0 + 1e-6


def test_an_enclosed_arena_with_no_pitch_is_left_alone():
    deck = _deck()
    scene, _ = _stadium([deck], with_field=False)
    stadium.augment(scene)
    assert not terraces(scene) and stadium.PLANS == {}
    assert scene.buildings == [deck]


def test_a_lone_stadium_polygon_is_replaced_by_stands_round_its_field():
    scene = Scene()
    scene.areas["green"].append(Area("w1", FIELD, {"leisure": "pitch", "surface": "artificial_turf"}, "turf"))
    lone = Building("w5", box(0, 0, 100, 90), 12.0, 0.0, "default", "#8a4b3c", "#5a5a5c", {"building": "stadium"}, False)
    scene.buildings.append(lone)
    stadium.augment(scene)
    assert lone not in scene.buildings
    stands = terraces(scene)
    assert stands and all(s.polygon.intersection(FIELD).area < 0.01 for s in stands)


def test_a_wide_bleachers_part_is_a_terrace_and_a_thin_one_with_a_height_is_a_wall():
    bleachers = _building("w20", box(0, 62, 100, 80), 10.0, {"building:part": "bleachers", "leisure": "bleachers"}, "w9")
    wall = _building("w21", box(28, 62, 72, 67), 11.33, {"building:part": "bleachers", "leisure": "bleachers", "height": "11.33", "stand:scoreboard": "yes"}, "w9")
    scene, _ = _stadium([bleachers, wall])
    stadium.augment(scene)
    assert bleachers not in scene.buildings  # replaced by a stand with its id
    stand = next(b for b in scene.buildings if b.id == "w20")
    assert stadium.part_plan(stand).role == "terrace" and stand.height <= 10.0
    assert stadium.part_plan(wall).role == "wall"
    assert stadium.part_plan(wall).scoreboard and stadium.part_plan(wall).topper


def test_the_stand_tag_overrides_the_choice():
    scene, _ = _stadium([_deck(tags={"stand": "none"})])
    stadium.augment(scene)
    assert stadium.PLANS.get(id(scene.buildings[0])) is None or stadium.part_plan(scene.buildings[0]) is None


def test_seat_colours_come_from_tags_else_the_plain_default():
    scene, _ = _stadium([_deck()], outline_tags={"seat:colour": "#1f4a36", "seat:tread": "#9d5d49"})
    stadium.augment(scene)
    plan = stadium.part_plan(terraces(scene)[0])
    assert (plan.seat, plan.tread) == ("#1f4a36", "#9d5d49")
    scene, _ = _stadium([_deck()])
    stadium.augment(scene)
    plan = stadium.part_plan(terraces(scene)[0])
    assert (plan.seat, plan.tread) == (stadium.SEAT, stadium.TREAD)


def _draw(scene, building):
    walls = MeshBuilder("building_walls", uvs=True, colours=True, features=True)
    roofs = MeshBuilder("building_roofs", colours=True, features=True)
    stadium.add_part(walls, roofs, building, stadium.part_plan(building), (0.0, 0.0), feature=3)
    return walls, roofs


def test_a_terrace_is_rows_of_steps_rising_from_a_front_wall_to_its_height():
    scene, _ = _stadium([_deck()])
    stadium.augment(scene)
    stand = max(terraces(scene), key=lambda b: b.polygon.area)
    walls, roofs = _draw(scene, stand)
    assert not roofs.empty
    top = roofs.finish().positions[:, 1]  # glTF y is up
    assert top.max() == pytest.approx(stand.height, abs=0.01)
    assert 0.5 < top[top > 0.01].min() < stadium.FRONT_WALL + 0.01  # the front wall is the lowest level drawn above the ground
    assert len(np.unique(np.round(top, 2))) >= stadium.part_plan(stand).rows  # one level a row


def test_walls_on_the_outside_of_a_stand_ask_the_shader_for_the_ballpark_facade():
    scene, _ = _stadium([_deck()])
    stadium.augment(scene)
    walls_found = False
    for building in scene.buildings:
        walls, _ = _draw(scene, building)
        if walls.empty:
            continue
        walls_found = True
        u = walls.finish().uvs[:, 0]
        u = u[u > 0]  # u = 0 is a wall too short for a bay
        assert len(u) and (np.floor((u + 0.5) / SEED_STRIDE) // FACADE_STYLE_STRIDE == stadium.FACADE_STYLE_OUTSIDE).all()
    assert walls_found


def test_a_deck_shows_its_field_side_as_a_fascia_and_a_wall_is_flat_colour_with_seats_on_top():
    deck = _deck()
    wall = _building("w21", box(28, 62, 72, 67), 11.33, {"building:part": "bleachers", "leisure": "bleachers", "height": "11.33"}, "w9")
    scene, _ = _stadium([deck, wall])
    stadium.augment(scene)
    _, deck_roofs = _draw(scene, deck)
    walls, wall_roofs = _draw(scene, wall)
    assert not deck_roofs.empty  # the wall facing the field and the roof
    assert walls.empty  # a field wall has no facade: both faces are flat colour
    heights = wall_roofs.finish().positions[:, 1]
    assert heights.max() > 11.33 + stadium.TOPPER_RAIL  # the seats on top


def test_facade_u_adds_the_style_to_the_seed_only_where_there_are_bays():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    length = np.array([10.0, 0.5])
    bays = np.array([3.0, 0.0])
    plain, _ = facade_u(a, length, bays)
    styled, _ = facade_u(a, length, bays, style=1)
    assert styled[0] - plain[0] == FACADE_STYLE_STRIDE * SEED_STRIDE
    assert styled[1] == plain[1] == 0.0


def test_tags_that_make_the_point_objects():
    assert stadium.interpret({"baseball": "foul_pole"}) == ("foul_pole", None, None)
    assert stadium.interpret({"man_made": "floodlight"})[0] == "light_tower"
    assert stadium.interpret({"man_made": "mast", "mast:type": "floodlight"})[0] == "light_tower"
    assert stadium.interpret({"man_made": "mast"}) is None
    sign = {"advertising": "sign", "tourism": "attraction", "height": "43"}
    assert stadium.interpret(sign)[0] == "landmark_sign"
    assert stadium.interpret({**sign, "height": "5"}) is None  # a shop sign
    assert stadium.interpret({"advertising": "sign", "height": "43"}) is None  # not a named landmark
    assert stadium.interpret({"amenity": "bench"}) is None


def test_features_reads_them_through_the_hook():
    extras.loaded.cache_clear()
    thing = _object("n1", {"baseball": "foul_pole", "height": "20", "direction": "90"}, 4.0, 5.0)
    assert (thing.kind, thing.height, thing.direction) == ("foul_pole", 20.0, 90.0)
    tower = _object("n2", {"man_made": "floodlight", "height": "27", "min_height": "18"}, 4.0, 5.0)
    assert tower.kind == "light_tower"
    sign = _object("n3", {"advertising": "sign", "tourism": "attraction", "height": "43", "direction": "245;65"}, 4.0, 5.0)
    assert sign.kind == "landmark_sign" and sign.direction == 245.0


def _extent(builder):
    positions = builder.finish().positions
    return positions.min(axis=0), positions.max(axis=0)


@pytest.mark.parametrize("kind, tags, top", [
    ("foul_pole", {}, 20.0),
    ("light_tower", {"min_height": "18"}, 27.0),
    ("landmark_sign", {"direction": "245;65"}, 43.0),
])
def test_the_point_objects_are_drawn_to_their_height(kind, tags, top):
    builder = MeshBuilder("furniture", colours=True, features=True)
    furniture.add_object(builder, kind, 50.0, 50.0, direction=90.0, height=top, tags=tags)
    low, high = _extent(builder)
    assert high[1] == pytest.approx(top, abs=0.05)  # up is y in glTF
    assert 0 < len(builder.finish().indices) // 3 < 600
    if kind == "light_tower":
        assert low[1] == pytest.approx(18.0, abs=0.05)  # the foot is on the roof, from min_height


def test_a_landmark_sign_is_two_sided_when_its_direction_names_two_ways():
    one, two = (MeshBuilder("furniture", colours=True, features=True) for _ in range(2))
    furniture.add_object(one, "landmark_sign", 0.0, 0.0, height=30.0, tags={"direction": "245"})
    furniture.add_object(two, "landmark_sign", 0.0, 0.0, height=30.0, tags={"direction": "245;65"})
    assert len(two.finish().indices) > len(one.finish().indices)


def test_a_landmark_signs_triangle_is_only_drawn_where_a_fix_asks_for_it():
    plain, triangle = (MeshBuilder("furniture", colours=True, features=True) for _ in range(2))
    furniture.add_object(plain, "landmark_sign", 0.0, 0.0, height=30.0, tags={"direction": "245"}, colour="#b22222")
    furniture.add_object(triangle, "landmark_sign", 0.0, 0.0, height=30.0, tags={"direction": "245", "sign:face": "triangle"})
    assert len(triangle.finish().indices) > len(plain.finish().indices)
    red = [c for c in plain.finish().colours if tuple(c[:3]) != tuple(triangle.finish().colours[0][:3])]
    assert len(red), "a `colour` tag colours the board"


def test_classify_finds_a_stadium_in_source_elements():
    """The whole path: OpenStreetMap-shaped elements to stands, without building anything by hand."""

    def way(ident, tags, ring):
        return {"type": "way", "id": ident, "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in ring]}

    # About 100 m square round a 40 m field, near Fenway Park, in longitude and latitude.
    x0, y0, dx, dy = -71.0980, 42.3460, 0.00122, 0.00090
    def at(u, v):
        return (x0 + u * dx, y0 + v * dy)

    field = [at(0.3, 0.3), at(0.7, 0.3), at(0.7, 0.62), at(0.3, 0.62), at(0.3, 0.3)]
    grandstand = [at(0.0, 0.0), at(1.0, 0.0), at(1.0, 1.0), at(0.0, 1.0), at(0.0, 0.0)]
    elements = [
        way(1, {"leisure": "pitch", "sport": "baseball", "surface": "grass"}, field),
        way(2, {"building": "grandstand"}, grandstand),
    ]
    scene = classify(elements)
    assert scene.buildings and all(b.tags.get("stand") == "terrace" for b in scene.buildings)
    assert all(stadium.part_plan(b) is not None for b in scene.buildings)
    centre = Point(np.mean([b.polygon.centroid.x for b in scene.buildings]), np.mean([b.polygon.centroid.y for b in scene.buildings]))
    assert not any(b.polygon.contains(centre) for b in scene.buildings)  # the field is open

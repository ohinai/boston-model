"""The Public Garden's landmarks: ducks, bell and shaft pedestals, detailed people, groups, fountain figures, the stone
lantern, the flag, and the Washington horse fitted to its pedestal (monuments.py, and the hooks in furniture.py and
features.py). test_monuments.py has the older statue, pedestal and animal tests; they are untouched."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from boston_pipeline import assets, fixes, monuments, world
from boston_pipeline.features import classify, flies_us_flag
from boston_pipeline.furniture import (
    ANIMAL_LENGTH,
    EQUESTRIAN_PEDESTAL_LENGTH,
    EQUESTRIAN_PEDESTAL_SHARE,
    KINDS,
    PLINTH_STONE,
    STATUE_FIGURE_MAX,
    add_object,
)
from boston_pipeline.mesh import MeshBuilder, colour_bytes
from boston_pipeline.monuments import ANIMAL_HEIGHT, species_family

X, Y = 10.0, 20.0


def _build(kind, **options):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, kind, X, Y, **options)
    return builder.finish()


def _enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0] - X, -p[:, 2] - Y, p[:, 1]])  # east, north, up, from the object


def _along(enu, heading):
    t = np.radians(heading)
    return enu[:, 0] * np.sin(t) + enu[:, 1] * np.cos(t)


def _triangles(mesh):
    return len(mesh.indices) // 3


def _check_solid(mesh):
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-9).all(), "degenerate triangle"
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()
    assert _enu(mesh)[:, 2].min() >= -1e-6


def _has_colour(mesh, hexcolour):
    return bool((mesh.colours == colour_bytes(hexcolour)).all(axis=1).any())


# --- which animal -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,family",
    [("duck", "bird"), ("Duck", "bird"), ("goose;duck", "bird"), ("swan", "bird"), ("dog", "four_legged"), ("lion", "four_legged"),
     ("panther", "four_legged"), ("sea_turtle", "turtle"), ("tortoise;hare", "turtle"), ("frog", None), ("", None), (None, None)],
)
def test_the_animal_tag_chooses_a_shape(tag, family):
    assert species_family({"animal": tag} if tag is not None else {}) == family


def test_no_tags_at_all_is_not_an_animal():
    assert species_family(None) is None and species_family({}) is None


@pytest.mark.parametrize("tag", ["duck", "dog", "sea_turtle"])
@pytest.mark.parametrize("length", [0.3, 0.42, 0.9, 2.0])
@pytest.mark.parametrize("direction", [0, 90, 239])
def test_an_animal_is_as_long_as_asked_and_stands_on_the_ground(tag, length, direction):
    mesh = _build("statue", variant="animal", size=length, direction=direction, tags={"animal": tag}, material="bronze")
    _check_solid(mesh)
    enu = _enu(mesh)
    assert np.ptp(_along(enu, direction)) == pytest.approx(length, rel=0.04)  # nose to tail is the length asked for
    assert enu[:, 2].min() == pytest.approx(0.0, abs=0.01)  # on the ground, no pedestal
    assert enu[:, 2].max() < 0.9 * length
    assert _triangles(mesh) < 200


@pytest.mark.parametrize("tag", ["duck", "dog", "sea_turtle"])
def test_an_animals_height_follows_the_family_ratio_and_the_height_tag(tag):
    family = species_family({"animal": tag})
    enu = _enu(_build("statue", variant="animal", size=1.0, tags={"animal": tag}))
    assert enu[:, 2].max() == pytest.approx(ANIMAL_HEIGHT[family], rel=0.12)
    # With only a height given, the length follows from it.
    enu = _enu(_build("statue", variant="animal", height=0.5, tags={"animal": tag}))
    assert enu[:, 2].max() == pytest.approx(0.5, rel=0.12)


def test_a_duck_faces_the_way_it_is_turned_and_has_a_head_up_front_and_a_tail_behind():
    enu = _enu(_build("statue", variant="animal", size=0.9, direction=90, tags={"animal": "duck"}))
    east = enu[:, 0]
    head = enu[enu[:, 2] > 0.5]  # only the neck and head rise this high
    assert len(head) and head[:, 0].mean() > 0.1  # ahead of the body's middle, toward the east
    assert east.max() > -east.min() * 0.9 and abs(enu[:, 1]).max() < 0.2  # it stays on its line


def test_a_duckling_is_a_smaller_duck_with_a_relatively_bigger_head():
    mother = _enu(_build("statue", variant="animal", size=0.9, tags={"animal": "duck"}))
    duckling = _enu(_build("statue", variant="animal", size=0.42, tags={"animal": "duck"}))
    assert duckling[:, 2].max() < 0.5 * mother[:, 2].max()
    assert duckling[:, 2].max() / 0.42 < mother[:, 2].max() / 0.9  # a lower, rounder shape for its length


def test_an_animal_with_no_stated_size_uses_the_default_length():
    enu = _enu(_build("statue", variant="animal", direction=0, tags={"animal": "duck"}))
    assert np.ptp(enu[:, 1]) == pytest.approx(ANIMAL_LENGTH, rel=0.04)


def test_an_animal_the_pipeline_has_no_shape_for_keeps_the_generic_one():
    frog = _build("statue", variant="animal", tags={"animal": "frog"})
    plain = _build("statue", variant="animal")
    assert np.array_equal(frog.positions, plain.positions)


def test_animals_are_deterministic():
    a = _build("statue", variant="animal", size=0.6, tags={"animal": "dog"}, seed=4)
    b = _build("statue", variant="animal", size=0.6, tags={"animal": "dog"}, seed=4)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)


# --- pedestals ----------------------------------------------------------------------------------------------------


def _section(mesh, z):
    """Where the surface crosses the horizontal plane at height z: points as (east, north) from the object."""
    tri = _enu(mesh)[mesh.indices.reshape(-1, 3)]
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, q = tri[:, a], tri[:, b]
        crossing = (p[:, 2] - z) * (q[:, 2] - z) < 0
        t = (z - p[crossing, 2]) / (q[crossing, 2] - p[crossing, 2])
        points.append(p[crossing, :2] + t[:, None] * (q[crossing, :2] - p[crossing, :2]))
    return np.concatenate(points)


def _radius_at(mesh, z):
    section = _section(mesh, z)
    return np.hypot(section[:, 0], section[:, 1]).max()


def test_a_bell_pedestal_is_round_and_flares_from_a_narrow_neck_to_a_wide_foot():
    mesh = _build("statue", height=3.4, tags={"pedestal": "bell"}, seed=3)
    _check_solid(mesh)
    top = max(3.4 - STATUE_FIGURE_MAX, 0.44 * 3.4)  # the pedestal's height
    foot, belly, neck, cap = (_radius_at(mesh, f * top) for f in (0.02, 0.30, 0.74, 0.95))
    assert foot > belly > neck  # the bell narrows as it rises
    assert cap > neck  # and a rounded cap overhangs the neck
    assert foot == pytest.approx(0.63 * min(0.34 * 3.4, 2.6), abs=0.02)
    # Round: the section is about as wide east-west as north-south.
    ring = _section(mesh, 0.3 * top)
    assert np.ptp(ring[:, 0]) == pytest.approx(np.ptp(ring[:, 1]), rel=0.06)


def test_a_bell_pedestal_stands_a_figure_on_it_at_the_same_total_height():
    mesh = _build("statue", height=3.4, tags={"pedestal": "bell"})
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(3.4, abs=0.02)
    plain = _enu(_build("statue", height=3.4))
    assert plain[:, 2].max() == pytest.approx(3.4, abs=0.02)
    assert _triangles(mesh) > _triangles(_build("statue", height=3.4))  # a round shape takes more triangles than boxes


def test_a_shaft_pedestal_is_tall_tapering_and_carries_a_group():
    total = 12.0
    mesh = _build("statue", height=total, tags={"pedestal": "shaft", "figures": "2"}, material="granite")
    _check_solid(mesh)
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(total, abs=0.02)
    stone_top = total - min(0.27 * total, STATUE_FIGURE_MAX)

    radius = lambda f: _radius_at(mesh, f * stone_top)  # noqa: E731

    assert radius(0.30) > radius(0.80)  # the shaft tapers
    assert radius(0.01) > radius(0.30)  # and steps out at its foot
    assert enu[enu[:, 2] > stone_top + 0.05][:, 2].max() - stone_top <= STATUE_FIGURE_MAX + 0.05  # the group is a person's height
    assert stone_top > 0.65 * total  # the shaft is most of the monument


def test_a_second_figure_lies_on_the_pedestal_in_front_of_the_first():
    one = _build("statue", height=12.0, tags={"pedestal": "shaft"}, material="granite")
    two = _build("statue", height=12.0, tags={"pedestal": "shaft", "figures": "2"}, material="granite")
    assert _triangles(two) > _triangles(one) + 60
    stone_top = 12.0 - min(0.27 * 12.0, STATUE_FIGURE_MAX)
    enu_one, enu_two = _enu(one), _enu(two)
    figures = enu_two[enu_two[:, 2] > stone_top + 0.01]
    assert figures[:, 1].max() > enu_one[enu_one[:, 2] > stone_top + 0.01][:, 1].max() + 0.3  # in front of the standing figure
    assert np.abs(figures[:, :2]).max() < 1.45  # on the pedestal's cap (a 2.5 m square under the footing), not off it


@pytest.mark.parametrize("style", ["", "stepped", "nonsense", "BELL "])
def test_an_unknown_or_missing_pedestal_style_is_the_stepped_pedestal_and_never_fails(style):
    mesh = _build("statue", height=3.4, tags={"pedestal": style})
    _check_solid(mesh)
    if style.strip().lower() != "bell":
        assert _triangles(mesh) == _triangles(_build("statue", height=3.4))


# --- people -------------------------------------------------------------------------------------------------------


def test_a_statue_a_walker_looks_up_at_is_drawn_in_full():
    mesh = _build("statue", height=3.4, seed=9)
    _check_solid(mesh)
    assert _enu(mesh)[:, 2].max() == pytest.approx(3.4, abs=1e-6)
    assert 250 < _triangles(_build("statue", height=2.0, variant="ground", seed=9)) < 500  # the figure alone


def test_a_small_statue_keeps_the_simple_figure():
    assert _triangles(_build("statue", height=1.2, variant="ground")) < 160


def test_the_four_poses_differ_and_each_is_a_sound_figure():
    meshes = [_build("statue", variant="ground", height=2.0, seed=seed) for seed in (0, 1, 2, 3)]
    for mesh in meshes:
        _check_solid(mesh)
    sizes = {_triangles(m) for m in meshes}
    assert len(sizes) >= 3  # a scroll and a cane add parts, so the poses are not all the same figure
    # The same seed is the same figure; a seed four apart has the same pose.
    again = _build("statue", variant="ground", height=2.0, seed=0)
    assert np.array_equal(meshes[0].positions, again.positions)


def test_a_figure_stands_on_its_base_and_its_head_is_the_highest_point():
    enu = _enu(_build("statue", variant="ground", height=2.0, seed=2))
    assert enu[:, 2].min() == pytest.approx(0.0, abs=1e-6) and enu[:, 2].max() == pytest.approx(2.0, abs=1e-6)
    top = enu[enu[:, 2] > 1.9]
    assert np.ptp(top[:, 0]) < 0.3  # the head, not the shoulders
    shoulders = enu[(enu[:, 2] > 1.5) & (enu[:, 2] < 1.62)]
    assert np.ptp(shoulders[:, 0]) > 0.45  # shoulders wider than the head


def test_a_child_has_a_bigger_head_and_shorter_legs_than_an_adult():
    adult = _enu(_build("statue", variant="ground", height=1.8, seed=0))
    child = _enu(_build("statue", variant="ground", height=1.8, seed=0, tags={"figure": "child"}))
    head = lambda e: np.ptp(e[e[:, 2] > 1.7][:, 0])  # noqa: E731
    assert head(child) > 1.15 * head(adult)
    assert child[:, 2].max() == pytest.approx(adult[:, 2].max(), abs=1e-6)


# --- Washington's horse on its pedestal ----------------------------------------------------------------------------


@pytest.mark.parametrize("assets_on", [False, True])
@pytest.mark.parametrize("direction", [0, 90, 250])
def test_the_horse_fits_on_the_pedestal_and_the_two_share_the_height(monkeypatch, assets_on, direction):
    if assets_on:
        monkeypatch.delenv(assets.DISABLE_VARIABLE, raising=False)
        assert assets.load("equestrian-figure") is not None
    total = 11.5824  # the Washington monument's height tag
    monument = MeshBuilder("monuments", colours=True, features=True)
    figures = MeshBuilder("figures", colours=True, features=True)
    add_object(monument, "statue", X, Y, variant="equestrian", height=total, direction=direction, material="bronze;granite", figures=figures)
    z0 = EQUESTRIAN_PEDESTAL_SHARE * total
    assert 0.45 * total <= z0 <= 0.55 * total  # roughly half pedestal, half bronze
    stone = _enu(monument.finish())
    if figures.empty:  # the procedural figure is in the same mesh as the pedestal
        bronze = stone[stone[:, 2] > z0 + 0.01]
    else:
        bronze = _enu(figures.finish())
    cap_length = 0.99 * EQUESTRIAN_PEDESTAL_LENGTH * total
    assert np.ptp(_along(bronze, direction)) <= cap_length  # no overhang past the pedestal's cap at either end
    assert np.abs(_along(bronze, direction)).max() <= cap_length / 2 + 0.02
    assert bronze[:, 2].max() == pytest.approx(total, abs=0.02)


def test_the_pedestal_is_warm_grey_granite_not_white():
    r, g, b = colour_bytes(PLINTH_STONE)[:3].astype(float)
    assert r > g > b  # warm
    assert (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.2  # a mid grey, in linear light, not white


# --- fountains ----------------------------------------------------------------------------------------------------


def test_a_fountain_with_a_figure_tag_has_a_bronze_child_on_a_pier_in_the_basin():
    plain = _build("fountain", height=1.5)
    sculpted = _build("fountain", height=1.5, direction=220, tags={"figure": "child"}, material="bronze")
    _check_solid(sculpted)
    enu = _enu(sculpted)
    assert enu[:, 2].max() == pytest.approx(1.5, abs=0.02)  # the stated height is the top of the figure
    figure = enu[enu[:, 2] > 0.8]
    assert len(figure) and np.hypot(figure[:, 0], figure[:, 1]).max() < 0.5  # in the middle
    assert _triangles(sculpted) > _triangles(plain) + 150
    assert not _has_colour(plain, "#4b4f44") and (sculpted.colours[enu[:, 2] > 1.0, :3].astype(int).sum(axis=1) < 200).all()


def test_a_fountain_with_an_animal_has_that_animal_on_a_pier():
    mesh = _build("fountain", tags={"animal": "panther"}, material="bronze")
    _check_solid(mesh)
    enu = _enu(mesh)
    assert 0.8 < enu[:, 2].max() < 1.75  # the default fountain height, with the figure on top
    assert enu[enu[:, 2] > 0.9][:, 0].size > 20


def test_two_figures_stand_round_the_pier_of_a_fountain():
    mesh = _build("fountain", height=1.0, tags={"figure": "child", "figures": "2"})
    _check_solid(mesh)
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(1.0, abs=0.02)
    up = enu[enu[:, 2] > 0.7]
    assert up[:, 0].min() < -0.05 or up[:, 1].min() < -0.05  # off to both sides of the middle
    assert up[:, 0].max() > 0.05 or up[:, 1].max() > 0.05


def test_a_plain_fountain_is_unchanged_by_all_this():
    before = _build("fountain", height=3.0)
    assert _enu(before)[:, 2].max() == pytest.approx(3.0, abs=0.05)
    assert not _has_colour(before, monuments.SPRAY)


def test_a_nozzle_fountain_throws_jets_up_to_its_height():
    mesh = _build("fountain", variant="nozzle", height=1.0)
    _check_solid(mesh)
    assert _has_colour(mesh, monuments.SPRAY)
    jets = _enu(mesh)
    assert jets[:, 2].max() == pytest.approx(1.0, abs=0.02)
    assert jets[(jets[:, 2] > 0.9)][:, 0].size > 0


def test_the_data_says_which_fountains_throw_jets():
    lon, lat = (float(v) for v in world.to_lonlat(5.0, 5.0))
    nodes = [{"type": "node", "id": i, "lon": lon, "lat": lat, "tags": {"amenity": "fountain", **tags}}
             for i, tags in enumerate(({"fountain": "nozzle"}, {"fountain": "jet_water"}, {"fountain": "bubbler"}, {}), start=1)]
    things = {t.id: t for t in classify(nodes).objects}
    assert [things[f"n{i}"].variant for i in (1, 2, 3, 4)] == ["nozzle", "nozzle", None, None]


def test_a_fountain_is_only_sculpted_when_the_data_says_so():
    assert monuments.is_sculpted_fountain({"figure": "child"}) and monuments.is_sculpted_fountain({"animal": "panther"})
    assert not monuments.is_sculpted_fountain({"artwork_type": "sculpture", "material": "steel"})  # an abstract steel piece
    assert not monuments.is_sculpted_fountain({"animal": "frog"}) and not monuments.is_sculpted_fountain(None)


# --- the stone lantern --------------------------------------------------------------------------------------------


def test_lantern_is_a_kind_and_a_japanese_stone_lantern_in_shape():
    assert "lantern" in KINDS
    mesh = _build("lantern", seed=5)
    _check_solid(mesh)
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(monuments.LANTERN_HEIGHT, abs=0.01)
    radius = lambda f: _radius_at(mesh, f * monuments.LANTERN_HEIGHT)  # noqa: E731
    assert radius(0.02) > 2.5 * radius(0.2)  # a wide base under a slender shaft
    assert radius(0.63) > 2 * radius(0.5)  # and a broad roof over the fire box
    assert _has_colour(mesh, monuments.LANTERN_OPENING)  # with openings in it
    assert _triangles(mesh) < 220


@pytest.mark.parametrize("height", [1.0, 2.4, 3.5])
def test_lantern_height_is_stated_height(height):
    assert _enu(_build("lantern", height=height))[:, 2].max() == pytest.approx(height, abs=0.01)


@pytest.mark.parametrize("direction", [0, 33, 90, 215])
def test_lantern_openings_face_four_ways_from_the_direction(direction):
    mesh = _build("lantern", direction=direction)
    openings = (mesh.colours == colour_bytes(monuments.LANTERN_OPENING)).all(axis=1)
    enu = _enu(mesh)[openings]
    angles = (np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) - direction) % 360  # compass bearing relative to `direction`
    for k in range(4):
        near = np.abs(((angles - 90 * k + 180) % 360) - 180) < 20
        assert near.any(), (direction, k)


def test_a_lantern_takes_a_stated_colour():
    mesh = _build("lantern", colour="#223344")
    assert _has_colour(mesh, "#223344")


def test_the_data_chooses_a_lantern_only_when_it_says_so():
    def node(ident, tags):
        lon, lat = (float(v) for v in world.to_lonlat(5.0, 5.0))
        return {"type": "node", "id": ident, "lon": lon, "lat": lat, "tags": tags}

    things = {t.id: t for t in classify([
        node(1, {"tourism": "artwork", "artwork_type": "lantern", "highway": "street_lamp", "name": "Japanese Lantern"}),
        node(2, {"tourism": "artwork", "artwork_type": "installation", "highway": "street_lamp"}),  # before the fix: a block
        node(3, {"tourism": "artwork", "artwork_type": "sculpture", "highway": "street_lamp", "height": "6.4", "colour": "gold"}),
    ]).objects}
    assert things["n1"].kind == "lantern"
    assert things["n2"].kind == "sculpture" and things["n3"].kind == "sculpture"  # the gold lamp sculptures stay as they were


# --- a memorial that flies the flag -------------------------------------------------------------------------------

US = {"country": "US", "flag:name": "United States", "flag:type": "national", "flag:wikidata": "Q42537"}


def test_which_poles_fly_the_us_flag():
    assert flies_us_flag(US) and flies_us_flag({"flag:wikidata": "Q42537"}) and flies_us_flag({"flag:name": "United States"})
    assert flies_us_flag({"country": "US", "flag:type": "national"})
    assert not flies_us_flag({}) and not flies_us_flag({"country": "US"}) and not flies_us_flag({"flag:wikidata": "Q1"})


def test_the_data_makes_a_flagpole_memorial_a_column_that_flies_a_flag():
    lon, lat = (float(v) for v in world.to_lonlat(5.0, 5.0))
    node = {"type": "node", "id": 1, "lon": lon, "lat": lat,
            "tags": {"historic": "memorial", "memorial": "column", "man_made": "flagpole", "material": "granite", **US}}
    plain = {"type": "node", "id": 2, "lon": lon, "lat": lat, "tags": {"historic": "memorial", "memorial": "column"}}
    things = {t.id: t for t in classify([node, plain]).objects}
    assert (things["n1"].kind, things["n1"].variant) == ("column", "flag")
    assert (things["n2"].kind, things["n2"].variant) == ("column", None)


def test_the_flag_memorial_is_a_granite_base_a_pole_and_a_striped_flag_with_a_canton():
    mesh = _build("column", variant="flag", direction=90)
    _check_solid(mesh)
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(12.0, abs=0.02)  # the default flagpole height
    for colour in (monuments.FLAG_RED, monuments.FLAG_WHITE, monuments.FLAG_BLUE):
        assert _has_colour(mesh, colour)
    cloth = (mesh.colours == colour_bytes(monuments.FLAG_RED)).all(axis=1)
    flag = enu[cloth]
    assert flag[:, 0].mean() > 0.8  # it streams east, the way the memorial is turned
    assert flag[:, 2].min() > 8 and flag[:, 2].max() < 12  # hoisted at the top of the pole
    assert np.ptp(flag[:, 0]) > 1.5 and np.ptp(flag[:, 2]) > 0.8


def test_the_flag_is_drawn_on_both_faces():
    mesh = _build("column", variant="flag")
    cloth = (mesh.colours == colour_bytes(monuments.FLAG_BLUE)).all(axis=1)
    normals = mesh.normals[cloth]
    assert (normals[:, 0] > 0.5).any() and (normals[:, 0] < -0.5).any()  # one side faces east, the other west (north-facing flag)


def test_a_plain_column_is_still_a_column_and_a_pole_flies_a_flag_only_if_the_data_says_which():
    column = _build("column", height=8.0)
    assert _enu(column)[:, 2].max() == pytest.approx(8.0, abs=0.01) and not _has_colour(column, monuments.FLAG_RED)
    assert _triangles(_build("flagpole")) == 24  # unchanged: a pole and a ball
    flying = _build("flagpole", tags=US)
    assert _triangles(flying) > 100 and _has_colour(flying, monuments.FLAG_BLUE)
    assert not _has_colour(_build("flagpole", tags={"flag:name": "Massachusetts"}), monuments.FLAG_BLUE)


# --- every shape, on odd input ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,options",
    [
        ("statue", {"height": 0.2, "tags": {"pedestal": "bell"}}),
        ("statue", {"height": 50.0, "tags": {"pedestal": "shaft", "figures": "3"}}),
        ("statue", {"height": 3.4, "tags": {"figures": "many", "figure": "adult"}}),
        ("statue", {"variant": "animal", "size": 0.01, "tags": {"animal": "duck"}}),
        ("statue", {"variant": "animal", "size": 500.0, "tags": {"animal": "lion"}}),
        ("statue", {"variant": "animal", "height": 0.001, "tags": {"animal": "turtle"}}),
        ("fountain", {"size": 1.2, "height": 12.0, "tags": {"figure": "child", "figures": "9"}}),
        ("fountain", {"size": 20.0, "height": 0.8, "tags": {"animal": "dog"}}),
        ("fountain", {"variant": "nozzle", "size": 1.2, "height": 0.8}),
        ("lantern", {"height": 0.0}),
        ("lantern", {"height": 99.0, "direction": 721}),
        ("column", {"variant": "flag", "height": 1.0}),
        ("column", {"variant": "flag", "height": 99.0}),
        ("flagpole", {"height": 3.0, "tags": US}),
    ],
)
def test_odd_input_still_makes_a_sound_mesh_near_its_position(kind, options):
    mesh = _build(kind, seed=-7, **options)
    _check_solid(mesh)
    enu = _enu(mesh)
    assert np.isfinite(enu).all() and np.hypot(enu[:, 0], enu[:, 1]).max() < 14.0


# --- the fix files ------------------------------------------------------------------------------------------------

FIXES = Path(__file__).resolve().parents[2] / "fixes"


def _loaded():
    return fixes.load(FIXES)


def test_the_garden_monument_fixes_load_and_say_why_and_touch_only_what_the_pipeline_reads():
    loaded = _loaded()
    mine = {i: p for i, p in loaded.patches.items() if p["file"] == "public-garden/garden-monuments.toml"}
    assert {"n4408379857", "n358276904", "n4408379858", "n4408379859", "n4408379867", "n4408379864", "n7693302800"} <= set(mine)
    readable = {"pedestal", "figures", "figure", "animal", "artwork_type"}
    for ident, patch in mine.items():
        assert set(patch["set"]) <= readable, (ident, patch["set"])
        assert not patch["delete"] and not patch["unset"]
    entries = tomllib.loads((FIXES / "public-garden" / "garden-monuments.toml").read_text(encoding="utf-8"))
    for ident, entry in entries.items():  # the loader insists on a `why`; ours say where the kind of thing comes from
        assert len(entry["why"]) > 60 and ("OpenStreetMap" in entry["why"] or "project owner" in entry["why"]), ident
    assert mine["n4408379857"]["set"] == {"pedestal": "bell"}  # Hale
    assert mine["n358276904"]["set"]["pedestal"] == "shaft" and mine["n358276904"]["set"]["figures"] == "2"  # the Ether Monument
    assert mine["n7693302800"]["set"] == {"artwork_type": "lantern"}


def test_the_nine_ducks_are_a_mother_and_eight_smaller_ducklings_all_facing_south_west():
    loaded = _loaded()
    ducks = {i: p["set"] for i, p in loaded.patches.items() if p["set"].get("animal") == "duck"}
    assert len(ducks) == 9
    lengths = {i: float(s["length"]) for i, s in ducks.items()}
    mother = lengths.pop("n2446221110")  # Mrs. Mallard
    assert len(lengths) == 8 and all(v < 0.6 * mother for v in lengths.values())
    assert all(v <= 0.58 for v in lengths.values())  # none longer than the closest two points are apart
    assert all(170 <= float(s["direction"]) <= 250 for s in ducks.values())
    assert all(s["support"] == "ground" for s in ducks.values())

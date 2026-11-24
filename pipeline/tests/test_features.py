import pytest

from boston_pipeline import world
from boston_pipeline.features import (
    LEVEL_HEIGHT,
    _road_width,
    building_height,
    classify,
    parse_colour,
    parse_length,
)


@pytest.mark.parametrize(
    ("text", "metres"),
    [("12", 12.0), ("12.5 m", 12.5), ("7,5", 7.5), ("40 ft", 12.192), ("40'", 12.192), ("12'6\"", 3.81), ("tall", None), (None, None)],
)
def test_parse_length(text, metres):
    assert parse_length(text) == (pytest.approx(metres) if metres is not None else None)


def test_parse_colour():
    assert parse_colour("#ABC") == "#aabbcc"
    assert parse_colour("#b5651d") == "#b5651d"
    assert parse_colour("red").startswith("#")
    assert parse_colour("mauve-ish") is None


def test_building_height_prefers_measured_height_then_levels_then_default():
    assert building_height({"building": "yes", "height": "61", "building:levels": "3"}) == (61.0, 0.0, "height")
    assert building_height({"building": "yes", "building:levels": "4"}) == (4 * LEVEL_HEIGHT + 1.0, 0.0, "levels")
    height, base, source = building_height({"building": "garage"})
    assert (height, base, source) == (3.0, 0.0, "default")


def test_building_part_and_roof_bases():
    assert building_height({"building:part": "yes", "height": "50", "min_height": "20"})[:2] == (50.0, 20.0)
    height, base, _ = building_height({"building": "roof"})
    assert base == pytest.approx(height - 0.4)  # a canopy you can walk under


def test_massgis_import_widths_are_ignored():
    # 30.2 m is a right-of-way width from the 2007 import, not a carriageway.
    imported = {"highway": "primary", "width": "30.2", "lanes": "3", "massgis:way_id": "1"}
    assert _road_width(imported, "primary") == pytest.approx(3 * 3.2 + 0.6)
    surveyed = {"highway": "primary", "width": "14"}
    assert _road_width(surveyed, "primary") == 14.0


def _way(ident, tags, points):
    return {"type": "way", "id": ident, "tags": tags, "geometry": [{"lon": lon, "lat": lat} for lon, lat in points]}


def _square(x, y, size):
    corners = [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]
    return [tuple(float(v) for v in world.to_lonlat(cx, cy)) for cx, cy in corners]


def test_classify_sorts_elements_into_features():
    lon, lat = (float(v) for v in world.to_lonlat(5.0, 5.0))
    elements = [
        _way(1, {"building": "yes", "height": "20"}, _square(0, 0, 20)),
        _way(2, {"highway": "residential", "name": "Test Street"}, _square(-50, -50, 10)[:3]),
        _way(3, {"highway": "footway", "tunnel": "yes"}, _square(-50, -50, 10)[:2]),
        _way(4, {"leisure": "park"}, _square(100, 100, 50)),
        _way(5, {"natural": "coastline"}, _square(200, 200, 10)[:2]),
        {"type": "node", "id": 6, "lon": lon, "lat": lat, "tags": {"natural": "tree"}},
        {"type": "node", "id": 7, "lon": lon, "lat": lat, "tags": {"shop": "books", "name": "A Bookshop"}},
        {"type": "node", "id": 8, "lon": lon, "lat": lat, "tags": {"amenity": "bench"}},
    ]
    scene = classify(elements)
    assert [b.id for b in scene.buildings] == ["w1"]
    assert scene.buildings[0].polygon.area == pytest.approx(400, rel=1e-3)
    assert [(w.id, w.surface, w.name) for w in scene.ways] == [("w2", "road", "Test Street")]  # tunnel skipped
    assert len(scene.areas["green"]) == 1 and len(scene.coastlines) == 1
    assert [t.id for t in scene.trees] == ["n6"]
    assert [(p.name, p.kind) for p in scene.pois] == [("A Bookshop", "shop=books")]


def test_outline_with_parts_is_dropped():
    elements = [
        _way(1, {"building": "yes", "height": "10", "name": "The Hall"}, _square(0, 0, 20)),
        _way(2, {"building:part": "yes", "height": "30"}, _square(2, 2, 6)),
        _way(3, {"building": "yes"}, _square(100, 0, 20)),
    ]
    buildings = {b.id: b for b in classify(elements).buildings}
    assert set(buildings) == {"w2", "w3"}
    assert (buildings["w2"].part_of, buildings["w2"].part_of_name) == ("w1", "The Hall")
    assert buildings["w3"].part_of is None


def test_features_keep_their_source_id_and_tags():
    elements = [
        _way(1, {"leisure": "park", "name": "A Park"}, _square(0, 0, 20)),
        {**_way(2, {"natural": "water"}, _square(100, 0, 20)), "ident": "fix:ponds#0"},
    ]
    scene = classify(elements)
    assert [(a.id, a.tags["name"]) for a in scene.areas["green"]] == [("w1", "A Park")]
    assert [a.id for a in scene.areas["water"]] == ["fix:ponds#0"]

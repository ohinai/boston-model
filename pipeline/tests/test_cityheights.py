"""The City of Boston's building survey heights (cityheights.py), offline.

The records are hand-written but shaped like the real replies of the Planning Department's `Boston_Buildings` layer 9, asked
with `f=geojson` on 27 November 2026: polygons of longitude and latitude with `IEL_TYPE` and `BLDG_HGT_2010` (feet above the ground).
"""

import json

import pytest

from boston_pipeline import brookline, cityheights, cli, tiles, world
from boston_pipeline.features import HEIGHT_FROM_CITY, building_height

ORIGIN = (-1500.0, -800.0)  # a spot in downtown Boston, in world metres
FOOT = 0.3048


def _lonlat(x, y):
    return [float(v) for v in world.to_lonlat(x, y)]


def rect(x, y, w, h):
    return [_lonlat(x, y), _lonlat(x + w, y), _lonlat(x + w, y + h), _lonlat(x, y + h), _lonlat(x, y)]


def part(x, y, w=10.0, h=12.0, feet=39.0, kind="BLDG"):
    return {
        "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [rect(ORIGIN[0] + x, ORIGIN[1] + y, w, h)]},
        "properties": {"IEL_TYPE": kind, "BLDG_HGT_2010": feet, "ROOF_ELEV_2010": feet + 5.0, "GRND_ELEV_2010": 5.0},
    }


def way(ident, x, y, w=10.0, h=12.0, **tags):
    return {
        "type": "way", "id": ident, "tags": {"building": "yes", **tags},
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in rect(ORIGIN[0] + x, ORIGIN[1] + y, w, h)],
    }


def usable(*records):
    return cityheights.footprints({"features": list(records)})


def heights(elements, *records):
    parts, _ = usable(*records)
    return cityheights.apply_heights(elements, parts)


# ------------------------------------------------------------------------------------------------- the parts


def test_heights_are_in_feet_and_what_is_not_a_building_is_set_aside():
    kept, skipped = usable(
        part(0, 0, feet=39.0),
        part(20, 0, feet=0.0),  # 975 parts of the real layer say 0
        part(40, 0, feet=5.0),  # lower than a shed
        part(60, 0, feet=39.0, kind="OVHD-WALKWAY"),
        part(80, 0, feet=788.2),  # the tallest real part, the John Hancock Tower
        part(100, 0, feet=39.0, kind="OUTBLDG"),
    )
    assert [round(p.feet, 1) for p in kept] == [39.0, 788.2, 39.0]
    assert [p.outbuilding for p in kept] == [False, False, True]
    assert skipped == {"no usable height": 2, "not a building": 1}


def test_a_part_with_no_height_value_is_set_aside():
    record = part(0, 0)
    record["properties"]["BLDG_HGT_2010"] = None
    kept, skipped = usable(record)
    assert kept == [] and skipped == {"no height": 1}


# ------------------------------------------------------------------------------------------------- giving heights to buildings


def test_a_building_with_no_height_gets_the_surveys_in_metres_and_says_where_from():
    out, stats = heights([way(1, 0, 0)], part(0, 0, feet=39.0))
    assert out[0]["tags"]["height"] == f"{39.0 * FOOT:.1f}" and out[0]["tags"]["height:source"] == HEIGHT_FROM_CITY
    assert stats["given"] == 1 and stats["over_levels"] == 0
    height, _, source = building_height(out[0]["tags"])
    assert (round(height, 1), source) == (11.9, "city")


def test_the_input_list_is_not_changed():
    source = [way(1, 0, 0)]
    heights(source, part(0, 0))
    assert "height" not in source[0]["tags"]


def test_openstreetmaps_own_height_and_the_unsuitable_buildings_are_left_alone():
    source = [
        way(1, 0, 0, height="30"), way(2, 20, 0, **{"min_height": "10"}), way(3, 40, 0, **{"building:part": "yes", "building": "no"}),
        way(4, 60, 0, building="garage"), way(5, 80, 0, building="roof"), way(6, 100, 0, **{"start_date": "2015"}),
    ]
    out, stats = heights(source, *[part(20 * i, 0) for i in range(6)])
    assert [e["tags"].get("height:source") for e in out] == [None] * 6
    assert stats["given"] == 0


def test_a_building_dated_in_or_before_the_survey_year_still_gets_a_height():
    out, _ = heights([way(1, 0, 0, start_date="1895"), way(2, 20, 0, start_date="2010-05")], part(0, 0), part(20, 0))
    assert [e["tags"]["height:source"] for e in out] == [HEIGHT_FROM_CITY] * 2


def test_a_count_of_levels_that_agrees_with_the_survey_is_replaced_by_the_measured_height():
    # 3 levels is 3 x 3.4 + 1.0 = 11.2 m; the survey says 38 ft = 11.6 m
    out, stats = heights([way(1, 0, 0, **{"building:levels": "3"})], part(0, 0, feet=38.0))
    assert out[0]["tags"]["height"] == "11.6" and stats["over_levels"] == 1 and stats["kept_levels"] == 0


def test_a_count_of_levels_the_survey_contradicts_is_kept():
    # 10 levels is 35 m; the survey (2010) says 36 ft: a block rebuilt since
    out, stats = heights([way(1, 0, 0, **{"building:levels": "10"})], part(0, 0, feet=36.0))
    assert "height" not in out[0]["tags"] and stats["kept_levels"] == 1 and stats["given"] == 0


def test_several_parts_give_the_height_that_covers_most_of_the_building():
    # a podium 8 x 12 at 30 ft under a bulkhead 2 x 12 at 60 ft
    out, _ = heights([way(1, 0, 0)], part(0, 0, w=8, feet=30.0), part(8, 0, w=2, feet=60.0))
    assert out[0]["tags"]["height"] == f"{30.0 * FOOT:.1f}"


def test_an_outbuilding_part_counts_only_where_nothing_else_does():
    out, _ = heights([way(1, 0, 0), way(2, 20, 0)], part(0, 0, feet=39.0), part(0, 0, feet=10.0, kind="OUTBLDG"), part(20, 0, feet=11.0, kind="OUTBLDG"))
    assert [e["tags"]["height"] for e in out] == [f"{39.0 * FOOT:.1f}", f"{11.0 * FOOT:.1f}"]


def test_a_part_that_covers_too_little_of_the_building_gives_nothing():
    out, stats = heights([way(1, 0, 0, w=40.0, h=40.0)], part(0, 0, w=10, h=12))
    assert "height" not in out[0]["tags"] and stats["uncovered"] == 1


def test_a_part_elsewhere_gives_nothing():
    out, stats = heights([way(1, 0, 0)], part(500, 500))
    assert "height" not in out[0]["tags"] and stats["given"] == 0


# ------------------------------------------------------------------------------------------------- download and build


def test_fetch_writes_the_pages_with_credit_and_licence(tmp_path, monkeypatch):
    pages = [([part(0, 0)], True), ([part(20, 0)], False)]
    seen = []

    def fake_page(endpoint, bbox, offset, who="the Brookline layer"):
        seen.append((endpoint, offset, who))
        return pages[len(seen) - 1]

    monkeypatch.setattr(brookline, "_page", fake_page)
    path = cityheights.fetch((-71.1, 42.3, -71.0, 42.4), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["features"]) == 2 and payload["credit"] == "City of Boston Planning Department"
    assert "PDDL" in payload["licence"] and [offset for _, offset, _ in seen] == [0, 1]
    assert cityheights.fetch((-71.1, 42.3, -71.0, 42.4), tmp_path) == path and len(seen) == 2  # cached, not asked again


def test_the_tiles_credit_the_city_only_where_the_area_uses_its_data():
    assert tiles.attribution({"extra_sources": ["city_heights"]}) == f"{tiles.ATTRIBUTION}; {tiles.CITY_ATTRIBUTION}"
    both = tiles.attribution({"extra_sources": ["brookline", "city_heights"]})
    assert tiles.BROOKLINE_ATTRIBUTION in both and tiles.CITY_ATTRIBUTION in both
    assert tiles.attribution({}) == tiles.ATTRIBUTION


def test_areas_may_ask_for_the_city_heights():
    assert cli._extra_sources({"extra_sources": ["brookline", "city_heights"]}) == ["brookline", "city_heights"]
    with pytest.raises(SystemExit):
        cli._extra_sources({"extra_sources": ["nonsense"]})

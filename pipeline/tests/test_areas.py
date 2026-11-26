"""Areas: a wide area is downloaded in strips and merged, and areas.toml is consistent."""

import json
import tomllib
from pathlib import Path

import pytest

from boston_pipeline import osm, world

AREAS = tomllib.loads((Path(__file__).resolve().parents[2] / "areas.toml").read_text(encoding="utf-8"))


def _payload(ids, stamp="2026-11-26T10:00:00Z", box=(0.0, 0.0, 1.0, 1.0)):
    return {
        "osm3s": {"timestamp_osm_base": stamp},
        "bbox_wsen": list(box),
        "elements": [{"type": "way", "id": i, "tags": {}} for i in ids],
    }


def test_a_narrow_box_is_one_strip():
    box = (-71.1, 42.3, -71.09, 42.4)
    assert osm.strips(box) == [box]


def test_a_wide_box_is_cut_into_equal_contiguous_strips_that_cover_it():
    box = (-71.15, 42.30, -71.05, 42.40)
    parts = osm.strips(box)
    assert len(parts) == 5
    assert parts[0][0] == pytest.approx(box[0]) and parts[-1][2] == pytest.approx(box[2])
    for before, after in zip(parts, parts[1:]):
        assert before[2] == pytest.approx(after[0])
    assert all(p[1] == box[1] and p[3] == box[3] for p in parts)
    assert all(p[2] - p[0] <= osm.STRIP_DEGREES + 1e-9 for p in parts)
    assert len({round(p[2] - p[0], 9) for p in parts}) == 1


def test_merge_keeps_each_element_once_and_the_oldest_timestamp():
    a = _payload([1, 2, 3], stamp="2026-11-26T10:00:00Z", box=(0, 0, 1, 1))
    b = _payload([3, 4], stamp="2026-11-25T23:59:59Z", box=(1, 0, 2, 1))
    merged = osm.merge([a, b])
    assert [e["id"] for e in merged["elements"]] == [1, 2, 3, 4]
    assert merged["osm3s"]["timestamp_osm_base"] == "2026-11-25T23:59:59Z"
    assert merged["bbox_wsen"] == [0, 0, 2, 1]


def test_merge_tells_a_node_from_a_way_with_the_same_id():
    a = {"osm3s": {"timestamp_osm_base": "2026-11-26T10:00:00Z"}, "elements": [{"type": "node", "id": 7}]}
    b = {"osm3s": {"timestamp_osm_base": "2026-11-26T10:00:00Z"}, "elements": [{"type": "way", "id": 7}]}
    assert len(osm.merge([a, b])["elements"]) == 2


def _fake_fetch(monkeypatch, calls):
    def fake(box, dest, force=False):
        calls.append(box)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_payload([len(calls), 99], box=box)), encoding="utf-8")
        return dest

    monkeypatch.setattr(osm, "fetch", fake)


def test_fetch_area_asks_once_for_a_narrow_box(tmp_path, monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls)
    dest = tmp_path / "osm.json"
    assert osm.fetch_area((-71.1, 42.3, -71.09, 42.4), dest) == dest
    assert len(calls) == 1
    assert not list(tmp_path.glob("osm_strip*.json"))


def test_fetch_area_fetches_a_wide_box_in_strips_and_merges_them(tmp_path, monkeypatch):
    calls = []
    _fake_fetch(monkeypatch, calls)
    dest = tmp_path / "osm.json"
    osm.fetch_area((-71.15, 42.30, -71.05, 42.40), dest)
    assert len(calls) == 5
    merged = json.loads(dest.read_text(encoding="utf-8"))
    assert sorted(e["id"] for e in merged["elements"]) == [1, 2, 3, 4, 5, 99]  # 99 is in every strip, kept once
    assert merged["bbox_wsen"][0] == pytest.approx(-71.15) and merged["bbox_wsen"][2] == pytest.approx(-71.05)
    assert len(list(tmp_path.glob("osm_strip*.json"))) == 5

    osm.fetch_area((-71.15, 42.30, -71.05, 42.40), dest)  # cached: nothing asked
    assert len(calls) == 5


@pytest.mark.parametrize("name", sorted(AREAS))
def test_every_place_is_inside_its_area(name):
    w, s, e, n = AREAS[name]["bbox"]
    for place in AREAS[name].get("places", []):
        assert w <= place["lon"] <= e and s <= place["lat"] <= n, place["name"]
        assert 0 <= place["heading"] < 360


def test_the_boston_area_contains_the_pilot_and_keeps_its_places():
    pilot, boston = AREAS["pilot"], AREAS["boston"]
    pw, ps, pe, pn = pilot["bbox"]
    bw, bs, be, bn = boston["bbox"]
    assert bw <= pw and bs <= ps and be >= pe and bn >= pn
    assert set(world.tiles_covering(tuple(pilot["bbox"]))) <= set(world.tiles_covering(tuple(boston["bbox"])))
    assert {p["name"] for p in pilot["places"]} <= {p["name"] for p in boston["places"]}
    # the pilot's tiles keep their ids: the origin does not move
    assert boston["places"][0]["name"] == pilot["places"][0]["name"]

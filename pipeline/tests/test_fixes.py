import json

import pytest

from boston_pipeline import fixes, lookup, world
from boston_pipeline.features import classify


def _lonlat(x, y):
    return [float(v) for v in world.to_lonlat(x, y)]


def _way(ident, tags, points):
    return {"type": "way", "id": ident, "tags": tags, "geometry": [{"lon": lon, "lat": lat} for lon, lat in points]}


def _square(x, y, size):
    return [_lonlat(cx, cy) for cx, cy in [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]]


def _source():
    lon, lat = _lonlat(50, 50)
    return [
        _way(1, {"building": "yes", "building:levels": "3"}, _square(0, 0, 20)),
        _way(2, {"building": "yes"}, _square(100, 0, 20)),
        _way(3, {"leisure": "park", "name": "Old Park"}, _square(200, 0, 40)),
        {"type": "node", "id": 4, "lon": lon, "lat": lat, "tags": {"natural": "tree"}},
    ]


def _write(folder, name, text):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_no_fixes_folder_changes_nothing(tmp_path):
    applied = fixes.apply(_source(), fixes.load(tmp_path / "missing"))
    assert (applied.patched, applied.removed, applied.added, applied.unmatched) == (0, 0, 0, [])
    assert len(applied.elements) == 4


def test_tag_patches_change_delete_and_report_unmatched(tmp_path):
    _write(
        tmp_path,
        "heights.toml",
        """
["w1"]
why = "Measured from the 2010 city roof survey"
set = { height = 61, "building:colour" = "#112233" }
unset = ["building:levels"]

["n4"]
why = "This tree was cut down in 2024"
delete = true

["w999"]
why = "Refers to something that no longer exists"
set = { height = "10" }
""",
    )
    source = _source()
    applied = fixes.apply(source, fixes.load(tmp_path))
    assert (applied.patched, applied.removed, applied.added) == (1, 1, 0)
    assert applied.unmatched == ["w999"]
    assert source[0]["tags"] == {"building": "yes", "building:levels": "3"}  # input untouched

    scene = classify(applied.elements)
    fixed = next(b for b in scene.buildings if b.id == "w1")
    assert (fixed.height, fixed.height_source, fixed.wall_colour) == (61.0, "override", "#112233")
    assert next(b for b in scene.buildings if b.id == "w2").height_source == "default"
    assert scene.trees == []


def test_geojson_adds_and_replaces_features(tmp_path):
    park = [_square(200, 0, 60)]
    pond = [_square(300, 0, 30), _square(310, 10, 5)]  # a pond with an island
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": _lonlat(60, 60)},
             "properties": {"natural": "tree", "why": "Missing elm by the gate, seen on site"}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": park},
             "properties": {"leisure": "park", "name": "New Park", "replaces": "w3", "why": "The park was extended east"}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": pond},
             "properties": {"natural": "water", "why": "Pond is missing"}},
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [_lonlat(0, 80), _lonlat(90, 80)]},
             "properties": {"highway": "footway", "surface": "gravel", "why": "Path is missing"}},
        ],
    }
    _write(tmp_path, "parks/garden.geojson", json.dumps(collection))
    applied = fixes.apply(_source(), fixes.load(tmp_path))
    assert (applied.patched, applied.removed, applied.added, applied.unmatched) == (0, 1, 4, [])

    scene = classify(applied.elements)
    assert sorted(t.id for t in scene.trees) == ["fix:parks/garden#0", "n4"]
    assert [(a.id, a.tags["name"], round(a.polygon.area)) for a in scene.areas["green"]] == [("fix:parks/garden#1", "New Park", 3600)]
    assert [(a.id, round(a.polygon.area)) for a in scene.areas["water"]] == [("fix:parks/garden#2", 875)]
    assert [(w.id, w.surface) for w in scene.ways] == [("fix:parks/garden#3", "path")]


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("a.toml", '["w1"]\nset = { height = "5" }\n', "needs a `why`"),
        ("a.toml", '["w1"]\nwhy = "x"\n', "changes nothing"),
        ("a.toml", '["building-1"]\nwhy = "x"\ndelete = true\n', "not a feature id"),
        ("a.toml", '["w1"]\nwhy = "x"\nheight = 5\n', "unknown keys"),
        ("a.toml", '["w1"\nwhy = "x"\n', "not valid TOML"),
        ("a.geojson", '{"type": "Feature"}', "FeatureCollection"),
        ("a.geojson", '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[0,0]},"properties":{"natural":"tree"}}]}', "needs a `why`"),
        ("a.geojson", '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[0,0]},"properties":{"why":"x"}}]}', "at least one tag"),
    ],
)
def test_bad_fix_files_are_rejected_with_a_pointer_to_the_file(tmp_path, name, text, message):
    _write(tmp_path, name, text)
    with pytest.raises(fixes.FixError, match=message) as error:
        fixes.load(tmp_path)
    assert f"fixes/{name}" in str(error.value)


def test_the_same_feature_cannot_be_changed_in_two_files(tmp_path):
    for name in ("a.toml", "b.toml"):
        _write(tmp_path, name, '["w1"]\nwhy = "x"\ndelete = true\n')
    with pytest.raises(fixes.FixError, match="already changed in fixes/a.toml"):
        fixes.load(tmp_path)


def test_inspect_reports_what_covers_a_point_and_what_is_near():
    scene = classify(_source())
    report = lookup.inspect(scene, 10.0, 10.0, radius=60.0)
    assert report["drawn_as"] == "building"
    assert [(e["id"], e["layer"]) for e in report["here"]] == [("w1", "building")]
    assert report["here"][0]["height_source"] == "levels"
    assert [(e["id"], e["layer"]) for e in report["nearby"]] == [("n4", "tree")]
    assert report["nearby"][0]["distance"] == pytest.approx(56.57, abs=0.01)
    assert report["tile"] == [0, 0]

    in_park = lookup.inspect(scene, 220.0, 20.0, radius=1.0)
    assert in_park["drawn_as"] == "green" and in_park["here"][0]["tags"]["name"] == "Old Park"
    assert lookup.inspect(scene, -500.0, -500.0, radius=1.0)["drawn_as"] == "ground"
    assert "Old Park" in lookup.format_report(in_park, 1.0)

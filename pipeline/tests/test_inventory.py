import json
import urllib.request

import pytest
from shapely.geometry import box

from boston_pipeline import inventory
from boston_pipeline.features import Area, Point, Scene


@pytest.mark.parametrize(
    "given, expected",
    [
        ("Linden, Littleleaf", "Littleleaf linden"),
        ("Elm, American", "American elm"),
        ("Maple, Red", "Red maple"),
        ("Honeylocust, Thornless", "Thornless honeylocust"),
        ("Katsura Tree", "Katsura tree"),
        ("Crabapple spp.", "Crabapple"),
        ("  Oak,   Red ", "Red oak"),
        ("", None),
        (None, None),
    ],
)
def test_common_names_are_put_in_natural_order(given, expected):
    assert inventory.common_name(given) == expected


def _record(ident, common="Linden, Littleleaf", botanical="Tilia cordata", dbh="10.0", park=None, coords=(-71.07, 42.35)):
    return {
        "type": "Feature",
        "properties": {"id": ident, "spp_com": common, "spp_bot": botanical, "dbh": dbh, "park": park},
        "geometry": {"type": "Point", "coordinates": list(coords)} if coords else None,
    }


def test_elements_skip_placeholders_and_build_tags():
    payload = {
        "features": [
            _record(1, park="Boston Common"),
            _record(2, common="Empty Pit/Planting Site", botanical="Vacant Site", dbh="0"),
            _record(3, common="Scheduled Planting Site - Fall 2026"),
            _record(4, coords=None),
            _record(1),  # duplicate id
            _record(5, common="Elm, American", botanical="Ulmus species", dbh="0"),
        ]
    }
    result = inventory.elements(payload)
    assert [e["ident"] for e in result] == ["bprd:1", "bprd:5"]
    tags = result[0]["tags"]
    assert tags["natural"] == "tree" and tags["species"] == "Tilia cordata" and tags["genus"] == "Tilia"
    assert tags["species:en"] == "Littleleaf linden" and tags["circumference"] == "0.80"
    assert tags["bprd:park"] == "Boston Common" and tags["source"] == "City of Boston tree inventory"
    assert (result[0]["lon"], result[0]["lat"]) == (-71.07, 42.35)
    assert "species" not in result[1]["tags"] and "circumference" not in result[1]["tags"]
    assert result[1]["tags"]["genus"] == "Ulmus"


def _scene(inventory_trees, osm_trees, park=None):
    scene = Scene()
    scene.trees = [Point(f"bprd:{i}", x, y, tags={"natural": "tree", "bprd:park": "P"}) for i, (x, y) in enumerate(inventory_trees)]
    scene.trees += [Point(f"n{i}", x, y, tags={"natural": "tree", **extra}) for i, (x, y, extra) in enumerate(osm_trees)]
    if park:
        scene.areas["green"].append(Area("w1", box(*park), {"leisure": "park"}))
    return scene


def test_reconcile_merges_close_pairs_and_keeps_distant_ones():
    scene = _scene([(0, 0), (100, 0)], [(3, 0, {}), (106, 0, {})])
    counts = inventory.reconcile(scene)
    assert counts == {"inventory": 2, "osm": 2, "merged": 1, "dropped": 0, "kept_osm": 1}
    assert sorted(t.id for t in scene.trees) == ["bprd:0", "bprd:1", "n1"]
    assert next(t for t in scene.trees if t.id == "bprd:0").tags["osm"] == "n0"


def test_reconcile_is_one_to_one_and_nearest_wins():
    scene = _scene([(0, 0)], [(3, 0, {}), (1, 0, {})])
    counts = inventory.reconcile(scene)
    assert counts["merged"] == 1 and counts["kept_osm"] == 1
    assert next(t for t in scene.trees if t.id == "bprd:0").tags["osm"] == "n1"
    assert [t.id for t in scene.trees if not t.id.startswith("bprd:")] == ["n0"]


def test_reconcile_copies_osm_tags_the_inventory_lacks():
    scene = _scene([(0, 0)], [(1, 0, {"name": "Founders Elm", "height": "20", "species": "Wrong"})])
    inventory.reconcile(scene)
    tags = scene.trees[0].tags
    assert (tags["name"], tags["height"]) == ("Founders Elm", "20")
    assert scene.trees[0].name == "Founders Elm"
    assert "species" not in tags  # only the listed OpenStreetMap tags are copied


def test_reconcile_drops_unmatched_osm_trees_only_in_inventoried_parks():
    grid = [(x * 5.0, y * 5.0) for x in range(5) for y in range(5)]  # 25 inventory trees inside the park
    scene = _scene(grid, [(60, 60, {}), (500, 500, {})], park=(-10, -10, 100, 100))
    counts = inventory.reconcile(scene)
    assert counts["dropped"] == 1 and counts["kept_osm"] == 1
    assert [t.id for t in scene.trees if not t.id.startswith("bprd:")] == ["n1"]


def test_reconcile_trusts_a_park_only_when_enough_trees_are_inventoried():
    scene = _scene([(0, 0)] * 3, [(60, 60, {})], park=(-10, -10, 100, 100))
    assert inventory.reconcile(scene)["dropped"] == 0


def test_fetch_pages_through_the_results(tmp_path, monkeypatch):
    pages = [
        {"features": [_record(i) for i in range(inventory.PAGE_SIZE)], "properties": {"exceededTransferLimit": True}},
        {"features": [_record(i) for i in range(inventory.PAGE_SIZE, inventory.PAGE_SIZE + 5)]},
    ]
    offsets = []

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def fake_urlopen(request, timeout=None):
        offsets.append(request.full_url.split("resultOffset=")[1].split("&")[0])
        assert "orderByFields=ObjectId" in request.full_url
        return Reply(pages[len(offsets) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "trees.json"
    inventory.fetch((-71.1, 42.3, -71.0, 42.4), dest)
    saved = json.loads(dest.read_text())
    assert offsets == ["0", str(inventory.PAGE_SIZE)]
    assert len(saved["features"]) == inventory.PAGE_SIZE + 5 and saved["licence"] == "ODC PDDL"
    inventory.fetch((-71.1, 42.3, -71.0, 42.4), dest)  # cached: no further requests
    assert len(offsets) == 2


def test_fetch_gives_up_with_a_clear_error(tmp_path, monkeypatch):
    def failing(request, timeout=None):
        raise OSError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    monkeypatch.setattr(inventory, "RETRY_PAUSE", 0)
    with pytest.raises(RuntimeError, match="failed 3 times"):
        inventory.fetch((0, 0, 1, 1), tmp_path / "t.json")

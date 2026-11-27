"""The Town of Brookline's tree inventory and building footprints (brookline.py), offline.

The records below are hand-written but shaped like the real replies of the two public ArcGIS layers on 26 November 2026,
asked with `f=geojson`: a FeatureCollection whose features have `properties` (the layer's fields: for trees ID, DBH in
inches, YR_PLANT, ScientificName, Cultivar, CommonName, IsStump, CONDITION, PARKTREE, CG_NOTES and so on; for buildings
TYPE, NUMSTORIES, BLDGHEIGHT in feet, TOP_GL (a LiDAR top in feet), COMMENT) and a `geometry` of longitude and latitude.
"""

import json
import math
import urllib.request

import pytest
from shapely.geometry import box

from boston_pipeline import brookline, cli, inventory, tiles, world
from boston_pipeline.features import HEIGHT_FROM_TOWN, Area, Point, Scene, building_height, classify

ORIGIN = (-4500.0, -2000.0)  # a spot in Brookline, in world metres


def _lonlat(x, y):
    return [float(v) for v in world.to_lonlat(x, y)]


# ------------------------------------------------------------------------------------------------- trees


def tree(ident="8838", sci="Tilia cordata", common="Littleleaf Linden", dbh=11, year=1994, cultivar=None, **more):
    props = {
        "OBJECTID": int(ident) if ident else 99, "ID": ident, "DBH": dbh, "YR_PLANT": year, "ScientificName": sci,
        "CommonName": common, "CommonNameTxt": common, "Cultivar": cultivar, "IsStump": None, "CONDITION": 1,
        "PARKTREE": 0, "MULTITRUNK": 0, "CG_NOTES": "", **more,
    }
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": _lonlat(*ORIGIN)}, "properties": props}


def elements(*records):
    return brookline.tree_elements({"features": list(records)})


def test_a_tree_becomes_a_node_with_the_tags_trees_py_reads():
    (node,) = elements(tree())
    assert node["ident"] == "brk:8838" and node["type"] == "node"
    tags = node["tags"]
    assert (tags["natural"], tags["species"], tags["genus"], tags["species:en"]) == ("tree", "Tilia cordata", "Tilia", "Littleleaf linden")
    assert tags["source"] == "Town of Brookline tree inventory"
    assert node["lon"] == pytest.approx(_lonlat(*ORIGIN)[0])


def test_dbh_is_inches_and_becomes_a_girth_in_metres():
    # 11 inches across is 0.88 m round: the same conversion as Boston's inventory (inventory.elements).
    assert elements(tree(dbh=11))[0]["tags"]["circumference"] == f"{math.pi * 11 * 0.0254:.2f}" == "0.88"
    boston = inventory.elements({"features": [{"properties": {"id": 1, "spp_com": "Maple, Red", "spp_bot": "Acer rubrum", "dbh": "11"},
                                               "geometry": {"coordinates": [-71.07, 42.35]}}]})
    assert boston[0]["tags"]["circumference"] == "0.88"


def test_absurd_and_missing_diameters_give_no_girth_but_a_size_to_assume():
    for dbh in (455, 1721, 0, None):
        tags = elements(tree(dbh=dbh, year=1994))[0]["tags"]
        assert "circumference" not in tags
        assert tags["denotation"] == "street"
    assert elements(tree(dbh=None, PARKTREE=1))[0]["tags"]["denotation"] == "park"
    assert elements(tree(dbh=100))[0]["tags"]["circumference"] == f"{math.pi * 100 * 0.0254:.2f}"  # trees.describe trusts up to 8 m


def test_the_planting_caliper_of_a_tree_planted_after_the_baseline_is_aged():
    year = brookline.THIS_YEAR - 20
    aged = 3 + brookline.PLANTED_GROWTH * 20
    assert elements(tree(dbh=3, year=year))[0]["tags"]["circumference"] == f"{math.pi * aged * 0.0254:.2f}"
    assert elements(tree(dbh=3, year=1994))[0]["tags"]["circumference"] == f"{math.pi * 3 * 0.0254:.2f}"  # the baseline: measured as is
    assert elements(tree(dbh=14, year=year))[0]["tags"]["circumference"] == f"{math.pi * 14 * 0.0254:.2f}"  # a large figure was remeasured
    # planted 11 years ago with no DBH at all: a nursery caliper plus the growth
    expected = brookline.PLANTING_CALIPER + brookline.PLANTED_GROWTH * 11
    assert elements(tree(dbh=None, year=brookline.THIS_YEAR - 11))[0]["tags"]["circumference"] == f"{math.pi * expected * 0.0254:.2f}"


def test_stumps_dead_trees_empty_records_and_repeats_are_left_out():
    records = [
        tree("1"),
        tree("2", IsStump=1),
        tree("3", CONDITION=4),
        tree("4", sci=None, common=None, dbh=None, year=None),  # no species, no size: nothing says a tree stands there
        tree("5", sci=None, common="Empty Planting Site", dbh=None),
        tree("1"),  # the same ID again
        tree("6", CONDITION=0),  # "Excellent" is a code too: it is not dead
    ]
    assert [e["ident"] for e in elements(*records)] == ["brk:1", "brk:6"]


def test_records_with_no_id_get_a_numeric_id_from_the_objectid():
    record = tree(ident=None)
    record["properties"]["OBJECTID"] = 7432
    assert elements(record)[0]["ident"] == f"brk:{brookline.NO_ID_BASE + 7432}"  # `brk:<digits>`: what fixes/ accepts


def test_cultivars_genus_only_names_and_typos():
    cultivar = tree("1", sci="Pyrus calleryana Aristocrat", common="Callery Pear Aristocrat", cultivar="Aristocrat")
    assert elements(cultivar)[0]["tags"]["species"] == "Pyrus calleryana 'Aristocrat'"
    assert elements(cultivar)[0]["tags"]["species:en"] == "Callery pear 'Aristocrat'"
    genus = elements(tree("2", sci="Malus", common="Crabapple"))[0]["tags"]
    assert "species" not in genus and genus["genus"] == "Malus" and genus["species:en"] == "Crabapple"
    assert elements(tree("3", sci="Fraxinus pennsylvanicum", common="Green Ash"))[0]["tags"]["species"] == "Fraxinus pennsylvanica"
    quoted = elements(tree("4", sci="Prunus 'Okame'", common="Okame Cherry"))[0]["tags"]
    assert quoted["species"] == "Prunus 'Okame'" and quoted["genus"] == "Prunus"


def test_the_newest_records_have_a_common_name_and_no_scientific_name():
    records = [
        tree("1", sci="Quercus palustris", common="Pin Oak"),
        tree("2", sci="Gleditsia triacanthos", common="Honeylocust"),
        tree(None, sci=None, common="Pin Oak", dbh=None, year=2015),
        tree("9", sci=None, common="Honeylocust Skyline", dbh=None, year=2015),  # the species with a cultivar after it
    ]
    records[2]["properties"]["OBJECTID"] = 40
    records[3]["properties"]["OBJECTID"] = 41
    by_id = {e["ident"]: e["tags"] for e in elements(*records)}
    assert by_id[f"brk:{brookline.NO_ID_BASE + 40}"]["species"] == "Quercus palustris"
    assert by_id["brk:9"]["species"] == "Gleditsia triacanthos 'Skyline'" and by_id["brk:9"]["species:en"] == "Honeylocust 'Skyline'"


def test_an_unknown_tree_takes_its_species_from_the_surveyors_note_only_when_the_note_is_a_known_one():
    pine = elements(tree("1", sci=None, common="Unknown", CG_NOTES="Pine"))[0]["tags"]
    assert pine["genus"] == "Pinus" and "species" not in pine and pine["species:en"] == "Pine"
    beech = elements(tree("2", sci=None, common="Unknown", CG_NOTES="European  Beech"))[0]["tags"]
    assert beech["species"] == "Fagus sylvatica"
    plain = elements(tree("3", sci=None, common="Unknown", CG_NOTES="unknown, alternate, leaf has 3 lobes"))[0]["tags"]
    assert "genus" not in plain and "species" not in plain and "species:en" not in plain  # a generic broadleaf of the right size
    assert plain["circumference"] == "0.88"


# ------------------------------------------------------------------------------------------------- merging with OpenStreetMap


def _scene_with(brk, osm_trees, bprd=(), park=None):
    scene = Scene()
    scene.trees = [Point(f"brk:{i}", x, y, tags={"natural": "tree", "source": brookline.TREE_SOURCE_TAG}) for i, (x, y) in enumerate(brk)]
    scene.trees += [Point(f"bprd:{i}", x, y, tags={"natural": "tree", "bprd:park": "P"}) for i, (x, y) in enumerate(bprd)]
    scene.trees += [Point(f"n{i}", x, y, tags={"natural": "tree", **extra}) for i, (x, y, extra) in enumerate(osm_trees)]
    if park:
        scene.areas["green"].append(Area("w1", box(*park), {"leisure": "park"}))
    return scene


def test_a_brookline_tree_and_an_openstreetmap_tree_within_4_m_are_one_tree():
    scene = _scene_with([(0, 0), (50, 0)], [(3, 0, {"name": "Old Elm"}), (60, 0, {})])
    counts = inventory.reconcile(scene, brookline.TREE_PREFIX, drop_in_parks=False)
    assert (counts["inventory"], counts["merged"], counts["kept_osm"]) == (2, 1, 1)
    merged = next(t for t in scene.trees if t.id == "brk:0")
    assert merged.tags["osm"] == "n0" and merged.name == "Old Elm"
    assert sorted(t.id for t in scene.trees) == ["brk:0", "brk:1", "n1"]


def test_openstreetmap_trees_stay_in_a_park_the_town_inventories():
    # Boston drops them (the city's inventory of its parks is complete); the Town's is a street and park inventory, so they stay.
    grid = [(x * 5.0, y * 5.0) for x in range(5) for y in range(5)]
    scene = _scene_with(grid, [(60, 60, {})], park=(-10, -10, 100, 100))
    counts = inventory.reconcile(scene, brookline.TREE_PREFIX, drop_in_parks=False)
    assert counts["dropped"] == 0 and counts["kept_osm"] == 1
    scene = _scene_with(grid, [(60, 60, {})], park=(-10, -10, 100, 100))
    assert inventory.reconcile(scene, brookline.TREE_PREFIX)["dropped"] == 0  # the Town's trees carry no `bprd:park` tag either


def test_the_two_inventories_leave_each_others_trees_alone():
    def run(scene):
        city = inventory.reconcile(scene, other=(brookline.TREE_PREFIX,))
        town = inventory.reconcile(scene, brookline.TREE_PREFIX, other=("bprd:",), drop_in_parks=False)
        return city, town

    scene = _scene_with([(0, 0), (500, 0)], [(2, 0, {}), (502, 0, {}), (900, 0, {})], bprd=[(502, 1), (700, 0)])
    city, town = run(scene)
    assert city["inventory"] == 2 and city["merged"] == 1 and town["inventory"] == 2 and town["merged"] == 1
    assert sorted(t.id for t in scene.trees) == ["bprd:0", "bprd:1", "brk:0", "brk:1", "n2"]
    assert next(t for t in scene.trees if t.id == "bprd:0").tags["osm"] == "n1"


def test_without_the_town_the_city_merge_is_unchanged():
    scene = _scene_with([], [(3, 0, {}), (106, 0, {})], bprd=[(0, 0), (100, 0)])
    assert inventory.reconcile(scene) == {"inventory": 2, "osm": 2, "merged": 1, "dropped": 0, "kept_osm": 1, "added": 0}
    assert sorted(t.id for t in scene.trees) == ["bprd:0", "bprd:1", "n1"]


# ------------------------------------------------------------------------------------------------- footprints


def rect(x, y, w, h):
    return [_lonlat(x, y), _lonlat(x + w, y), _lonlat(x + w, y + h), _lonlat(x, y + h), _lonlat(x, y)]


def footprint(x, y, w=10.0, h=12.0, height=26, stories=2, kind="BLDG", lidar=None, **more):
    props = {"TYPE": kind, "BLDGHEIGHT": height, "NUMSTORIES": stories, "TOP_GL": lidar, "COMMENT": " ", "COMMENTS": " ", **more}
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [rect(ORIGIN[0] + x, ORIGIN[1] + y, w, h)]}, "properties": props}


def usable(*records):
    return brookline.footprints({"features": list(records)})


def test_bldgheight_is_in_feet_and_junk_is_set_aside():
    kept, skipped = usable(
        footprint(0, 0, height=26),
        footprint(20, 0, height=0, stories=0),  # no height and no storeys
        footprint(40, 0, height=2021, stories=2021),  # junk
        footprint(60, 0, height=3, stories=0),  # lower than a shed
        footprint(80, 0, height=26, COMMENT="DEMOLISHED 2021"),
        footprint(100, 0, height=39, kind="OVHD-WALKWAY"),
    )
    assert [round(f.feet, 2) for f in kept] == [26.0]
    assert skipped == {"no height": 3, "demolished": 1, "not a building": 1}


def test_a_footprint_with_storeys_and_no_height_gets_the_towns_typical_height_for_that_many():
    # Five 3-storey buildings with the Town's 35.75 ft say what 3 storeys means in its figures; a 3-storey one with no height gets it.
    records = [footprint(20 * i, 0, height=35.75, stories=3) for i in range(5)] + [footprint(0, 30, height=0, stories=3), footprint(30, 30, height=0, stories=7)]
    kept, _ = usable(*records)
    assert [round(f.feet, 2) for f in kept][-2:] == [35.75, 91.0]  # no typical figure for 7 storeys: 13 ft each


def test_an_outbuilding_is_as_tall_as_its_lidar_top_not_as_its_houses_storeys():
    records = [
        footprint(0, 0, w=5, h=6, kind="OUTBLDG", height=35.75, stories=3, lidar=11.0),  # the house's figure is copied onto it
        footprint(20, 0, w=5, h=6, kind="OUTBLDG", height=35.75, stories=3, lidar=11.8),
        footprint(40, 0, w=5, h=6, kind="OUTBLDG", height=35.75, stories=3, lidar=90.0),  # a tree over it: not believed
        footprint(60, 0, w=5, h=6, kind="OUTBLDG", height=35.75, stories=3, lidar=None),
    ]
    kept, _ = usable(*records)
    assert [f.outbuilding for f in kept] == [True] * 4
    assert [round(f.feet, 1) for f in kept] == [11.0, 11.8, 11.4, 11.4]  # the last two: the median of the believed ones


def test_a_tall_claim_the_lidar_top_contradicts_is_set_aside_unless_the_lidar_is_broken():
    kept, skipped = usable(
        footprint(0, 0, height=78, stories=6, lidar=30.0),  # six storeys on a 50 m2 plot, LiDAR says 30 ft: rowhouses
        footprint(30, 0, height=104, stories=8, lidar=5.4),  # LiDAR "5.4 ft" on an 8-storey building: the LiDAR is wrong
        footprint(60, 0, height=130, stories=10, lidar=95.0),  # corroborated
        footprint(90, 0, height=26, stories=2, lidar=12.0),  # a small figure needs no corroboration
    )
    assert [f.feet for f in kept] == [104.0, 130.0, 26.0]
    assert skipped == {"tall, not confirmed by the LiDAR top": 1}


# ------------------------------------------------------------------------------------------------- giving heights to OpenStreetMap buildings


def way(ident, x, y, w=10.0, h=12.0, **tags):
    return {
        "type": "way", "id": ident, "tags": {"building": "yes", **tags},
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in rect(ORIGIN[0] + x, ORIGIN[1] + y, w, h)],
    }


def heights(elements_, *records):
    feet, _ = usable(*records)
    return brookline.apply_heights(elements_, feet)


def test_a_building_with_no_height_gets_the_towns_in_metres_and_says_where_from():
    out, stats = heights([way(1, 0, 0)], footprint(0, 0, height=26))
    assert out[0]["tags"]["height"] == "7.9" and out[0]["tags"]["height:source"] == HEIGHT_FROM_TOWN  # 26 ft
    assert stats["given"] == 1
    height, _, source = building_height(out[0]["tags"])
    assert (round(height, 1), source) == (7.9, "town")


def test_the_input_list_is_not_changed():
    source = [way(1, 0, 0)]
    heights(source, footprint(0, 0))
    assert "height" not in source[0]["tags"]


def test_openstreetmaps_own_height_or_levels_win():
    source = [way(1, 0, 0, height="30"), way(2, 20, 0, **{"building:levels": "5"}), way(3, 40, 0, **{"min_height": "10"}),
              way(4, 60, 0, **{"building:part": "yes"}), way(5, 80, 0, building="garage"), way(6, 100, 0, building="roof")]
    source[3]["tags"].pop("building")
    out, stats = heights(source, *[footprint(x, 0, height=65, stories=5) for x in (0, 20, 40, 60, 80, 100)])
    assert [e["tags"].get("height") for e in out] == ["30", None, None, "19.8", None, None]
    assert stats["given"] == 1  # the part with no height of its own: the others keep what they say


def test_a_footprint_must_overlap_half_of_the_smaller_building_and_cover_it_by_a_third():
    source = [way(1, 0, 0), way(2, 100, 0), way(3, 200, 0, w=40, h=40)]
    records = [
        footprint(7, 0, height=26),  # shifted 7 m: 30% of the 10 m building. Too little.
        footprint(100, 0, height=26),  # exactly on the second building
        footprint(205, 5, w=10, h=10, height=39),  # inside a much bigger building: matches (all of the smaller), but covers 6% of it
    ]
    out, stats = heights(source, *records)
    assert [e["tags"].get("height") for e in out] == [None, "7.9", None]
    assert stats["uncovered"] == 1


def test_several_footprints_on_one_building_give_the_figure_that_covers_most_of_its_area():
    # A 30 x 12 m block mapped as one building; the Town splits it into a 3-storey house (24 m) and two small 6-storey annexes.
    source = [way(1, 0, 0, w=30, h=12)]
    out, _ = heights(source, footprint(0, 0, w=24, h=12, height=35.75, stories=3), footprint(24, 0, w=3, h=12, height=78, stories=6, lidar=75.0),
                     footprint(27, 0, w=3, h=12, height=78, stories=6, lidar=75.0))
    assert out[0]["tags"]["height"] == f"{35.75 * 0.3048:.1f}"


def test_one_footprint_on_several_building_parts_gives_each_part_the_figure():
    source = [way(1, 0, 0, w=10, h=12, **{"building:part": "yes"}), way(2, 10, 0, w=10, h=12, **{"building:part": "yes"})]
    for element in source:
        element["tags"].pop("building")
    out, _ = heights(source, footprint(0, 0, w=20, h=12, height=52, stories=4))
    assert [e["tags"]["height"] for e in out] == ["15.8", "15.8"]


def test_outbuilding_footprints_count_only_for_a_building_with_no_other():
    source = [way(1, 0, 0, w=10, h=12), way(2, 50, 0, w=5, h=6)]
    out, stats = heights(
        source,
        footprint(0, 0, w=10, h=12, height=26),
        footprint(0, 0, w=10, h=12, kind="OUTBLDG", height=26, lidar=11.0),  # the same spot listed twice: the house wins
        footprint(50, 0, w=5, h=6, kind="OUTBLDG", height=26, lidar=11.0),
    )
    assert [e["tags"]["height"] for e in out] == ["7.9", "3.4"] and stats["from_outbuilding"] == 1


def test_a_multipolygon_building_gets_a_height_too():
    ring = [{"lon": lon, "lat": lat} for lon, lat in rect(ORIGIN[0], ORIGIN[1], 10, 12)]
    relation = {"type": "relation", "id": 9, "tags": {"building": "yes", "type": "multipolygon"}, "members": [{"type": "way", "role": "outer", "geometry": ring}]}
    out, _ = heights([relation], footprint(0, 0, height=39, stories=3))
    assert out[0]["tags"]["height"] == "11.9"


def test_buildings_outside_the_towns_footprints_are_untouched():
    out, stats = heights([way(1, 500, 500)], footprint(0, 0))
    assert out[0]["tags"] == {"building": "yes"} and stats["given"] == 0


def test_a_fix_that_sets_levels_or_height_is_never_overruled():
    # `_scene` applies the heights after the fixes: an element a fix gave `building:levels` has one, so it is skipped.
    fixed = way(1, 0, 0, **{"building:levels": "7"})
    assert heights([fixed], footprint(0, 0))[0][0] is fixed


def test_the_scene_reports_the_height_source_as_town():
    out, _ = heights([way(1, 0, 0), way(2, 50, 0)], footprint(0, 0, height=39, stories=3))
    scene = classify(out)
    assert {(b.id, b.height_source) for b in scene.buildings} == {("w1", "town"), ("w2", "default")}
    assert next(b for b in scene.buildings if b.id == "w1").height == pytest.approx(11.9, abs=0.05)


def test_an_openstreetmap_height_with_the_towns_source_tag_is_not_mistaken_for_one_of_its_own():
    assert building_height({"building": "yes", "height": "10"})[2] == "height"
    assert building_height({"building": "yes", "height": "10", "height:source": HEIGHT_FROM_TOWN})[2] == "town"
    assert building_height({"building": "yes", "height:source": HEIGHT_FROM_TOWN})[2] == "default"  # no height tag: the source says nothing


# ------------------------------------------------------------------------------------------------- areas, credit, switch


def test_only_the_boston_area_asks_for_the_towns_data():
    areas = cli._areas()
    assert "brookline" in cli._extra_sources(areas["boston"])
    assert "brookline" not in cli._extra_sources(areas["pilot"])  # the pilot is downtown; it asks for the city's heights only (test_cityheights.py)
    with pytest.raises(SystemExit, match="unknown extra_sources"):
        cli._extra_sources({"extra_sources": ["nowhere"]})


def test_the_credit_names_the_town_only_where_its_data_is_used():
    assert tiles.attribution({}) == tiles.ATTRIBUTION == tiles.attribution(None)
    assert "Brookline" not in tiles.attribution(cli._areas()["pilot"])
    credit = tiles.attribution(cli._areas()["boston"])
    assert credit.startswith(tiles.ATTRIBUTION) and "Town of Brookline tree inventory and building footprints" in credit


def _write_area(root, extra):
    (root / "areas.toml").write_text(f'[t]\nname = "t"\nbbox = [-71.13, 42.33, -71.11, 42.35]\n{extra}\n', encoding="utf-8")
    folder = root / "data" / "raw" / "t"
    folder.mkdir(parents=True)
    osm_payload = {"osm3s": {}, "elements": [way(1, 0, 0), {"type": "node", "id": 2, "lon": _lonlat(*ORIGIN)[0], "lat": _lonlat(*ORIGIN)[1], "tags": {"natural": "tree"}}]}
    (folder / "osm.json").write_text(json.dumps(osm_payload), encoding="utf-8")
    (folder / brookline.TREES_FILE).write_text(json.dumps({"features": [tree("11"), tree("12")]}), encoding="utf-8")
    (folder / brookline.BUILDINGS_FILE).write_text(json.dumps({"features": [footprint(0, 0, height=39, stories=3)]}), encoding="utf-8")


def test_the_towns_data_is_used_when_the_area_asks_for_it_and_ignored_otherwise(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "FIXES", tmp_path / "fixes")
    _write_area(tmp_path, 'extra_sources = ["brookline"]')
    scene, payload, _ = cli._scene("t")
    assert sorted(t.id for t in scene.trees if t.id.startswith("brk:")) == ["brk:11", "brk:12"]
    assert [b.height_source for b in scene.buildings] == ["town"]
    assert payload["brookline_tree_counts"]["inventory"] == 2 and payload["brookline_heights"]["given"] == 1
    assert payload["tree_counts"]["inventory"] == 0

    # One line turns it off: the files are still there and nothing of the Town's is read.
    (tmp_path / "areas.toml").write_text('[t]\nname = "t"\nbbox = [-71.13, 42.33, -71.11, 42.35]\n', encoding="utf-8")
    scene, payload, _ = cli._scene("t")
    assert [t.id for t in scene.trees] == ["n2"] and [b.height_source for b in scene.buildings] == ["default"]
    assert "brookline_tree_counts" not in payload and "brookline_heights" not in payload


def test_a_fix_can_delete_or_change_a_town_tree_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "FIXES", tmp_path / "fixes")
    _write_area(tmp_path, 'extra_sources = ["brookline"]')
    (tmp_path / "fixes").mkdir()
    (tmp_path / "fixes" / "brookline.toml").write_text(
        '["brk:11"]\nwhy = "Tree that is gone"\ndelete = true\n\n["brk:12"]\nwhy = "Wrong species"\nset = { species = "Quercus rubra" }\n', encoding="utf-8"
    )
    scene, _, applied = cli._scene("t")
    assert [t.id for t in scene.trees if t.id.startswith("brk:")] == ["brk:12"]
    assert next(t for t in scene.trees if t.id == "brk:12").tags["species"] == "Quercus rubra"
    assert applied.unmatched == []


# ------------------------------------------------------------------------------------------------- download


class _Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def test_fetch_pages_through_both_layers_and_caches_them(tmp_path, monkeypatch):
    urls = []

    def fake_urlopen(request, timeout=None):
        urls.append(request.full_url)
        assert "orderByFields=OBJECTID" in request.full_url and "f=geojson" in request.full_url and "outFields=%2A" in request.full_url
        offset = int(request.full_url.split("resultOffset=")[1].split("&")[0])
        full = offset == 0
        return _Reply({"features": [tree(str(i + offset)) for i in range(brookline.PAGE_SIZE if full else 3)], **({"properties": {"exceededTransferLimit": True}} if full else {})})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    paths = brookline.fetch((-71.14, 42.33, -71.04, 42.37), tmp_path)
    assert [p.name for p in paths] == [brookline.TREES_FILE, brookline.BUILDINGS_FILE]
    assert [u.split("?")[0] for u in urls] == [brookline.TREES_ENDPOINT] * 2 + [brookline.BUILDINGS_ENDPOINT] * 2
    saved = json.loads(paths[0].read_text())
    assert len(saved["features"]) == brookline.PAGE_SIZE + 3 and saved["credit"] == "Town of Brookline" and "none stated" in saved["licence"]
    brookline.fetch((-71.14, 42.33, -71.04, 42.37), tmp_path)  # cached: no further requests
    assert len(urls) == 4


def test_fetch_gives_up_with_a_clear_error(tmp_path, monkeypatch):
    def failing(request, timeout=None):
        raise OSError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    monkeypatch.setattr(brookline, "RETRY_PAUSE", 0)
    with pytest.raises(RuntimeError, match="Brookline_Tree_Viewer_Web_WFL1 failed 3 times"):
        brookline.fetch((0, 0, 1, 1), tmp_path)

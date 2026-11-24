"""Park detail: surface types, planted beds, furniture, monuments and footbridges."""

import json
import struct

import numpy as np
import pytest
from shapely.geometry import LineString, box

from boston_pipeline import world
from boston_pipeline.features import (
    Area,
    Object,
    Scene,
    Way,
    bridge_rise,
    classify,
    green_cover,
    parse_direction,
    path_cover,
)
from boston_pipeline.mesh import MeshBuilder, add_flat, add_walls, colour_bytes, weld
from boston_pipeline.tiles import BED_HEIGHT, COVERS, _by_cover, _by_size, build


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"leisure": "garden", "garden:type": "flowerbed"}, ("bed", "flowers")),
        ({"landuse": "flowerbed"}, ("bed", "flowers")),
        ({"leisure": "garden"}, ("green", "garden")),
        ({"leisure": "pitch", "surface": "artificial_turf"}, ("green", "turf")),
        ({"leisure": "pitch", "sport": "tennis", "surface": "acrylic"}, ("green", "court")),
        ({"leisure": "pitch", "sport": "baseball", "surface": "grass"}, ("green", "pitch")),
        ({"leisure": "playground"}, ("green", "playground")),
        ({"leisure": "dog_park", "surface": "dirt"}, ("green", "dirt")),
        ({"landuse": "grass"}, ("green", "grass")),
        ({"natural": "wood"}, ("green", "wood")),
        ({"leisure": "park"}, ("green", "lawn")),
    ],
)
def test_green_cover(tags, expected):
    assert green_cover(tags) == expected


def test_path_cover_reads_the_surface_tag():
    assert path_cover({"highway": "footway", "surface": "brick"}) == "brick"
    assert path_cover({"highway": "footway", "surface": "fine_gravel"}) == "gravel"
    assert path_cover({"highway": "footway"}) == "paved"
    assert path_cover({"man_made": "pier"}) == "wood"
    # Every cover a feature can be given has a colour to be drawn in.
    assert {"paved", "wood", *[path_cover({"surface": s}) for s in ("asphalt", "concrete", "sett", "sand", "dirt", "metal")]} <= set(COVERS["paths"])


def test_parse_direction():
    assert parse_direction("165") == 165
    assert parse_direction("NE") == 45
    assert parse_direction("10;190") == 10
    assert parse_direction("-90") == 270
    assert parse_direction("sideways") is None and parse_direction(None) is None


def test_bridge_rise_grows_with_length_within_limits():
    assert bridge_rise(10) == pytest.approx(0.3)
    assert bridge_rise(20) == pytest.approx(0.5)
    assert bridge_rise(39.5) == pytest.approx(0.8)


def _node(ident, tags, x=5.0, y=5.0):
    lon, lat = (float(v) for v in world.to_lonlat(x, y))
    return {"type": "node", "id": ident, "lon": lon, "lat": lat, "tags": tags}


def test_classify_reads_furniture_and_monuments():
    elements = [
        _node(1, {"amenity": "bench", "direction": "300", "backrest": "no", "seats": "3", "height": "0.5"}),
        _node(2, {"highway": "street_lamp", "lamp_mount": "ground"}),
        _node(3, {"highway": "street_lamp", "lamp_mount": "bent_mast", "height": "7"}),
        _node(4, {"tourism": "artwork", "artwork_type": "statue", "name": "A Hero", "material": "bronze;granite"}),
        _node(5, {"historic": "memorial", "memorial": "plaque", "support": "fence"}),
        _node(6, {"amenity": "fountain", "fountain": "nozzle", "height": "1"}),
        _node(7, {"amenity": "waste_basket", "indoor": "yes"}),
        _node(8, {"historic": "monument", "name": "The Column", "height": "20'"}),
        _node(9, {"historic": "monument", "tourism": "artwork", "artwork_type": "statue", "name": "The Rider"}),
    ]
    scene = classify(elements)
    things = {thing.id: thing for thing in scene.objects}
    assert set(things) == {"n1", "n3", "n4", "n6", "n8", "n9"}  # ground light, fixed plaque and indoor bin are not drawn
    bench = things["n1"]
    assert (bench.kind, bench.direction, bench.variant, bench.height) == ("bench", 300, "backless", None)
    assert bench.size == pytest.approx(1.8)
    assert (things["n3"].kind, things["n3"].variant, things["n3"].height) == ("lamp", "arm", 7.0)
    assert (things["n4"].kind, things["n4"].name, things["n4"].material) == ("statue", "A Hero", "bronze")
    assert (things["n6"].kind, things["n6"].variant) == ("fountain", "nozzle")
    assert (things["n8"].kind, things["n8"].height) == ("monument", pytest.approx(6.096))
    assert things["n9"].kind == "statue"  # a monument that is a statue is drawn as one
    # A statue is drawn and is also a named place to be near.
    assert {poi.name for poi in scene.pois} == {"A Hero", "The Column", "The Rider"}


def test_classify_marks_bridges_beds_and_covers():
    def way(ident, tags, points):
        return {"type": "way", "id": ident, "tags": tags, "geometry": [dict(zip(("lon", "lat"), map(float, world.to_lonlat(*p)))) for p in points]}

    square = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    scene = classify(
        [
            way(1, {"highway": "footway", "bridge": "yes", "surface": "wood"}, [(0, 20), (30, 20)]),
            way(2, {"leisure": "garden", "garden:type": "flowerbed"}, square),
            way(3, {"leisure": "pitch", "surface": "artificial_turf"}, [(x + 50, y) for x, y in square]),
        ]
    )
    assert [(w.id, w.bridge, w.cover) for w in scene.ways] == [("w1", True, "wood")]
    assert [a.id for a in scene.areas["bed"]] == ["w2"]
    assert [(a.id, a.cover) for a in scene.areas["green"]] == [("w3", "turf")]


def _area_by_cover(pieces):
    totals = {}
    for polygon, cover in pieces:
        totals[cover] = totals.get(cover, 0.0) + polygon.area
    return totals


def test_smaller_green_areas_show_on_top_of_larger_ones():
    lawn, court = box(0, 0, 10, 10), box(2, 2, 4, 4)
    totals = _area_by_cover(_by_size(lawn, [(lawn, "lawn"), (court, "court")], "lawn"))
    assert totals == pytest.approx({"court": 4.0, "lawn": 96.0})
    # One kind only: nothing to split.
    assert _area_by_cover(_by_size(lawn, [(lawn, "lawn")], "lawn")) == pytest.approx({"lawn": 100.0})


def test_overlapping_paths_show_the_cover_listed_first():
    strip = box(0, 0, 10, 2)
    labelled = [(box(0, 0, 6, 2), "asphalt"), (box(4, 0, 10, 2), "brick")]
    totals = _area_by_cover(_by_cover(strip, labelled, list(COVERS["paths"])))
    assert totals == pytest.approx({"brick": 12.0, "asphalt": 8.0})


def test_weld_shares_vertices_without_moving_triangles():
    builder = MeshBuilder("box", colours=True)
    white = colour_bytes("#ffffff")
    add_walls(builder, box(0, 0, 2, 2), 0.0, 1.0, np.zeros(2), white)
    add_flat(builder, box(0, 0, 2, 2), 1.0, np.zeros(2), white)
    mesh = builder.finish()
    welded = weld(mesh)
    assert len(mesh.positions) == 20 and len(welded.positions) == 8
    before = mesh.positions[mesh.indices].reshape(-1, 3, 3)
    after = welded.positions[welded.indices].reshape(-1, 3, 3)
    assert np.array_equal(before, after)
    # Vertices that differ in colour stay apart.
    add_flat(builder, box(0, 0, 2, 2), 1.0, np.zeros(2), colour_bytes("#000000"))
    assert len(weld(builder.finish()).positions) == 12


def _read_glb(path):
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + length])


def test_build_draws_a_park(tmp_path):
    scene = Scene()
    scene.areas["water"].append(Area("w10", box(50, 50, 150, 110), {"natural": "water"}))
    scene.areas["green"].append(Area("w31", box(160, 0, 256, 256), {"leisure": "park", "name": "Test Park"}, "lawn"))
    scene.areas["green"].append(Area("w32", box(220, 100, 250, 140), {"leisure": "pitch"}, "turf"))
    scene.areas["bed"].append(Area("w30", box(180, 20, 190, 30), {"leisure": "garden", "garden:type": "flowerbed"}, "flowers"))
    bridge = {"highway": "footway", "bridge": "yes"}
    scene.ways += [
        # A footbridge of its own across the pond, and a sidewalk on a road bridge beside it.
        Way("w11", LineString([(100, 40), (100, 120)]), "footway", "path", 3.0, "Pond Bridge", bridge, "wood", True),
        Way("w12", LineString([(130, 0), (130, 256)]), "residential", "road", 7.0, "Pond Road"),
        Way("w13", LineString([(135.5, 40), (135.5, 120)]), "footway", "path", 2.2, None, bridge, "concrete", True),
        Way("w14", LineString([(170, 10), (250, 10)]), "footway", "path", 2.2, None, {}, "brick"),
    ]
    scene.objects += [
        Object("n20", 97.0, 80.0, "lamp"),  # in the water beside the bridge: stands on a pier
        Object("n21", 200.0, 200.0, "bench", direction=90.0),
        Object("n22", 210.0, 220.0, "statue", "A Hero"),
    ]

    stats = build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    assert stats["bridges"] == 1 and stats["objects"] == 3

    document = _read_glb(tmp_path / "tiles" / "0_0.glb")
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    assert {"green", "beds", "paths", "water", "roads", "furniture", "monuments", "bridges"} <= set(layers)
    # No layer stores normals; small layers index their vertices with 16 bits.
    assert all("NORMAL" not in layer["attributes"] for layer in layers.values())
    assert document["accessors"][layers["beds"]["indices"]]["componentType"] == 5123
    for name in ("green", "paths", "beds", "furniture", "monuments", "bridges"):
        assert "COLOR_0" in layers[name]["attributes"], name

    def top(name):
        return document["accessors"][layers[name]["attributes"]["POSITION"]]["max"][1]

    assert top("beds") == pytest.approx(BED_HEIGHT)
    assert top("bridges") == pytest.approx(0.03 + 0.8 + 1.125, abs=0.01)  # the top rail at mid-span
    # Only the stretch over the pond (y 50 to 110, plus a short landing) is raised;
    # the rest of the bridge way lies on the banks as ordinary path. glTF z is -north.
    deck = document["accessors"][layers["bridges"]["attributes"]["POSITION"]]
    assert -deck["max"][2] == pytest.approx(49.7, abs=0.2) and -deck["min"][2] == pytest.approx(110.3, abs=0.2)
    assert top("furniture") == pytest.approx(0.03 + 0.8 + 1.1 + 2.8, abs=0.05)  # the lamp on its pier

    semantics = json.loads((tmp_path / "tiles" / "0_0.json").read_text(encoding="utf-8"))
    things = {entry["id"]: entry for entry in semantics["objects"]}
    assert {entry["kind"] for entry in things.values()} == {"bridge", "lamp", "bench", "statue"}
    assert things["w11"]["name"] == "Pond Bridge" and things["n22"]["name"] == "A Hero" and "w13" not in things
    for name in ("furniture", "monuments", "bridges"):
        ids = layers[name]["extensions"]["EXT_mesh_features"]["featureIds"][0]
        assert ids["featureCount"] <= len(semantics["objects"])
    (park,) = semantics["areas"]
    assert (park["id"], park["name"], park["kind"], park["area"]) == ("w31", "Test Park", "park", 96 * 256)
    assert len(park["rings"]) == 1 and len(park["rings"][0]) == 5

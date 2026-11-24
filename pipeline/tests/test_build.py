import json
import struct

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from boston_pipeline import world
from boston_pipeline.features import Building, Point, Scene, Way
from boston_pipeline.mesh import MeshBuilder, add_flat, add_skirt, add_walls, colour_bytes, triangulate
from boston_pipeline.tiles import build, sea_polygons


def _triangle_area(vertices, triangles):
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def test_triangulate_respects_holes_and_winds_counter_clockwise():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (4, 6), (6, 6), (6, 4)]])
    areas = _triangle_area(*triangulate(polygon))
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(96.0)


def test_walls_face_outward_and_count_facade_cells():
    builder = MeshBuilder("walls", uvs=True, colours=True)
    add_walls(builder, box(0, 0, 9, 6), 0.0, 7.0, np.zeros(2), colour_bytes("#ffffff"))
    mesh = builder.finish()
    assert len(mesh.positions) == 16 and len(mesh.indices) == 24
    # glTF axes are (east, up, -north); push each vertex along its normal and it must leave the box.
    east, north = mesh.positions[:, 0] + mesh.normals[:, 0], -(mesh.positions[:, 2] + mesh.normals[:, 2])
    assert ((east < 0) | (east > 9) | (north < 0) | (north > 6)).all()
    assert mesh.uvs[:, 0].max() == 3.0  # 9 m wall = 3 bays
    assert mesh.uvs[:, 1].max() == 2.0  # 7 m tall = 2 storeys
    # Triangle winding must agree with the stored normals.
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()


def test_skirt_skips_edges_on_the_tile_boundary():
    builder = MeshBuilder("curbs")
    # A road strip crossing the whole tile west to east: only its two long sides get curbs.
    add_skirt(builder, box(0, 100, 256, 110), -0.12, 0.0, np.zeros(2), (0, 0, 256, 256))
    mesh = builder.finish()
    assert len(mesh.positions) == 8
    assert np.allclose(np.abs(mesh.normals[:, 2]), 1.0)  # facing north or south, into the road


def test_flat_faces_point_up_or_down():
    for up, expected in ((True, 1.0), (False, -1.0)):
        builder = MeshBuilder("flat")
        add_flat(builder, box(0, 0, 4, 4), 2.0, np.zeros(2), up=up)
        mesh = builder.finish()
        tri = mesh.positions[mesh.indices.reshape(-1, 3)]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert (np.sign(face[:, 1]) == expected).all() and (mesh.normals[:, 1] == expected).all()


def test_sea_is_on_the_right_of_a_coastline():
    extent = box(0, 0, 100, 100)
    northbound = LineString([(60, -10), (60, 110)])  # land to the west, sea to the east
    sea = sea_polygons([northbound], extent)
    assert len(sea) == 1 and sea[0].bounds == (60, 0, 100, 100)
    assert sea_polygons([LineString(northbound.coords[::-1])], extent)[0].bounds == (0, 0, 60, 100)
    assert sea_polygons([], extent) == []


def _read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    assert json_type == b"JSON" and json_length % 4 == 0
    document = json.loads(data[20 : 20 + json_length])
    bin_length, bin_type = struct.unpack_from("<I4s", data, 20 + json_length)
    assert bin_type == b"BIN\0" and bin_length == document["buffers"][0]["byteLength"]
    return document


def test_build_writes_a_loadable_tile_and_tileset(tmp_path):
    scene = Scene()
    scene.buildings.append(Building("w1", box(100, 100, 120, 130), 20.0, 0.0, "height", "#8a4b3c", "#5a5a5c", {"name": "Test Hall"}))
    scene.ways.append(Way("w2", LineString([(0, 50), (256, 50)]), "residential", "road", 7.0, "Test Street"))
    scene.areas["green"].append(box(150, 150, 250, 250))
    scene.areas["water"].append(box(10, 150, 60, 250))
    scene.trees.append(Point("n3", 200.0, 200.0))
    scene.pois.append(Point("n4", 110.0, 95.0, "A Bookshop", "shop=books"))
    area = {"name": "Test", "places": [{"name": "Here", "lat": 42.3587, "lon": -71.063, "heading": 0}]}

    stats = build(scene, [(0, 0)], tmp_path, area, "2026-01-01T00:00:00Z")
    assert stats["tiles"] == 1 and stats["buildings"] == 1

    document = _read_glb(tmp_path / "tiles" / "0_0.glb")
    layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
    assert set(layers) == {"ground", "green", "water", "roads", "curbs", "building_walls", "building_roofs", "trees"}
    assert "TEXCOORD_0" in layers["building_walls"]["attributes"]
    assert "© OpenStreetMap" in document["asset"]["copyright"]
    road = document["accessors"][layers["roads"]["attributes"]["POSITION"]]
    assert road["min"][1] == pytest.approx(-0.12) and road["max"][1] == pytest.approx(-0.12)

    tileset = json.loads((tmp_path / "tileset.json").read_text(encoding="utf-8"))
    child = tileset["root"]["children"][0]
    assert tileset["asset"]["version"] == "1.1" and child["content"]["uri"] == "tiles/0_0.glb"
    centre, half_height = child["boundingVolume"]["box"][2], child["boundingVolume"]["box"][11]
    assert centre + half_height == pytest.approx(20.0, abs=0.01)  # the building's roof
    assert tileset["extras"]["world"]["tile_size"] == world.TILE_SIZE
    assert tileset["extras"]["places"][0]["pos"] == pytest.approx([0, 0], abs=5)

    semantics = json.loads((tmp_path / "tiles" / "0_0.json").read_text(encoding="utf-8"))
    assert semantics["streets"][0]["name"] == "Test Street"
    assert semantics["buildings"][0]["name"] == "Test Hall"
    assert semantics["pois"][0]["kind"] == "shop=books"

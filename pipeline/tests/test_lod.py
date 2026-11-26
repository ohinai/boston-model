"""Levels of detail (lod.py): the simplified cells written beside the full tiles."""

import json
import struct

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from boston_pipeline import lod, tiles, world
from boston_pipeline.features import Area, Building, Point, RoofTags, Scene, Way
from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.trees import describe

SIZE = world.TILE_SIZE


def read_glb(path):
    """(document, binary chunk) of a .glb file."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    json_length, _ = struct.unpack_from("<I4s", data, 12)
    document = json.loads(data[20 : 20 + json_length])
    bin_length, _ = struct.unpack_from("<I4s", data, 20 + json_length)
    return document, data[28 + json_length : 28 + json_length + bin_length]


def mesh_arrays(document, binary, mesh_index):
    """(positions (n, 3), triangles (m, 3)) of a mesh, in glTF axes."""
    primitive = document["meshes"][mesh_index]["primitives"][0]

    def read(accessor_index, dtype, width):
        accessor = document["accessors"][accessor_index]
        view = document["bufferViews"][accessor["bufferView"]]
        count = accessor["count"] * width
        return np.frombuffer(binary, dtype=dtype, count=count, offset=view["byteOffset"]).reshape(-1, width)

    positions = read(primitive["attributes"]["POSITION"], np.float32, 3)
    index_accessor = document["accessors"][primitive["indices"]]
    dtype = np.uint16 if index_accessor["componentType"] == 5123 else np.uint32
    return positions, read(primitive["indices"], dtype, 1).reshape(-1, 3)


def cell_nodes(document):
    """name -> node, for the cells that are the file's scene roots."""
    return {document["nodes"][i]["name"]: document["nodes"][i] for i in document["scenes"][0]["nodes"]}


def land(document, binary, cell):
    """(positions, triangles) of a cell's `lod_land` mesh."""
    for child in cell_nodes(document)[cell]["children"]:
        if document["nodes"][child]["name"] == "lod_land":
            return mesh_arrays(document, binary, document["nodes"][child]["mesh"])
    raise AssertionError(f"{cell} has no lod_land")


def building(ident, polygon, height, min_height=0.0, wall="#8a4b3c", roof_colour="#5a5a5c", **more) -> Building:
    return Building(ident, polygon, height, min_height, "height", wall, roof_colour, {}, **more)


def triangle_area(positions, triangles):
    a, b, c = (positions[triangles[:, i]][:, [0, 2]] for i in range(3))
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def test_super_tiles_group_base_tiles_by_floor_division():
    assert lod.SUPER == 4
    assert lod.super_tile(0, 0) == lod.super_tile(3, 3) == (0, 0)
    assert lod.super_tile(4, 0) == (1, 0)
    assert lod.super_tile(-1, -1) == lod.super_tile(-4, -4) == (-1, -1)
    assert lod.super_tile(-5, -4) == (-2, -1)


def test_cells_are_nodes_of_their_super_tile_file_and_the_tileset_lists_them(tmp_path):
    scene = Scene()
    scene.buildings.append(building("w1", box(100, 100, 130, 130), 20.0))
    cells = [(0, 0), (1, 0), (3, 3), (4, 0), (-1, -1)]
    area = {"name": "Test", "places": []}
    stats = tiles.build(scene, cells, tmp_path, area, "2026-01-01T00:00:00Z")

    tileset = json.loads((tmp_path / "tileset.json").read_text(encoding="utf-8"))
    levels = tileset["extras"]["lod"]["levels"]
    assert len(levels) == 1 and levels[0]["level"] == 1 and levels[0]["super"] == 4
    files = {tuple(entry["super"]): entry for entry in levels[0]["tiles"]}
    assert set(files) == {(0, 0), (1, 0), (-1, -1)}
    assert files[(0, 0)]["uri"] == "lod1/0_0.glb" and files[(-1, -1)]["uri"] == "lod1/-1_-1.glb"
    assert files[(0, 0)]["cells"] == [[0, 0], [1, 0], [3, 3]]
    assert stats["lod"]["files"] == 3 and stats["lod"]["cells"] == 5 and stats["lod"]["triangles"] > 0

    document, _ = read_glb(tmp_path / "lod1" / "0_0.glb")
    nodes = cell_nodes(document)
    assert set(nodes) == {"cell_0_0", "cell_1_0", "cell_3_3"}
    # Each cell is placed relative to the super tile's south-west corner: glTF axes are (east, up, south).
    assert nodes["cell_0_0"].get("translation") == [0, 0, 0]
    assert nodes["cell_1_0"]["translation"] == [SIZE, 0, 0]
    assert nodes["cell_3_3"]["translation"] == [3 * SIZE, 0, -3 * SIZE]
    document, _ = read_glb(tmp_path / "lod1" / "-1_-1.glb")
    assert cell_nodes(document)["cell_-1_-1"]["translation"] == [3 * SIZE, 0, -3 * SIZE]  # tile -1 is the fourth of the block -4..-1
    # The layers are the cell's children, and the full tiles are untouched by all this.
    assert {document["nodes"][i]["name"] for i in cell_nodes(document)["cell_-1_-1"]["children"]} <= {"lod_land", "lod_water"}
    assert (tmp_path / "tiles" / "0_0.glb").exists()


def test_a_cell_with_nothing_in_it_writes_no_node(tmp_path, monkeypatch):
    real = lod.build_cell
    monkeypatch.setattr(lod, "build_cell", lambda ctx, tx, ty: [] if (tx, ty) == (1, 0) else real(ctx, tx, ty))
    ctx = tiles.prepare(Scene(), [(0, 0), (1, 0), (8, 8)])
    info = lod.build(ctx, [(0, 0), (1, 0), (8, 8)], tmp_path)
    listed = {tuple(entry["super"]): entry["cells"] for entry in info["extras"]["levels"][0]["tiles"]}
    assert listed == {(0, 0): [[0, 0]], (2, 2): [[8, 8]]}
    assert set(cell_nodes(read_glb(tmp_path / "lod1" / "0_0.glb")[0])) == {"cell_0_0"}

    monkeypatch.setattr(lod, "build_cell", lambda ctx, tx, ty: [])
    empty = lod.build(ctx, [(0, 0)], tmp_path)
    assert empty["extras"]["levels"][0]["tiles"] == [] and not list((tmp_path / "lod1").iterdir())

    builder = MeshBuilder("lod_land", colours=True)
    lod.add_buildings(builder, [], np.zeros(2))
    lod.add_trees(builder, [], np.zeros(2))
    assert builder.empty


def test_output_is_deterministic(tmp_path):
    scene = Scene()
    scene.buildings.append(building("w1", box(40, 40, 70, 90), 22.0, wall="#6f8796"))
    scene.buildings.append(building("w2", box(120, 30, 160, 60), 9.0))
    scene.ways.append(Way("w3", LineString([(0, 120), (300, 130)]), "residential", "road", 7.0, "Test Street"))
    scene.areas["green"].append(Area("w4", box(150, 150, 250, 250), {"leisure": "park", "name": "Test Park"}, "grass"))
    scene.areas["water"].append(Area("w5", box(10, 150, 60, 250)))
    scene.trees.append(Point("n6", 200.0, 200.0, tags={"natural": "tree", "genus": "Quercus", "height": "18", "diameter_crown": "14"}))
    for name in ("a", "b"):
        tiles.build(scene, [(0, 0), (1, 0)], tmp_path / name, {"name": "Test", "places": []}, "2026-01-01T00:00:00Z")
    first, second = (tmp_path / name / "lod1" / "0_0.glb" for name in ("a", "b"))
    assert first.read_bytes() == second.read_bytes()


def test_footprint_simplification_keeps_the_area_and_drops_the_jogs():
    # A 30 x 20 m building whose long sides have 0.4 m jogs every 3 m.
    top = [(x, 20.0 + (0.4 if (x // 3) % 2 else 0.0)) for x in np.arange(30.0, -0.01, -1.5)]
    jagged = Polygon([(0, 0), (30, 0), (30, 20), *top, (0, 20)])
    assert jagged.is_valid
    simple = lod.simplified_footprint(jagged)
    assert abs(simple.area - jagged.area) <= 0.03 * jagged.area
    assert len(simple.exterior.coords) < len(jagged.exterior.coords)
    # A small building is hardly simplified at all, and never collapses.
    small = Polygon([(0, 0), (6, 0), (6, 4), (3, 4.5), (0, 4)])
    kept = lod.simplified_footprint(small)
    assert kept.is_valid and abs(kept.area - small.area) <= 0.06 * small.area
    # A courtyard is kept when it is big, and filled in when it is small.
    big_yard = Polygon(box(0, 0, 60, 60).exterior, [box(20, 20, 40, 40).exterior])
    small_yard = Polygon(box(0, 0, 60, 60).exterior, [box(28, 28, 32, 32).exterior])
    assert len(lod.simplified_footprint(big_yard).interiors) == 1
    assert len(lod.simplified_footprint(small_yard).interiors) == 0


def lod_walls(buildings):
    builder = MeshBuilder("lod_land", colours=True)
    lod.add_buildings(builder, buildings, np.zeros(2))
    return builder


def test_sheds_are_dropped_but_tall_or_wide_buildings_are_not():
    assert lod_walls([building("shed", box(0, 0, 5, 5), 3.0)]).empty  # under 60 m2 and under 6 m
    assert not lod_walls([building("tower", box(0, 0, 5, 5), 30.0)]).empty  # small but tall
    assert not lod_walls([building("barn", box(0, 0, 12, 12), 3.0)]).empty  # low but 144 m2
    assert not lod_walls([building("house", box(0, 0, 8, 8), 8.0)]).empty


def test_a_part_on_a_podium_has_no_wall_below_its_min_height():
    mesh = lod_walls([building("w1", box(0, 0, 20, 20), 40.0, min_height=12.0)]).finish()
    assert mesh.positions[:, 1].min() == pytest.approx(12.0)  # glTF y is up
    assert mesh.positions[:, 1].max() == pytest.approx(40.0)
    plain = lod_walls([building("w2", box(0, 0, 20, 20), 40.0)]).finish()
    assert plain.positions[:, 1].min() == pytest.approx(0.0)


def test_a_pitched_roof_is_drawn_at_its_eaves_plus_part_of_its_rise():
    house = building("w1", box(0, 0, 8, 14), 12.0, roof=RoofTags("gable", height=4.0))
    assert lod.top_of(house) == pytest.approx(8.0 + lod.ROOF_SHARE * 4.0)
    flat = building("w2", box(0, 0, 8, 14), 12.0)
    assert lod.top_of(flat) == 12.0
    mesh = lod_walls([house]).finish()
    assert mesh.positions[:, 1].max() == pytest.approx(lod.top_of(house))


def test_a_building_hidden_under_another_is_dropped_but_a_floating_part_is_not():
    tower = building("tower", box(0, 0, 20, 20), 60.0)
    duplicate = building("duplicate", box(0, 0, 20, 20), 60.0)
    podium = building("podium", box(2, 2, 18, 18), 12.0)
    assert len(lod_walls([tower, duplicate]).finish().indices) == len(lod_walls([tower]).finish().indices)
    assert len(lod_walls([tower, podium]).finish().indices) == len(lod_walls([tower]).finish().indices)
    beside = building("beside", box(30, 0, 50, 20), 60.0)
    assert len(lod_walls([tower, beside]).finish().indices) == 2 * len(lod_walls([tower]).finish().indices)
    # A part that starts higher than the building under it is not hidden by it: it adds a roof the other lacks.
    short = building("short", box(0, 0, 20, 20), 20.0)
    floating = building("deck", box(2, 2, 18, 18), 30.0, min_height=20.0)
    assert len(lod_walls([short, floating]).finish().indices) > len(lod_walls([short]).finish().indices)


def test_walls_are_the_colour_the_facade_shader_fades_to():
    brick = lod.far_wall_colour("#8a4b3c", 3)
    assert brick[3] == 255 and brick[0] > brick[1] > brick[2]  # still reddish
    glass_hex = "#6f8796"
    glass = lod.far_wall_colour(glass_hex, 20)  # cool and tall: a curtain wall
    stone = lod.far_wall_colour(glass_hex, 3)  # the same colour on a low building is stone
    assert int(glass[:3].sum()) < int(stone[:3].sum())
    assert glass[2] > glass[0]  # glass stays blue
    # The shader reads a wall of more than 62 storeys as of unknown height, which is never glass.
    assert (lod.far_wall_colour(glass_hex, 70) == lod.far_wall_colour(glass_hex, 0)).all()


def test_flat_surfaces_cover_the_tile_without_overlapping(tmp_path):
    scene = Scene()
    scene.ways.append(Way("w1", LineString([(0, 120), (256, 130)]), "primary", "road", 12.0, "Main Street"))
    scene.ways.append(Way("w2", LineString([(40, 0), (40, 256)]), "footway", "path", 6.0, None, cover="paving_stones"))
    scene.areas["green"].append(Area("w3", box(120, 150, 240, 250), {"leisure": "park", "name": "Test Park"}, "grass"))
    scene.areas["green"].append(Area("w4", box(150, 170, 200, 220), {"leisure": "pitch"}, "pitch"))
    scene.areas["water"].append(Area("w5", box(60, 150, 110, 250), {"natural": "water"}, "pond"))
    scene.buildings.append(building("w6", box(150, 20, 190, 70), 25.0))
    ctx = tiles.prepare(scene, [(0, 0)])
    layers = lod.build_cell(ctx, 0, 0)
    assert [mesh.name for mesh, _ in layers] == ["lod_land", "lod_water"]

    flat = []
    for mesh, _ in layers:
        on_ground = np.isclose(mesh.positions[mesh.indices.reshape(-1, 3)][:, :, 1], 0.0).all(axis=1)
        triangles = mesh.indices.reshape(-1, 3)[on_ground]
        flat += [Polygon(mesh.positions[t][:, [0, 2]]) for t in triangles]
    areas = [p.area for p in flat]
    assert sum(areas) == pytest.approx(SIZE * SIZE, rel=1e-3)  # the whole tile is covered ...
    assert unary_union(flat).area == pytest.approx(sum(areas), rel=1e-3)  # ... exactly once, by triangles that do not overlap


# A ground polygon from the coverage simplification of tile (-12, -8) whose rings touch and double back (a vertex repeated a
# hair apart, sliver holes); the ear clipper covers 22,668 m2 of it, which is 16,295.
BAD_POLYGON = (
    "POLYGON ((-3015.102766966024 -1999.5737106412935, -3030.7407241695905 -1991.189962524046, "
    "-3051.4775460930723 -1976.5714531980998, -3061.1288548166704 -1968.3433169800658, "
    "-3072.0 -1956.7389941167864, -3072.0 -1947.8321233502013, -3072.0 -1944.445101258438, "
    "-3072.0 -1880.8969569220794, -3072.0 -1878.548674200686, -3072.0 -1878.1050911520522, "
    "-3067.293991103852 -1878.5723261444684, -3066.943729027751 -1879.0109422649152, "
    "-3025.6055283247792 -1930.7767555109122, -3022.87056979619 -1928.5927221213105, "
    "-3054.784211894454 -1888.6288241101531, -3063.4932346708433 -1877.7229395062961, "
    "-3064.573033454309 -1876.3707601366668, -3065.2467752376197 -1875.5691669108407, "
    "-2977.3423137450577 -1842.5790369636784, -2976.4306950647506 -1842.236910771049, "
    "-2975.2793108961528 -1841.8117453552968, -2974.476820056969 -1841.5154138956636, "
    "-2974.2862322018573 -1842.0304053411558, -2973.928275587353 -1842.9976474260168, "
    "-2940.898140927037 -1932.1342438913389, -2939.6406222385917 -1936.4026004421753, "
    "-2952.736990217351 -1940.8725208916464, -2960.0654316460123 -1931.2405105401733, "
    "-2967.353523605937 -1928.8561318148415, -2981.6005932038797 -1934.2659056029072, "
    "-2980.358156434874 -1937.537961665236, -2966.5557484323454 -1932.297030924289, "
    "-2962.6262527221475 -1933.6550089468387, -2956.206202724902 -1942.093090718032, "
    "-2988.0261536373673 -1953.7200177589882, -2986.8027208486656 -1956.9992268422168, "
    "-2972.0018440092117 -1951.4772007782092, -2960.3622752228816 -1980.5893028197384, "
    "-2957.1124039021256 -1979.2899428239984, -2968.712358065068 -1950.276922198591, "
    "-2941.498016143733 -1940.7304439223092, -2941.166683649525 -1940.6184336616823, "
    "-2940.1679133656075 -1940.2807893866304, -2940.163103366721 -1940.2791555355084, "
    "-2939.082978429525 -1939.910511339265, -2938.5171756290297 -1939.717404124129, "
    "-2923.089835411379 -1979.9051547863771, -2922.9981645767643 -1980.1439545141677, "
    "-2920.1902228206122 -1987.4661030043292, -2941.3456391353643 -1997.0136377141362, "
    "-2955.3540699967157 -2000.8630803451629, -2982.503309976195 -2003.7240472850908, "
    "-2993.7924142775646 -2010.0094365508583, -3010.1428896862717 -2002.3078415960754, "
    "-3014.1273586008097 -2000.1114042281433, -3014.1273586008097 -2000.1114042281436, "
    "-3015.102766966024 -1999.5737106412935), (-3005.3663184880056 -1853.6640590588943, "
    "-2976.7969377597 -1842.9933793989303, -2977.1235518542217 -1843.1153699456702, "
    "-3005.3663184880056 -1853.6640590588943), (-3011.6911009206397 -1993.3251928219793, "
    "-3037.0611048884475 -1981.5811401607457, -3012.7407514520464 -1992.8392980963665, "
    "-3011.6911009206397 -1993.3251928219793), (-3000.1347882372906 -1996.2289229249334, "
    "-3011.4953154386135 -1993.3942364406291, -3010.6570492904357 -1993.6034011808194, "
    "-3000.1347882372906 -1996.2289229249334), (-2976.9864580483118 -1998.2532090131426, "
    "-2961.9376352071413 -1996.838450790383, -2942.331927779014 -1991.892179743154, "
    "-2925.2917842731695 -1984.6141910792794, -2926.4248919085367 -1988.9848700170405, "
    "-2936.4077250261907 -1993.486046200851, -2940.6944863718236 -1993.1285465917317, "
    "-2953.540551934042 -1996.8322989438893, -2978.297557875514 -1999.8255619518459, "
    "-2993.2802556047973 -2006.5494801744353, -2998.711077436077 -2006.4861191251548, "
    "-3012.0218687774322 -1998.9731165394187, -3010.6519776892965 -1994.1237070418429, "
    "-2997.148484381556 -1997.0483005194692, -2976.9864580483118 -1998.2532090131426))"
)


def test_triangles_cover_a_polygon_exactly_even_where_ear_clipping_does_not():
    from shapely import wkt

    polygon = wkt.loads(BAD_POLYGON)
    assert polygon.is_valid and polygon.area == pytest.approx(16294.8, abs=0.1)
    vertices, triangles = lod.covering_triangles(polygon)
    assert lod._area_of(vertices, triangles) == pytest.approx(polygon.area, rel=1e-6)
    shapes = [Polygon(vertices[t]) for t in triangles]
    assert all(shape.exterior.is_ccw for shape in shapes)
    assert unary_union(shapes).symmetric_difference(polygon).area < 1e-3  # no gap, and (with the area) no overlap
    # A plain polygon takes the ear clipper's answer unchanged.
    plain = box(0, 0, 10, 10)
    assert len(lod.covering_triangles(plain)[1]) == 2


def test_a_polygon_the_ear_clipper_gets_wrong_is_triangulated_again(monkeypatch):
    monkeypatch.setattr(lod, "triangulate", lambda polygon: (np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]), np.array([[0, 1, 2]])))
    polygon = box(0, 0, 10, 10)
    vertices, triangles = lod.covering_triangles(polygon)
    assert lod._area_of(vertices, triangles) == pytest.approx(100.0)
    builder = MeshBuilder("lod_land", colours=True)
    lod.add_flat_checked(builder, polygon, 3.0, np.zeros(2), np.array([1, 2, 3, 255], dtype=np.uint8))
    mesh = builder.finish()
    assert (mesh.positions[:, 1] == 3.0).all() and len(mesh.indices) // 3 >= 2


def test_trees_are_crowns_of_at_most_eight_triangles_and_small_ones_are_left_out():
    big = Point("n1", 50.0, 60.0, tags={"natural": "tree", "genus": "Quercus", "height": "20", "diameter_crown": "14"})
    small = Point("n2", 80.0, 60.0, tags={"natural": "tree", "height": "5", "diameter_crown": "3"})
    fir = Point("n3", 110.0, 60.0, tags={"natural": "tree", "leaf_type": "needleleaved", "height": "18", "diameter_crown": "7"})
    assert describe(big.tags).crown_width >= lod.TREE_MIN_CROWN > describe(small.tags).crown_width
    for tree, most in ((big, 8), (fir, 6)):
        builder = MeshBuilder("lod_land", colours=True)
        assert lod.add_trees(builder, [tree], np.zeros(2)) == 1
        mesh = builder.finish()
        assert 4 <= len(mesh.indices) // 3 <= most
        shape = describe(tree.tags)
        assert mesh.positions[:, 1].max() == pytest.approx(shape.height, abs=0.01)  # glTF y is up
        spread = np.ptp(mesh.positions[:, [0, 2]], axis=0)
        assert spread.max() <= shape.crown_width * lod.CROWN_RADIUS + 0.01
    builder = MeshBuilder("lod_land", colours=True)
    assert lod.add_trees(builder, [small], np.zeros(2)) == 0 and builder.empty
    # Counter-clockwise from outside: every face's normal points away from the crown's middle.
    builder = MeshBuilder("lod_land", colours=True)
    lod.add_trees(builder, [big], np.zeros(2))
    mesh = builder.finish()
    corners = mesh.positions[mesh.indices.reshape(-1, 3)]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    middle = corners.mean(axis=1) - mesh.positions.mean(axis=0)
    assert (np.einsum("ij,ij->i", normal, middle) > 0).all()

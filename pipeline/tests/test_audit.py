import json
import zlib

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402
from shapely.geometry import LineString, box  # noqa: E402

from boston_pipeline import audit, trees  # noqa: E402
from boston_pipeline.features import Area, Object, Point, Scene, Way  # noqa: E402

PIXELS = 400
BOUNDS = (0.0, 0.0, 100.0, 100.0)


def make_scene() -> Scene:
    scene = Scene()
    scene.areas["green"].append(Area("w1", box(10, 10, 150, 60), {"leisure": "park", "name": "Test Garden"}))
    scene.areas["bed"].append(Area("w2", box(60, 20, 70, 30), {"leisure": "garden"}))
    scene.ways.append(Way("w3", LineString([(0, 80), (100, 80)]), "footway", "path", 4.0, "Lagoon Path", {"surface": "asphalt"}))
    scene.trees.append(Point("bprd:1", 30.0, 50.0, tags={"natural": "tree", "species": "Tilia cordata", "circumference": "1.2"}))
    scene.trees.append(Point("n2", 50.0, 70.0, tags={"natural": "tree"}))
    scene.trees.append(Point("n3", 300.0, 70.0, tags={"natural": "tree"}))  # far outside the cell
    scene.objects.append(Object("n4", 40.0, 40.0, "bench"))
    return scene


def photo():
    return Image.new("RGB", (PIXELS, PIXELS), (128, 128, 128))


def count(image, colour, near):
    """Pixels of an exact colour within 3 px of a point."""
    wanted = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
    x, y = near
    return sum(
        image.getpixel((px, py)) == wanted
        for px in range(max(0, int(x) - 3), min(PIXELS, int(x) + 4))
        for py in range(max(0, int(y) - 3), min(PIXELS, int(y) + 4))
    )


def test_pixel_mapping_inverts_and_north_is_up():
    for x, y in [(0, 0), (12.5, 99.0), (100, 100), (33.3, 71.1)]:
        col, row = audit.to_pixel(x, y, BOUNDS, PIXELS)
        assert audit.to_world(col, row, BOUNDS, PIXELS) == pytest.approx((x, y))
    assert audit.to_pixel(0, 100, BOUNDS, PIXELS) == (0, 0)  # north-west corner is the top left
    assert audit.to_pixel(100, 0, BOUNDS, PIXELS) == (PIXELS, PIXELS)
    assert audit.to_pixel(50, 75, BOUNDS, PIXELS)[1] < audit.to_pixel(50, 25, BOUNDS, PIXELS)[1]


def test_cells_for_park_are_aligned_and_complete():
    geometry = audit.park_geometry(make_scene(), "test garden")  # case-insensitive
    cells = audit.cells_for(geometry, 100.0)
    assert sorted(cells) == [(0, 0), (1, 0)]
    assert audit.cell_name(-5, -4) == "c_-5_-4"
    assert audit.cell_bounds(-1, 2, 120.0) == (-120.0, 240.0, 0.0, 360.0)


def test_cells_for_square_straddling_the_origin():
    cells = audit.cells_for(box(-30, -30, 30, 30), 100.0)
    assert sorted(cells) == [(-1, -1), (-1, 0), (0, -1), (0, 0)]


def test_unknown_park_lists_names(capsys):
    with pytest.raises(SystemExit) as error:
        audit.park_geometry(make_scene(), "Nowhere")
    assert "Test Garden" in str(error.value)


def test_render_cell_colours_trees_and_numbers_north_to_south():
    scene = make_scene()
    overlay, features = audit.render_cell(scene, [], BOUNDS, PIXELS, photo())
    assert overlay.size == (PIXELS, PIXELS)
    first, second = features["trees"]
    assert (first["id"], second["id"]) == ("n2", "bprd:1")  # n2 is further north
    assert [t["n"] for t in features["trees"]] == [1, 2]

    mpp = 100 / PIXELS
    inventory = scene.trees[0]
    other = scene.trees[1]
    r_inv = trees.describe(inventory.tags, zlib.crc32(b"bprd:1")).crown_width / 2 / mpp
    r_other = trees.describe(other.tags, zlib.crc32(b"n2")).crown_width / 2 / mpp
    cx, cy = audit.to_pixel(inventory.x, inventory.y, BOUNDS, PIXELS)
    assert count(overlay, audit.YELLOW, (cx + r_inv, cy)) > 0
    assert count(overlay, audit.ORANGE, (cx + r_inv, cy)) == 0
    ox, oy = audit.to_pixel(other.x, other.y, BOUNDS, PIXELS)
    assert count(overlay, audit.ORANGE, (ox + r_other, oy)) > 0


def test_features_content():
    scene = make_scene()
    _, features = audit.render_cell(scene, [], BOUNDS, PIXELS, photo())
    by_id = {t["id"]: t for t in features["trees"]}
    assert "n3" not in by_id  # outside the cell
    linden = by_id["bprd:1"]
    shape = trees.describe(scene.trees[0].tags, zlib.crc32(b"bprd:1"))
    assert linden["crown_width"] == round(shape.crown_width, 2)
    assert linden["trunk_cm"] == round(1.2 / 3.14159265 * 100)
    assert by_id["n2"]["trunk_cm"] is None
    assert linden["px"] == [round(v) for v in audit.to_pixel(30.0, 50.0, BOUNDS, PIXELS)]
    assert linden["pos"] == [30.0, 50.0]
    assert features["paths"] == [{"id": "w3", "name": "Lagoon Path", "kind": "footway", "surface": "asphalt", "width": 4.0}]
    assert features["beds"] == [{"id": "w2", "area": 100.0}]
    assert features["objects"] == {"bench": 1}
    assert features["metres_per_pixel"] == 0.25
    json.dumps(features)  # must be serialisable


def test_deck_is_listed_and_labelled():
    way = Way("w9", LineString([(10, 90), (40, 90)]), "footway", "path", 3.0, "Bridge", {}, bridge=True)
    _, features = audit.render_cell(Scene(), [(way, way.line)], BOUNDS, PIXELS, photo())
    assert features["bridges"] == [{"label": "B1", "id": "w9", "name": "Bridge", "width": 3.0, "length": 30.0}]


def test_photo_cache_is_reused(tmp_path, monkeypatch):
    calls = []

    def fake(url):
        calls.append(url)
        import io
        buffer = io.BytesIO()
        photo().save(buffer, "JPEG")
        return buffer.getvalue()

    monkeypatch.setattr(audit, "download", fake)
    cache = tmp_path / "aerial" / "c_0_0.jpg"
    audit.fetch_photo(BOUNDS, PIXELS, cache)
    audit.fetch_photo(BOUNDS, PIXELS, cache)
    assert len(calls) == 1
    audit.fetch_photo(BOUNDS, PIXELS, cache, force=True)
    assert len(calls) == 2
    assert "bboxSR=26986" in calls[0]


def test_photo_url_uses_crs_metres():
    from boston_pipeline import world
    url = audit.photo_url(BOUNDS, PIXELS)
    assert f"bbox={world.ORIGIN_E:.3f},{world.ORIGIN_N:.3f},{world.ORIGIN_E + 100:.3f},{world.ORIGIN_N + 100:.3f}" in url


# --- canopy ------------------------------------------------------------------------------------

LAWN, LEAF = (120, 170, 60), (30, 110, 40)  # NDVI_Color style: lawn is lighter green, dense canopy dark green


def ndvi(discs=(), rects=()):
    """A synthetic NDVI picture on the test bounds: discs are (x, y, radius) and rects (x0, y0, x1, y1), in metres."""
    from PIL import ImageDraw
    image = Image.new("RGB", (PIXELS, PIXELS), LAWN)
    draw = ImageDraw.Draw(image)
    for x, y, r in discs:
        (cx, cy), k = audit.to_pixel(x, y, BOUNDS, PIXELS), r / 0.25
        draw.ellipse([cx - k, cy - k, cx + k, cy + k], fill=LEAF)
    for x0, y0, x1, y1 in rects:
        (a, b), (c, d) = audit.to_pixel(x0, y1, BOUNDS, PIXELS), audit.to_pixel(x1, y0, BOUNDS, PIXELS)
        draw.rectangle([a, b, c, d], fill=LEAF)
    return image


def tree_scene(*spots) -> Scene:
    scene = Scene()
    for n, (x, y) in enumerate(spots):
        scene.trees.append(Point(f"n{n}", x, y, tags={"natural": "tree"}))
    return scene


def canopy_of(scene, picture):
    return audit.render_cell(scene, [], BOUNDS, PIXELS, photo(), ndvi=picture)[1]


def test_blobs_join_diagonally_and_keep_separate_ones_apart():
    import numpy as np
    mask = np.zeros((10, 10), bool)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = True
    mask[7:9, 7:9] = True
    labels, blobs = audit.label_blobs(mask)
    assert sorted(b.area_px for b in blobs) == [3, 4]
    assert labels[1, 1] == labels[3, 3] != labels[7, 7]
    square = next(b for b in blobs if b.area_px == 4)
    assert (square.col, square.row, square.fill, square.aspect) == (8.0, 8.0, 1.0, 1.0)


def test_mask_follows_the_rule_and_ignores_speckle():
    mask = audit.canopy_mask(ndvi(discs=[(50, 50, 3)]))
    assert mask[200, 200] and not mask[10, 10]
    speckled = ndvi()
    speckled.putpixel((100, 100), LEAF)  # one dark pixel is smoothed away
    assert not audit.canopy_mask(speckled).any()


def test_a_crown_with_no_tree_is_proposed_at_its_own_size():
    features = canopy_of(Scene(), ndvi(discs=[(50, 50, 3)]))
    [proposal] = features["proposed"]
    assert proposal["label"] == "C1"
    assert proposal["pos"] == pytest.approx([50, 50], abs=0.5)
    assert proposal["crown_m"] == pytest.approx(6.0, rel=0.1)
    assert proposal["area_m2"] == pytest.approx(28.3, rel=0.15)
    assert proposal["px"] == pytest.approx([200, 200], abs=2)
    assert 0 < features["canopy_share"] < 0.05
    json.dumps(features)


def test_proposals_are_numbered_north_to_south_then_west_to_east():
    features = canopy_of(Scene(), ndvi(discs=[(20, 20, 2), (70, 80, 2), (30, 80, 2)]))
    assert [p["pos"] for p in features["proposed"]] == [pytest.approx(v, abs=0.5) for v in ([30, 80], [70, 80], [20, 20])]
    assert [p["label"] for p in features["proposed"]] == ["C1", "C2", "C3"]


def test_a_tree_in_the_crown_is_measured_not_proposed():
    features = canopy_of(tree_scene((50, 50)), ndvi(discs=[(50, 50, 3)]))
    assert features["proposed"] == []
    [tree] = features["trees"]
    assert tree["seen_crown"] == pytest.approx(6.0, rel=0.1)
    assert tree["seen_px"] == pytest.approx([200, 200], abs=2)


def test_a_trunk_just_outside_the_crown_still_owns_it():
    features = canopy_of(tree_scene((54.0, 50)), ndvi(discs=[(50, 50, 3)]))  # 1 m beyond the edge
    assert features["proposed"] == [] and "seen_crown" in features["trees"][0]


def test_two_trees_in_one_blob_are_not_measured():
    features = canopy_of(tree_scene((48, 50), (52, 50)), ndvi(discs=[(50, 50, 5)]))
    assert features["proposed"] == []
    assert all("seen_crown" not in t for t in features["trees"])


def test_a_nearby_modelled_tree_blocks_a_proposal():
    # A small crown 6 m from a tree whose modelled crown is wider than that.
    scene = tree_scene((56, 50))
    scene.trees[0].tags["diameter_crown"] = "14"
    assert canopy_of(scene, ndvi(discs=[(50, 50, 2)]))["proposed"] == []


def test_long_thin_blobs_are_not_trees():
    features = canopy_of(Scene(), ndvi(rects=[(20, 40, 40, 43)]))  # a hedge-like strip
    assert features["proposed"] == []


def test_blobs_touching_the_picture_edge_are_ignored():
    scene = tree_scene((1.0, 50))
    features = canopy_of(scene, ndvi(discs=[(1.0, 50, 3)]))
    assert features["proposed"] == [] and "seen_crown" not in features["trees"][0]


def test_a_crown_on_a_building_or_a_road_is_not_proposed():
    from boston_pipeline.features import Building
    scene = Scene()
    scene.buildings.append(Building("b1", box(40, 40, 60, 60), 10, 0, "default", "#aaaaaa", "#555555", {}))
    scene.ways.append(Way("r1", LineString([(0, 20), (100, 20)]), "residential", "road", 8.0, "Road St", {}))
    picture = ndvi(discs=[(50, 50, 2.5), (30, 20, 2.5), (80, 80, 2.5)])
    features = canopy_of(scene, picture)
    assert [p["pos"] for p in features["proposed"]] == [pytest.approx([80, 80], abs=0.5)]


def test_proposals_are_drawn_in_white_and_measured_crowns_thinly():
    scene = tree_scene((20, 50))
    picture = ndvi(discs=[(70, 50, 3), (20, 50, 4)])
    overlay, features = audit.render_cell(scene, [], BOUNDS, PIXELS, photo(), ndvi=picture)
    [proposal] = features["proposed"]
    cx, cy = proposal["px"]
    r = proposal["crown_m"] / 2 / 0.25
    assert count(overlay, audit.WHITE, (cx + r, cy)) > 4
    assert count(overlay, audit.WHITE, (cx - r, cy)) > 4
    assert count(overlay, audit.WHITE, (cx, cy)) == 0  # a ring, not a disc; the mask itself is not drawn
    seen = features["trees"][0]
    assert count(overlay, audit.WHITE, (seen["seen_px"][0] + seen["seen_crown"] / 2 / 0.25, seen["seen_px"][1])) > 0


def test_without_a_canopy_picture_the_output_is_as_before():
    overlay, features = audit.render_cell(make_scene(), [], BOUNDS, PIXELS, photo())
    assert "proposed" not in features and "canopy_share" not in features
    assert all("seen_crown" not in t for t in features["trees"])


def test_canopy_picture_outlines_the_mask():
    mask = audit.canopy_mask(ndvi(discs=[(50, 50, 3)]))
    outline = audit.canopy_picture(photo(), mask)
    assert outline.getpixel((200, 200)) == (128, 128, 128)  # inside: untouched
    assert sum(outline.getpixel((x, 200)) == (255, 61, 242) for x in range(180, 221)) == 2


def test_ndvi_url_adds_the_rendering_rule_and_uses_its_own_cache(tmp_path, monkeypatch):
    url = audit.photo_url(BOUNDS, PIXELS, audit.NDVI)
    assert "renderingRule=%7B%22rasterFunction%22%3A%22NDVI_Color%22%7D" in url and "bboxSR=26986" in url
    assert audit.NDVI.name != audit.SOURCE.name

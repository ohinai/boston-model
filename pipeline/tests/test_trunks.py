"""What the viewer needs to make tree trunks solid: each tree's trunk radius, in the tile's data file.

The viewer (web/src/trunks.ts) builds a vertical cylinder at every tree that lists a `trunk`. These tests check that the
number is the radius of the trunk as it is drawn, at 1.3 m, and that it reaches the tile's `.json` for the right trees."""

import json
import math
import zlib

import numpy as np
import pytest

from boston_pipeline.features import Object, Point, Scene
from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.tiles import build
from boston_pipeline.trees import BREAST_HEIGHT, add_tree, describe, trunk_radius, _assemble

CM = 100 / math.pi


def girth(d_cm):
    return str(d_cm / CM)


def seed_of(ident):
    return zlib.crc32(ident.encode())


def drawn_radius(parts, z=BREAST_HEIGHT):
    """The trunk's radius at height z measured another way: slice its mesh with the plane and find how far the cut is
    from the cut's centre. A slice crosses the vertical edges at the ring corners, which are the furthest points."""
    (trunk,) = [p for p in parts if p.kind == "trunk"]
    points = []
    for face in trunk.faces:
        corners = trunk.positions[face]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = corners[a][2], corners[b][2]
            if (za - z) * (zb - z) < 0:
                t = (z - za) / (zb - za)
                points.append((corners[a] + t * (corners[b] - corners[a]))[:2])
    points = np.array(points)
    return float(np.hypot(*(points - points.mean(axis=0)).T).max())


def parts_of(tags, ident="n1"):
    shape = describe(tags, seed_of(ident))
    return shape, _assemble(shape, seed_of(ident))


@pytest.mark.parametrize("tags", [
    {"species": "Quercus rubra", "circumference": "2.0"},  # a big oak: round, forks high
    {"species": "Ulmus americana", "circumference": "3.2"},  # a vase-shaped elm
    {"species": "Salix babylonica", "circumference": "2.2"},  # a willow, drawn with a skirt
    {"species": "Pinus strobus", "circumference": "1.2"},  # a pine: a bare trunk with pads
    {"species": "Picea abies", "circumference": "1.0"},  # a spruce: tiers to the ground
    {"species": "Acer rubrum", "circumference": "0.4"},  # a young tree
    {"natural": "tree"},  # nothing known
])
def test_radius_is_the_drawn_trunk_at_breast_height(tags):
    shape, parts = parts_of(tags)
    radius = trunk_radius(parts)
    assert 0 < radius < shape.trunk_radius  # thinner than the flared foot, and a trunk
    assert radius == pytest.approx(drawn_radius(parts), rel=0.03)


def test_a_thicker_trunk_gives_a_larger_radius():
    radii = [trunk_radius(parts_of({"species": "Quercus rubra", "circumference": girth(d)})[1]) for d in (20, 40, 80, 140)]
    assert radii == sorted(radii) and len(set(radii)) == 4
    # About the girth the tag gives: a 80 cm trunk is 0.4 m across the middle, and the drawn trunk is a little thinner.
    assert 0.25 < radii[2] < 0.42


def test_a_trunk_that_ends_below_breast_height_gives_its_top_radius():
    shape = describe({"species": "Thuja occidentalis", "circumference": "0.6"}, 0)
    assert shape.form == "spire"  # its trunk stops at about the foot of the foliage
    parts = _assemble(shape, 0)
    (trunk,) = [p for p in parts if p.kind == "trunk"]
    assert trunk.positions[:, 2].max() < BREAST_HEIGHT
    assert trunk_radius(parts) == pytest.approx(0.5 * shape.trunk_radius, rel=0.03)  # the top ring of the profile


def test_a_clipped_evergreen_draws_no_trunk_so_has_no_radius():
    shape, parts = parts_of({"species": "Taxus baccata"})
    assert shape.form == "clipped" and not any(p.kind == "trunk" for p in parts)
    assert trunk_radius(parts) == 0.0


def test_the_radius_is_deterministic_and_does_not_change_the_drawing():
    shape = describe({"species": "Acer platanoides", "circumference": "1.4"}, 7)
    builder = MeshBuilder("trees", colours=True, features=True)
    again = MeshBuilder("trees", colours=True, features=True)
    first = add_tree(builder, shape, 10.0, 20.0, 7, 3)
    second = add_tree(again, shape, 10.0, 20.0, 7, 3)
    assert first == second == trunk_radius(_assemble(shape, 7))
    one, two = builder.finish(), again.finish()
    assert np.array_equal(one.positions, two.positions) and np.array_equal(one.indices, two.indices)


# --- the tile's data file ----------------------------------------------------------------------------


def objects_of(out_dir, trees):
    scene = Scene()
    for ident, x, y, tags in trees:
        scene.trees.append(Point(ident, x, y, tags=tags))
    scene.objects.append(Object("n99", 50.0, 50.0, "bench"))
    build(scene, [(0, 0)], out_dir, {"name": "Test", "places": []}, "2026-01-01T00:00:00Z")
    return {o["id"]: o for o in json.loads((out_dir / "tiles" / "0_0.json").read_text(encoding="utf-8"))["objects"]}


def test_each_tree_with_a_trunk_lists_its_radius(tmp_path):
    trees = [
        ("n1", 100.0, 100.0, {"natural": "tree", "species": "Quercus rubra", "circumference": "2.0"}),
        ("n2", 140.0, 90.0, {"natural": "tree", "species": "Salix babylonica", "circumference": "2.2"}),
        ("n3", 60.0, 30.0, {"natural": "tree", "species": "Pinus strobus", "circumference": "1.2"}),
    ]
    objects = objects_of(tmp_path, trees)
    for ident, x, y, tags in trees:
        entry = objects[ident]
        assert entry["kind"] == "tree" and entry["pos"] == pytest.approx([x, y])
        expected = trunk_radius(_assemble(describe(tags, seed_of(ident)), seed_of(ident)))
        assert entry["trunk"] == pytest.approx(expected, abs=0.006)  # written to the centimetre
    assert objects["n1"]["trunk"] > objects["n3"]["trunk"]  # the thicker oak has the thicker trunk


def test_a_tree_without_a_drawn_trunk_and_other_objects_list_none(tmp_path):
    objects = objects_of(tmp_path, [("n1", 100.0, 100.0, {"natural": "tree", "species": "Taxus baccata"})])
    assert objects["n1"]["kind"] == "tree" and "trunk" not in objects["n1"]
    assert objects["n99"]["kind"] == "bench" and "trunk" not in objects["n99"]


def test_the_data_file_is_the_same_every_time(tmp_path):
    trees = [("n1", 100.0, 100.0, {"natural": "tree", "species": "Quercus rubra", "circumference": "2.0"})]
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    assert objects_of(tmp_path / "a", trees) == objects_of(tmp_path / "b", trees)

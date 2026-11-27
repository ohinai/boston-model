"""Trees that stand inside a building's outline are dropped (inventory.drop_trees_in_buildings)."""

from shapely.geometry import box

from boston_pipeline.features import Building, Point, Scene
from boston_pipeline.inventory import drop_trees_in_buildings


def building(polygon, height=12.0, min_height=0.0, **tags):
    return Building("w1", polygon, height, min_height, "levels", "#aa5533", "#555555", tags)


def scene_with(buildings, trees):
    scene = Scene()
    scene.buildings = buildings
    scene.trees = trees
    return scene


def test_a_tree_inside_a_building_is_dropped_and_one_outside_stays():
    scene = scene_with([building(box(0, 0, 10, 10))], [Point("bprd:1", 5, 5), Point("bprd:2", 15, 5)])
    assert drop_trees_in_buildings(scene) == 1
    assert [t.id for t in scene.trees] == ["bprd:2"]


def test_a_tree_a_fix_put_inside_a_building_stays():
    scene = scene_with([building(box(0, 0, 10, 10))], [Point("fix:a", 5, 5)])
    assert drop_trees_in_buildings(scene) == 0
    assert len(scene.trees) == 1


def test_a_low_shed_an_open_roof_and_a_raised_part_do_not_count():
    scene = scene_with(
        [building(box(0, 0, 10, 10), height=2.0), building(box(20, 0, 30, 10), building="roof"), building(box(40, 0, 50, 10), min_height=6.0)],
        [Point("n1", 5, 5), Point("n2", 25, 5), Point("n3", 45, 5)],
    )
    assert drop_trees_in_buildings(scene) == 0
    assert len(scene.trees) == 3


def test_nothing_to_do_with_no_buildings_or_no_trees():
    assert drop_trees_in_buildings(scene_with([], [Point("n1", 1, 1)])) == 0
    assert drop_trees_in_buildings(scene_with([building(box(0, 0, 10, 10))], [])) == 0

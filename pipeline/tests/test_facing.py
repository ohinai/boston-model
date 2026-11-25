"""Benches that state no direction are turned to face a path, a plaza or the water."""

import numpy as np
import pytest
from shapely.geometry import LineString, box
from test_parks import _mesh_data, _node, _read_glb

from boston_pipeline.facing import bench_facings
from boston_pipeline.features import BENCH_REACH, BENCH_WATERSIDE, Object, Scene, Way, classify
from boston_pipeline.furniture import BENCH_SEAT_TOP
from boston_pipeline.tiles import build


def _path(ident, *points, kind="footway", surface="path"):
    return Way(ident, LineString(points), kind, surface, 2.0, None)


def _bench(ident, x, y, direction=None):
    return Object(ident, x, y, "bench", direction=direction)


EAST_WEST = _path("w1", (0, 10), (100, 10))


def test_a_bench_looks_at_the_nearest_path():
    benches = [_bench("south", 50, 7), _bench("north", 50, 13), _bench("far", 50, 10 + BENCH_REACH + 1)]
    facings = bench_facings(benches, [EAST_WEST], [], [])
    assert facings == {"south": 0.0, "north": 180.0}  # the far one has nothing within reach to look at


def test_the_nearer_of_two_paths_wins_and_roads_do_not_count():
    ways = [EAST_WEST, _path("w2", (60, 0), (60, 100)), _path("w3", (50, 21), (150, 21), kind="residential", surface="road")]
    # 3 m from the path to the east, 10 m from the one to the south, 1 m from the road.
    assert bench_facings([_bench("b", 57, 20)], ways, [], []) == {"b": 90.0}


def test_a_stated_direction_is_left_alone():
    benches = [_bench("north", 50, 13, direction=0.0), _bench("east", 50, 13, direction=90.0), _bench("lamp", 50, 13)]
    benches[2].kind = "lamp"
    assert bench_facings(benches, [EAST_WEST], [], []) == {}


def test_classify_keeps_an_unstated_direction_unstated():
    scene = classify([_node(1, {"amenity": "bench"}), _node(2, {"amenity": "bench", "direction": "N"})])
    assert [thing.direction for thing in scene.objects] == [None, 0.0]


def test_a_bench_between_a_path_and_water_looks_at_the_water():
    lake = box(0, -50, 100, 0)  # south of the path, its shore along y = 0
    waterside, landward = _bench("waterside", 50, 6), _bench("landward", 50, 14)
    facings = bench_facings([waterside, landward], [EAST_WEST], [], [lake])
    # The first has the path behind it and the shore 6 m ahead. The second looks at
    # the path, and across it at the water, which is too far to matter anyway.
    assert facings == {"waterside": 180.0, "landward": 180.0}
    # Water farther off than BENCH_WATERSIDE does not turn a bench round.
    distant = box(0, -50, 100, 6 - BENCH_WATERSIDE - 1)
    assert bench_facings([waterside], [EAST_WEST], [], [distant]) == {"waterside": 0.0}


def test_a_bench_in_a_paved_area_looks_into_it():
    plaza = box(0, 20, 40, 60)
    # 1 m inside the west edge; a path runs just outside that edge, behind the bench.
    behind = _path("w4", (-1, 0), (-1, 100))
    assert bench_facings([_bench("b", 1, 40)], [behind], [plaza], []) == {"b": 90.0}
    # 2 m inside the north edge.
    assert bench_facings([_bench("b", 20, 58)], [behind], [plaza], []) == {"b": 180.0}


def test_a_bench_mapped_on_a_path_looks_across_it():
    (facing,) = bench_facings([_bench("b", 50, 10)], [EAST_WEST], [], []).values()
    assert facing in (0.0, 180.0)


@pytest.mark.parametrize(("path_north", "back_is_north"), [(203.0, False), (197.0, True)])
def test_build_turns_a_bench_to_the_path(tmp_path, path_north, back_is_north):
    scene = Scene()
    scene.ways.append(_path("w1", (150, path_north), (250, path_north)))
    scene.objects.append(_bench("n1", 200.0, 200.0))
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)

    glb = tmp_path / "tiles" / "0_0.glb"
    document = _read_glb(glb)
    (furniture,) = [mesh["primitives"][0] for mesh in document["meshes"] if mesh["name"] == "furniture"]
    positions, _ = _mesh_data(glb, document, furniture)
    # The backrest is all that stands above the seat. glTF y is up and z is -north.
    back = positions[positions[:, 1] > BENCH_SEAT_TOP + 0.01]
    north = -back[:, 2].mean()
    assert (north > 200.0) == back_is_north and abs(north - 200.0) > 0.1
    assert np.ptp(back[:, 0]) == pytest.approx(1.8, abs=0.01)  # the bench runs along the path, east to west

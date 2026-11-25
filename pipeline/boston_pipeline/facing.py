"""Which way a bench faces when the data does not say.

Most benches in the data state no `direction`. Left alone they would all face
north. Each is turned to look at what a sitter there would look at:

1. inside a paved area (a plaza, a wharf): away from its nearest edge, because
   benches stand round the edge of such places and look in;
2. otherwise at the nearest path;
3. unless water is close by on the other side: a bench standing between a path
   and the water looks at the water.

Checked against the 368 benches in the pilot area that do state a direction
(data of November 2026): 312 come out within 45 degrees of it, 24 sideways and 30
facing the wrong way. Away from the Charles River Esplanade, where benches look
at the river from farther back, it is 164 of 176. "The nearest path" alone gets
290 of the 368.

A stated `direction` always wins, so a wrong guess is corrected with a file in
fixes/ that sets `direction` on the bench.
"""

from __future__ import annotations

import math

from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from .features import BENCH_REACH, BENCH_WATERSIDE

ON_IT = 0.05  # closer than this to a line or an edge counts as standing on it


def _bearing(a, b) -> float:
    """Degrees clockwise from north of the way from point `a` to point `b`."""
    return math.degrees(math.atan2(b.x - a.x, b.y - a.y)) % 360


def _apart(a: float, b: float) -> float:
    """The angle between two directions, 0 to 180 degrees."""
    return abs((a - b + 180) % 360 - 180)


def _into_paved(point, paved, tree) -> float | None:
    """Looking away from the nearest edge of the paved area the point is inside."""
    homes = [paved[i] for i in tree.query(point, predicate="within")]
    if not homes:
        return None
    edge = nearest_points(min(homes, key=lambda polygon: polygon.area).boundary, point)[0]
    return _bearing(edge, point) if edge.distance(point) > ON_IT else None


def _toward_path(point, lines, tree) -> float | None:
    """Looking at the nearest path within reach."""
    near = sorted(tree.query(point, predicate="dwithin", distance=BENCH_REACH))
    if not near:
        return None
    line = min((lines[i] for i in near), key=lambda line: line.distance(point))
    along = line.project(point)
    foot = line.interpolate(along)
    if foot.distance(point) > ON_IT:
        return _bearing(point, foot)
    # Mapped on the path itself: look square across it.
    a, b = line.interpolate(max(0.0, along - 0.5)), line.interpolate(min(line.length, along + 0.5))
    return (_bearing(a, b) + 90) % 360


def _toward_water(point, water, tree) -> float | None:
    """Looking at the nearest shore, if there is water close by."""
    best = None
    for i in sorted(tree.query(point, predicate="dwithin", distance=BENCH_WATERSIDE)):
        shore = nearest_points(water[i], point)[0]
        distance = shore.distance(point)
        if distance > 0 and (best is None or distance < best[0]):
            best = (distance, _bearing(point, shore))
    return best[1] if best else None


def bench_facings(objects, ways, paved, water) -> dict[str, float]:
    """The direction (degrees clockwise from north) for each bench that states
    none, by id. A bench with nothing within reach to look at is left out.

    `ways` are the scene's ways (only paths count), `paved` the polygons of paths
    drawn as areas, `water` the polygons of all water, the sea included.
    """
    benches = [thing for thing in objects if thing.kind == "bench" and thing.direction is None]
    if not benches:
        return {}
    lines = [way.line for way in ways if way.surface == "path"]
    path_tree, paved_tree, water_tree = STRtree(lines), STRtree(paved), STRtree(water)
    facings = {}
    for bench in benches:
        point = Point(bench.x, bench.y)
        facing = _into_paved(point, paved, paved_tree)
        if facing is None:
            facing = _toward_path(point, lines, path_tree)
            if facing is None:
                continue
            waterside = _toward_water(point, water, water_tree)
            if waterside is not None and _apart(waterside, facing) > 90:
                facing = waterside
        facings[bench.id] = round(facing, 1)
    return facings

"""Pitched roofs over building walls, in z-up tile metres.

A building whose `roof:shape` is one of the shapes in features.ROOF_SHAPES (gabled, hipped, pyramidal, skillion,
gambrel, mansard, round, dome ...) is drawn with walls that stop at the eaves and a roof that rises from there, instead
of a box with a flat top. In OpenStreetMap `height` is the height to the top of the roof and `roof:height` the roof's
own height (else `roof:levels` of about 2.5 m each, else a default pitch), so the eaves are `height - roof height`. A
building known only by its `building:levels` has walls of levels x 3.4 m, and the roof stands on top of them.

Roofs are built over the footprint's smallest enclosing rectangle (the "frame"): the roof surface is the lowest of a
few planes (two for a gable, four for a hip, a plane and a flat top for a mansard, eight chords for a barrel), which
covers every ridge shape with one rule. Each plane's share of the frame is cut to the footprint, so a roof never
reaches beyond the walls it covers and an L-shaped or notched footprint still gets a roof; a footprint that fills less
than MIN_FILL of its frame, or is narrower than MIN_SPAN, keeps its flat slab. Where the roof is higher than the eaves
along a wall (a gable end, the high side of a skillion roof) the wall is filled up to the roof, blank, in the wall's
colour: its UVs are zero, which the facade shader reads as a wall too short for windows. A pyramid, cone or dome is a fan
or a set of rings round a point in the footprint. Nothing here reaches past the footprint, and anything odd (a hole in
the footprint, a footprint the apex cannot see all of, a rise too small to show) gives None, which is the old flat slab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polylabel, unary_union

from .features import ROOF_ALLOWANCE, ROOF_LEVEL_HEIGHT
from .mesh import MeshBuilder, colour_bytes, srgb_to_linear, triangulate

MIN_FILL = 0.80  # a ridge roof's footprint must fill this share of its enclosing rectangle
MIN_SPAN = 1.0  # metres: a ridge roof over a footprint narrower than this stays flat
MIN_AREA = 2.0  # square metres
MIN_RISE = 0.25  # metres: a roof lower than this is not worth drawing
# A pitch given as a rise over the half-span, when `roof:height` or `roof:levels` says nothing: (degrees, metre cap).
DEFAULT_PITCH = {"gable": (30.0, 4.5), "hip": (30.0, 4.5), "half_hip": (30.0, 4.5), "gambrel": (40.0, 4.5), "pyramid": (35.0, 8.0)}
MANSARD_RISE = 3.0  # about one storey of attic
MANSARD_PITCH = math.tan(math.radians(70.0))  # the steep lower slope of a mansard
SKILLION_PITCH, SKILLION_CAP = math.tan(math.radians(8.0)), 3.0  # a shed roof's slope, and how high it may rise
BARREL_SHARE, BARREL_CAP = 0.30, 6.0  # a barrel vault rises this share of its span, up to this
DOME_SHARE, DOME_CAP = 0.45, 12.0
# A tagged rise is trusted, but not past this many times the half-span (63 degrees) for a ridge or shed roof, nor this many
# times the footprint's long side for a spire or dome.
MAX_RISE_PER_HALF_SPAN = 2.0
MAX_SPIRE_RATIO = 8.0
DOME_ANGLES = (90.0, 65.0, 40.0, 18.0, 0.0)  # degrees from the top: the dome's rings, down to the eaves
BARREL_SEGMENTS = 8
GAMBREL_KNEE = (0.5, 0.75)  # (share of the half-span in from the wall, share of the rise) where a gambrel's slope changes
MANSARD_INSET_MAX = 0.35  # the flat top of a mansard keeps at least this much of the span
PARTY_GAP = 0.5  # metres: another building this close to a side of a mansard roof's frame stands against it
PARTY_SHARE = 0.6  # ... if it is that close along this share of the side
EDGE = 1e-3  # metres: a wall's fill shorter or lower than this is skipped


@dataclass
class RoofPlan:
    """One building's pitched roof, in world metres. Make it with plan_roof, draw it with add_roof."""

    shape: str
    eave: float  # where the walls stop
    top: float  # the highest point of the roof
    kind: str  # "planes": the lowest of `planes` over `bound`; "fan": triangles to `apex`; "dome": rings round `apex`
    planes: np.ndarray | None = None  # (k, 3): z = gx * x + gy * y + c
    bound: np.ndarray | None = None  # (n, 2) convex polygon the lowest plane is taken over
    apex: np.ndarray | None = None  # (2,)
    surface: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))  # the roof: triangles (m, 3, 3), facing up
    fill: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))  # the wall up to it: triangles facing out


def _frame(polygon: Polygon):
    """(centre, u, v, long, short) of the footprint's smallest enclosing rectangle: u runs along its long side, v is u turned a quarter."""
    mrr = polygon.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return None
    corners = np.asarray(mrr.exterior.coords)[:4]
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    lengths = [float(np.hypot(*e)) for e in edges]
    first = 0 if lengths[0] >= lengths[1] else 1
    if lengths[1 - first] < 1e-6:
        return None
    u = edges[first] / lengths[first]
    return corners.mean(axis=0), u, np.array([-u[1], u[0]]), lengths[first], lengths[1 - first]


def _across(shape: str, roof, u: np.ndarray, v: np.ndarray) -> bool:
    """Whether the ridge runs across the frame's long side. `roof:direction` (the way the roof slopes down) puts the ridge
    square to it; else `roof:orientation=across`; else it runs along the long side."""
    if shape not in ("gable", "gambrel", "barrel"):
        return False
    if roof.direction is not None:
        d = np.array([math.sin(math.radians(roof.direction)), math.cos(math.radians(roof.direction))])
        return abs(d @ u) > abs(d @ v)
    return roof.orientation == "across"


def _rise(shape: str, roof, half: float, extent: float, long_side: float):
    """(rise, tagged): the roof's own height in metres, and whether the data gave it (`roof:height`, else `roof:levels`)."""
    tagged = roof.height if roof.height else (roof.levels * ROOF_LEVEL_HEIGHT if roof.levels else None)
    if tagged is not None:
        if shape in ("pyramid", "dome"):
            return min(tagged, MAX_SPIRE_RATIO * long_side), True
        return min(tagged, (extent if shape == "skillion" else MAX_RISE_PER_HALF_SPAN * half)), True
    if shape == "mansard":
        return MANSARD_RISE, False
    if shape == "barrel":
        return min(BARREL_SHARE * 2 * half, BARREL_CAP), False
    if shape == "dome":
        return min(DOME_SHARE * 2 * half, DOME_CAP), False
    if shape == "skillion":
        return min(max(SKILLION_PITCH * extent, 0.4), SKILLION_CAP), False
    degrees, cap = DEFAULT_PITCH[shape]
    return min(math.tan(math.radians(degrees)) * half, cap), False


def _plane(gx: float, gy: float, z0: float, centre: np.ndarray, r: np.ndarray, s: np.ndarray) -> list[float]:
    """The plane z = z0 + gx * x + gy * y in the frame (x along r, y along s, from the centre), as (gx, gy, c) in world x, y."""
    g = gx * r + gy * s
    return [float(g[0]), float(g[1]), float(z0 - g @ centre)]


def _ridge_planes(shape: str, a: float, b: float, eave: float, rise: float, centre, r, s) -> np.ndarray:
    """The planes whose lowest is the roof. The frame has half-length a along the ridge and half-span b across it."""
    e, rr = eave, rise

    def plane(gx, gy, z0):
        return _plane(gx, gy, z0, centre, r, s)

    if shape == "gable":
        planes = [plane(0, -rr / b, e + rr), plane(0, rr / b, e + rr)]
    elif shape == "hip":
        k = rr / b
        planes = [plane(0, -k, e + k * b), plane(0, k, e + k * b), plane(-k, 0, e + k * a), plane(k, 0, e + k * a)]
    elif shape == "half_hip":
        # A gable whose ends are hipped over the upper half: the end wall stops at half the rise, and a hip as steep as the
        # sides climbs from it to the ridge, which is b / 2 short of each end.
        k = rr / b
        planes = [
            plane(0, -k, e + rr), plane(0, k, e + rr),
            plane(k, 0, e + rr / 2 + k * a), plane(-k, 0, e + rr / 2 + k * a),
        ]
    elif shape == "gambrel":
        knee_y, knee_z = GAMBREL_KNEE[0] * b, GAMBREL_KNEE[1] * rr
        steep, shallow = knee_z / (b - knee_y), (rr - knee_z) / knee_y
        planes = [
            plane(0, -steep, e + steep * b), plane(0, steep, e + steep * b),
            plane(0, -shallow, e + rr), plane(0, shallow, e + rr),
        ]
    elif shape == "mansard":
        inset = min(rr / MANSARD_PITCH, MANSARD_INSET_MAX * 2 * min(a, b))
        k = rr / inset
        planes = [
            plane(0, -k, e + k * b), plane(0, k, e + k * b), plane(-k, 0, e + k * a), plane(k, 0, e + k * a),
            plane(0, 0, e + rr),
        ]
    else:  # barrel: an elliptical arch across the span, as the chords of eight equal steps of angle
        angles = np.radians(np.linspace(-90.0, 90.0, BARREL_SEGMENTS + 1))
        ys, zs = b * np.sin(angles), rr * np.cos(angles)
        planes = []
        for y0, y1, z0, z1 in zip(ys[:-1], ys[1:], zs[:-1], zs[1:]):
            slope = (z1 - z0) / (y1 - y0)
            planes.append(plane(0, slope, e + z0 - slope * y0))
    return np.array(planes)


def _star_apex(polygon: Polygon):
    """A point from which every edge of the footprint (no holes) is seen with the inside on its left, else None: a fan of
    triangles from it then covers the footprint exactly."""
    if len(polygon.interiors):
        return None
    ring = np.asarray(orient(polygon, 1.0).exterior.coords)[:-1]
    edges = np.roll(ring, -1, axis=0) - ring
    for candidate in (polygon.centroid, polygon.representative_point(), polylabel(polygon, tolerance=0.1)):
        apex = np.array([candidate.x, candidate.y])
        if polygon.contains(candidate) and (edges[:, 0] * (apex[1] - ring[:, 1]) - edges[:, 1] * (apex[0] - ring[:, 0]) > 1e-9).all():
            return apex
    return None


def plan_roof(building, neighbours=()) -> RoofPlan | None:
    """The pitched roof of a building, or None to keep its flat slab (no roof tags, or a footprint the roof does not fit).

    `neighbours` are the footprints of the other buildings close by: a mansard roof does not slope down toward a
    neighbour it stands against (a party wall), so a row of them is one unbroken roof.
    """
    roof = building.roof
    polygon = building.polygon
    if roof is None or polygon.is_empty or polygon.area < MIN_AREA or not polygon.is_valid:
        return None
    try:
        plan = _plan(building, roof, polygon, neighbours)
        if plan is None:
            return None
        if plan.kind == "planes":
            plan.surface, plan.fill = _facets(plan, polygon), _fills(plan, polygon)
        else:
            plan.surface = (_fan if plan.kind == "fan" else _dome)(plan, polygon)
        return plan if _sound(plan, polygon) else None
    except (GEOSException, ValueError):  # odd geometry: draw nothing rather than something wrong
        return None


def _party_walls(centre, ridge, span_axis, a: float, b: float, neighbours) -> list[bool]:
    """For the frame's four sides, in the order of a mansard's planes (+y, -y, +x, -x): whether a neighbouring building
    stands against it. Seven points along the middle of each side must mostly lie within PARTY_GAP of a neighbour."""
    if not len(neighbours):
        return [False] * 4
    union = unary_union(list(neighbours))
    along, across = np.linspace(-0.8 * a, 0.8 * a, 7), np.linspace(-0.8 * b, 0.8 * b, 7)
    sides = [(along, np.full(7, b)), (along, np.full(7, -b)), (np.full(7, a), across), (np.full(7, -a), across)]
    result = []
    for x, y in sides:
        points = centre + x[:, None] * ridge + y[:, None] * span_axis
        result.append(bool((shapely.distance(union, shapely.points(points)) <= PARTY_GAP).mean() >= PARTY_SHARE))
    return result


def _plan(building, roof, polygon: Polygon, neighbours=()) -> RoofPlan | None:
    frame = _frame(polygon)
    if frame is None:
        return None
    centre, u, v, long_side, short_side = frame
    shape = roof.shape
    fan = shape in ("pyramid", "dome")
    # A ridge roof is built over the enclosing rectangle; a shed roof is one plane over any footprint; a fan needs no frame.
    if shape not in ("pyramid", "dome", "skillion") and (short_side < MIN_SPAN or polygon.area < MIN_FILL * long_side * short_side):
        return None
    across = _across(shape, roof, u, v)
    ridge, span_axis = (v, -u) if across else (u, v)
    along, span = (short_side, long_side) if across else (long_side, short_side)
    half = span / 2

    slope = None
    if shape == "skillion":
        slope = _skillion_direction(roof, v)
        corners = np.asarray(polygon.exterior.coords)[:, :2]
        extent = float(np.ptp(corners @ slope))
        if extent < MIN_SPAN:
            return None
    else:
        extent = span
    rise, tagged = _rise(shape, roof, half, extent, long_side)

    base = building.min_height
    if building.height_source == "levels":
        eave = max(base, building.height - ROOF_ALLOWANCE)  # the walls are the levels; the roof stands on them
    else:
        room = building.height - base
        rise = min(rise, room if tagged else room / 2)
        eave = building.height - rise
    if rise < MIN_RISE:
        return None
    top = eave + rise

    if fan:
        apex = _star_apex(polygon)
        return None if apex is None else RoofPlan(shape, eave, top, "dome" if shape == "dome" else "fan", apex=apex)
    if shape == "skillion":
        # z = eave + rise * (high - slope . p) / (high - low): the eaves at the downhill end, the full rise at the uphill end.
        low, high = float((corners @ slope).min()), float((corners @ slope).max())
        gradient = -slope * (rise / (high - low))
        planes = np.array([[gradient[0], gradient[1], eave + rise * high / (high - low)]])
        return RoofPlan(shape, eave, top, "planes", planes, np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:4])
    a = along / 2
    planes = _ridge_planes(shape, a, half, eave, rise, centre, ridge, span_axis)
    if shape == "mansard":  # the four slopes (+y, -y, +x, -x) and the flat top: leave out the slope toward a party wall
        gone = _party_walls(centre, ridge, span_axis, a, half, neighbours)
        planes = planes[[i for i in range(5) if i == 4 or not gone[i]]]
    bound = np.array(
        [centre - a * ridge - half * span_axis, centre + a * ridge - half * span_axis,
         centre + a * ridge + half * span_axis, centre - a * ridge + half * span_axis]
    )
    return RoofPlan(shape, eave, top, "planes", planes, bound)


def _skillion_direction(roof, v: np.ndarray) -> np.ndarray:
    """The unit vector a skillion roof slopes down toward: `roof:direction`, else across the footprint's short side, toward the south."""
    if roof.direction is not None:
        return np.array([math.sin(math.radians(roof.direction)), math.cos(math.radians(roof.direction))])
    return v if v[1] <= 0 else -v


def _clip(poly: np.ndarray, g: np.ndarray, c: float) -> np.ndarray:
    """The part of a convex polygon where g . p + c >= 0 (Sutherland-Hodgman against one line), without repeated points."""
    d = poly @ g + c
    out = []
    n = len(poly)
    for i in range(n):
        p, q, dp, dq = poly[i], poly[(i + 1) % n], d[i], d[(i + 1) % n]
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            out.append(p + (q - p) * (dp / (dp - dq)))
    kept = [point for k, point in enumerate(out) if not k or np.hypot(*(point - out[k - 1])) > 1e-7]
    if len(kept) > 1 and np.hypot(*(kept[0] - kept[-1])) <= 1e-7:
        kept.pop()
    return np.array(kept) if kept else poly[:0]


def _emit(builder: MeshBuilder, tris: np.ndarray, colour, feature: int, up: bool = False) -> None:
    """Triangles (m, 3, 3) in the winding given, each with its own normal. With `up`, any that would face down are turned over."""
    if not len(tris):
        return
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normals, axis=1)
    keep = length > 1e-9
    tris, normals, length = tris[keep], normals[keep], length[keep]
    if up:
        flip = normals[:, 2] < 0
        tris[flip] = tris[flip][:, ::-1]
        normals[flip] *= -1
    if not len(tris):
        return
    builder.add(
        tris.reshape(-1, 3), np.repeat(normals / length[:, None], 3, axis=0), np.arange(len(tris) * 3), colour=colour, feature=feature
    )


def _polygons(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area > 1e-6 else []
    return [g for part in getattr(geometry, "geoms", []) for g in _polygons(part)]


def _lift(piece: Polygon, plane: np.ndarray) -> np.ndarray:
    """A polygon on a plane as triangles (m, 3, 3), counter-clockwise from above."""
    vertices, triangles = triangulate(piece)
    if not len(triangles):
        return np.zeros((0, 3, 3))
    return np.column_stack([vertices, vertices @ plane[:2] + plane[2]])[triangles]


def _facets(plan: RoofPlan, polygon: Polygon) -> np.ndarray:
    """Every facet of a roof made of planes, cut to the footprint: triangles (m, 3, 3)."""
    planes, bound = plan.planes, plan.bound
    gradients, heights = planes[:, :2], planes[:, 2]
    out = []
    for i in range(len(planes)):
        region = bound
        for j in range(len(planes)):
            if j != i and len(region):
                region = _clip(region, gradients[j] - gradients[i], heights[j] - heights[i])
        if len(region) < 3:
            continue
        facet = Polygon(region)
        if not facet.is_valid:  # a sliver the clipping left folded over itself
            facet = shapely.make_valid(facet)
        for part in _polygons(facet):
            for piece in _polygons(part.intersection(polygon)):
                out.append(_lift(piece, planes[i]))
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def _fills(plan: RoofPlan, polygon: Polygon) -> np.ndarray:
    """The wall between the eaves and the roof along every edge of the footprint: triangles (m, 3, 3), facing out.

    Along an edge the roof's height is the lowest of the planes, which is straight between the places where the lowest
    plane changes, so each stretch is a trapezoid (or a triangle) from the eaves up to the roof.
    """
    planes = plan.planes
    oriented = orient(polygon, 1.0)
    out = []
    for ring in [oriented.exterior, *oriented.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        for a, b in zip(coords[:-1], coords[1:]):
            if np.hypot(*(b - a)) < EDGE:
                continue
            za, zb = planes[:, :2] @ a + planes[:, 2], planes[:, :2] @ b + planes[:, 2]
            da, db = za[:, None] - za[None, :], zb[:, None] - zb[None, :]
            crossing = np.triu(da * db < 0, 1)
            ts = np.concatenate([[0.0, 1.0], (da / np.where(crossing, da - db, 1.0))[crossing]])
            ts = np.unique(np.round(ts, 9))
            points = a + ts[:, None] * (b - a)
            zs = (points @ planes[:, :2].T + planes[:, 2]).min(axis=1)
            for k in range(len(ts) - 1):
                p, q, zp, zq = points[k], points[k + 1], zs[k], zs[k + 1]
                high_p, high_q = zp - plan.eave > EDGE, zq - plan.eave > EDGE
                if not (high_p or high_q) or np.hypot(*(q - p)) < EDGE:
                    continue
                p0, q0 = np.array([*p, plan.eave]), np.array([*q, plan.eave])
                p1, q1 = np.array([*p, zp]), np.array([*q, zq])
                if high_p and high_q:
                    out += [[p0, q0, q1], [p0, q1, p1]]
                elif high_q:
                    out.append([p0, q0, q1])
                else:
                    out.append([p0, q0, p1])
    return np.array(out) if out else np.zeros((0, 3, 3))


def _fan(plan: RoofPlan, polygon: Polygon) -> np.ndarray:
    """A pyramid (or cone): a triangle from each edge of the footprint, at the eaves, up to the apex."""
    ring = np.asarray(orient(polygon, 1.0).exterior.coords)[:-1]
    base = np.column_stack([ring, np.full(len(ring), plan.eave)])
    apex = np.array([*plan.apex, plan.top])
    return np.stack([base, np.roll(base, -1, axis=0), np.broadcast_to(apex, base.shape)], axis=1)


def _dome(plan: RoofPlan, polygon: Polygon) -> np.ndarray:
    """A dome: rings of the footprint drawn in toward its centre as the height climbs a sphere, with a fan on top."""
    ring = np.asarray(orient(polygon, 1.0).exterior.coords)[:-1]
    rise = plan.top - plan.eave
    levels = []
    for angle in DOME_ANGLES[:-1]:
        scale, height = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        levels.append(np.column_stack([plan.apex + scale * (ring - plan.apex), np.full(len(ring), plan.eave + rise * height)]))
    apex = np.array([*plan.apex, plan.top])
    out = []
    for low, high in zip(levels[:-1], levels[1:]):
        low_next, high_next = np.roll(low, -1, axis=0), np.roll(high, -1, axis=0)
        out.append(np.stack([low, low_next, high_next], axis=1))
        out.append(np.stack([low, high_next, high], axis=1))
    last = levels[-1]
    out.append(np.stack([last, np.roll(last, -1, axis=0), np.broadcast_to(apex, last.shape)], axis=1))
    return np.concatenate(out)


def _sound(plan: RoofPlan, polygon: Polygon) -> bool:
    """Whether a roof is fit to draw: seen from above its triangles cover the footprint once and no more, none lies outside
    it, and every height is between the eaves and the top. A roof that fails is dropped for the flat slab, which has no holes."""
    surface = plan.surface
    if not len(surface):
        return False
    a, b, c = surface[:, 0, :2], surface[:, 1, :2], surface[:, 2, :2]
    area = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum() / 2
    if abs(area - polygon.area) > 1e-3 * polygon.area + 1e-6:
        return False
    for tris in (surface, plan.fill):
        if len(tris) and (tris[:, :, 2].min() < plan.eave - 1e-6 or tris[:, :, 2].max() > plan.top + 1e-6):
            return False
    xy = surface[:, :, :2].reshape(-1, 2)
    return bool(shapely.contains_xy(polygon.buffer(1e-3), xy[:, 0], xy[:, 1]).all())


def _is_brick(hex_colour: str) -> bool:
    """Whether a wall colour is reddish, which the facade shader draws as brick (web/src/facade.ts)."""
    red, green, _ = srgb_to_linear(hex_colour)
    return red / max(green, 0.002) > 1.6


def add_roof(walls: MeshBuilder, roofs: MeshBuilder, building, plan: RoofPlan, origin, feature: int) -> None:
    """The roof's surface into `roofs`, and the wall between the eaves and the roof (gable ends, a skillion's high side) into `walls`.

    The caller draws the walls up to `plan.eave` and, for a building on a podium, the underside; this draws nothing else.
    `origin` is the tile's corner: the builders take positions relative to it.
    """
    shift = np.array([*np.asarray(origin, dtype=float), 0.0])
    _emit(roofs, plan.surface - shift, colour_bytes(building.roof_colour), feature, up=True)
    fill = plan.fill - shift
    normals = np.cross(fill[:, 1] - fill[:, 0], fill[:, 2] - fill[:, 0])
    length = np.linalg.norm(normals, axis=1) if len(fill) else np.zeros(0)
    fill, normals, length = fill[length > 1e-9], normals[length > 1e-9], length[length > 1e-9]
    if len(fill):
        # Blank (u = v = 0): the facade shader leaves a wall with no bays unpatterned. Brick is a touch darker than a flat fill of its colour.
        shade = 0.9 if _is_brick(building.wall_colour) else 1.0
        walls.add(
            fill.reshape(-1, 3), np.repeat(normals / length[:, None], 3, axis=0), np.arange(len(fill) * 3),
            uvs=np.zeros((len(fill) * 3, 2)), colour=colour_bytes(building.wall_colour, shade), feature=feature,
        )

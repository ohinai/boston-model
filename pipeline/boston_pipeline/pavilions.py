"""Open pavilions and painted docks, in z-up tile metres.

A `building=roof` with a `roof:shape` of hipped, gabled or pyramidal, and no `min_height`, is a
roof of that shape on posts, open on every side (a park pavilion, a boat landing). Any other
`building=roof` keeps the flat slab it has always had. A pier area with a `colour` tag is a
painted dock: its sides, from the water up to the deck, take that colour.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .bridges import _block, _emit, _emit_triangles
from .features import parse_colour, parse_length
from .mesh import MeshBuilder, colour_bytes

# Roof shapes drawn as a shaped roof on posts: `roof:shape` value -> the shape.
SHAPES = {"hipped": "hip", "hip": "hip", "pyramidal": "hip", "gabled": "gable"}
POST = 0.14  # square post side
POST_SPACING = 3.4  # posts stand about this far apart along each side
POST_INSET = 0.12  # the post's centre from the edge of the footprint
OVERHANG = 0.45  # the roof reaches this far beyond the footprint
ROOF_THICK = 0.1  # thickness of the roof's shell
FASCIA = 0.2  # depth of the board round the eaves
BEAM = (0.12, 0.2)  # (wide, tall) of the beam that joins the post tops
RISE_SHARE, RISE_LIMITS = 0.12, (0.4, 1.2)  # how far a roof rises: a share of its width, within these limits
MIN_EAVE = 2.0  # a walker can stand under the eaves
POST_COLOUR = "#7a5a3a"  # bare timber, when the data names no colour
DOCK_LIP = 0.05  # a painted dock's side reaches this far above the deck
DOCK_OUT = 0.02  # and stands this far off the water's edge, clear of the kerb drawn there
DOCK_REACH = 0.7  # water's edge this close to the dock's outline is its side


def roof_shape(tags: dict) -> str | None:
    """"hip" or "gable" when these tags describe an open roof on posts of that shape, else None."""
    if tags.get("building") != "roof" or tags.get("min_height") or tags.get("building:min_level"):
        return None
    return SHAPES.get(str(tags.get("roof:shape", "")).strip().lower())


def painted_pier(tags: dict) -> bool:
    """Whether a pier area is painted: it has a `colour` this module knows."""
    return tags.get("man_made") == "pier" and parse_colour(tags.get("colour")) is not None


def add_pavilion(walls: MeshBuilder, roofs: MeshBuilder, building, origin, feature: int) -> None:
    """Posts, a beam and a hipped or gabled roof over the footprint of `building`.

    The footprint is replaced by its smallest enclosing rectangle. The roof's top is the building's
    height; its eaves are lower by `roof:height`, or by a share of the roof's width.
    """
    shape = roof_shape(building.tags)
    corners = np.asarray(building.polygon.minimum_rotated_rectangle.exterior.coords)[:4, :2] - origin
    edge = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
    length = [float(np.hypot(*e)) for e in edge]
    first = 0 if length[0] >= length[1] else 1
    long_side, short_side = max(length[0], length[1]), min(length[0], length[1])
    if short_side < 0.5:
        return
    u = edge[first] / length[first]  # along the long side
    v = np.array([-u[1], u[0]])  # to its left
    centre = corners.mean(axis=0)
    top = building.height
    width = short_side + 2 * OVERHANG
    rise = parse_length(building.tags.get("roof:height"))
    if rise is None or rise <= 0:
        rise = min(RISE_LIMITS[1], max(RISE_LIMITS[0], RISE_SHARE * width))
    eave = max(top - rise, MIN_EAVE)
    rise = max(top - eave, 0.25)

    posts = colour_bytes(parse_colour(building.tags.get("colour") or building.tags.get("building:colour")) or POST_COLOUR)
    roof = colour_bytes(building.roof_colour)
    up = np.array([0.0, 0.0, 1.0])
    axes = np.array([[*u, 0.0], [*v, 0.0], up])

    def at(x: float, y: float, z: float) -> np.ndarray:
        return np.array([*(centre + x * u + y * v), z])

    # Posts round the footprint's edge, with a beam joining their tops.
    a, b = long_side / 2 - POST_INSET, short_side / 2 - POST_INSET
    ring = [(-a, -b), (a, -b), (a, b), (-a, b)]
    height = eave - ROOF_THICK
    for i in range(4):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % 4]
        span = math.hypot(x1 - x0, y1 - y0)
        count = max(1, math.ceil(span / POST_SPACING - 1e-9))
        for k in range(count):
            x, y = x0 + (x1 - x0) * k / count, y0 + (y1 - y0) * k / count
            _block(walls, posts, at(x, y, height / 2), axes, np.array([POST, POST, height]) / 2, feature, skip=((2, -1),))
        heading = np.array([x1 - x0, y1 - y0]) / span
        beam_axes = np.array([[*(heading[0] * u + heading[1] * v), 0.0], [*(-heading[1] * u + heading[0] * v), 0.0], up])
        middle = at((x0 + x1) / 2, (y0 + y1) / 2, height - BEAM[1] / 2)
        _block(walls, posts, middle, beam_axes, np.array([span + POST, BEAM[0], BEAM[1]]) / 2, feature)

    # The roof: slopes and ends, then the same underneath, then a board round the eaves.
    a, b = long_side / 2 + OVERHANG, short_side / 2 + OVERHANG
    ridge = a if shape == "gable" else max(a - b, 0.0)
    ze, zt = eave, eave + rise
    A, B, C, D = at(-a, -b, ze), at(a, -b, ze), at(a, b, ze), at(-a, b, ze)
    r1, r2 = at(-ridge, 0, zt), at(ridge, 0, zt)
    faces = [(A, B, r2), (A, r2, r1), (C, D, r1), (C, r1, r2), (B, C, r2), (D, A, r1)]
    pivot = at(0, 0, ze - 1.0)
    tris = []
    for tri in faces:
        tri = np.array(tri)
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(normal) < 1e-9:
            continue  # a slope that came to a point
        tris.append(tri if normal @ (tri.mean(axis=0) - pivot) > 0 else tri[::-1])
    tris = np.array(tris)
    _emit_triangles(roofs, tris, roof, feature)
    _emit_triangles(roofs, tris[:, ::-1] - np.array([0.0, 0.0, ROOF_THICK]), roof, feature)
    fascia = [np.array([p, q, q - np.array([0, 0, FASCIA]), p - np.array([0, 0, FASCIA])]) for p, q in ((A, B), (B, C), (C, D), (D, A))]
    _emit(roofs, np.array(fascia), colour_bytes(building.roof_colour, 0.8), feature)


def add_dock_edge(
    builder: MeshBuilder, water: list[Polygon], dock: Polygon, colour: str, z0: float, origin, tile_box, feature: int
) -> None:
    """The painted side of a dock: where the edge of the water lies along `dock` (within DOCK_REACH of it,
    which takes in the pier ways that widen it), a band from `z0` up to the deck, facing the water.
    `water` is the water of the tile, with the paths already cut out of it; edges on the tile's boundary are skipped."""
    near = dock.buffer(DOCK_REACH)
    x0, y0, x1, y1 = tile_box
    corners, normals = [], []
    for polygon in water:
        polygon = orient(polygon, 1.0)
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = np.asarray(ring.coords)[:, :2]
            for a, b in zip(coords[:-1], coords[1:]):
                delta = b - a
                length = float(np.hypot(*delta))
                if length < 1e-3 or not near.contains(ShapelyPoint(*((a + b) / 2))):
                    continue
                if any(abs(a[axis] - value) < 1e-3 and abs(b[axis] - value) < 1e-3 for axis, value in ((0, x0), (0, x1), (1, y0), (1, y1))):
                    continue
                side = np.array([-delta[1], delta[0]]) / length  # into the water: the water lies to the left of its ring
                p, q = a + side * DOCK_OUT - origin, b + side * DOCK_OUT - origin
                corners.append([(*q, z0), (*p, z0), (*p, DOCK_LIP), (*q, DOCK_LIP)])
    if corners:
        _emit(builder, np.array(corners), colour_bytes(colour), feature)

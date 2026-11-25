"""Fences, walls, hedges, chains, kerbs, guard rails and rows of bollards, in z-up tile metres.

Everything is low-poly and flat-shaded, with the ground at z = 0 and counter-clockwise
front faces. Faces nobody can see (the bottom of a post, the ends of a rail that
runs into posts, the underside of a rail) are not drawn, so a long fence stays cheap.

A fence is drawn in one of two ways. Where its tags say it is a metal railing (`material`
iron, steel or metal, up to 1.8 m tall) it is a wrought-iron picket fence: a row of thin
spear-headed bars between a top and a bottom rail, on posts with a cap. Any other fence
is the plain post-and-rail kind. A chain is ball-topped posts with a chain that sags
between them.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString

from .features import polygons_of
from .furniture import _CORNERS, _QUAD, _Parts, add_object
from .mesh import MeshBuilder, add_flat, add_walls, colour_bytes

FENCE_POST, FENCE_POST_SPACING = 0.06, 2.4
FENCE_RAIL_ACROSS, FENCE_RAIL_TALL = 0.04, 0.05
FENCE_LOW_RAIL = 0.15  # centre of the bottom rail above the ground
FENCE_MIDDLE_FROM = 1.4  # a fence this tall or taller gets a middle rail

# Railings, a wrought-iron picket fence: pointed bars between a top and a bottom rail, on capped posts.
# A bar is a three-sided spike (three triangles): the whole pilot has about 5 km of railings, and a
# bar with a separate spearhead would cost three times as much for something nobody sees apart.
IRON_MATERIALS = ("iron", "steel", "metal")  # `material` values that make a fence railings
IRON_MAX_HEIGHT = 1.8  # a taller metal fence is a security or sports fence, not railings
IRON_POST_RADIUS = 0.05  # a square post, this far from its centre to a corner
IRON_POST_CAP = 0.09  # height of the pointed cap on a post
IRON_RAIL_ACROSS, IRON_RAIL_TALL = 0.03, 0.035
IRON_LOW_RAIL = 0.14  # centre of the bottom rail above the ground
IRON_HEAD_SHARE, IRON_HEAD_MAX = 0.14, 0.2  # the top rail is this share of the height below the tips, at most this far
IRON_BAR_RADIUS = 0.019  # circumradius of a bar at its foot; it tapers to a point at the top
IRON_BAR_SPACING = 0.25  # between bar centres; real railings are nearer 0.12 m, which costs too much
IRON_POST_GAP = 0.07  # no bar this close to a post's centre

CHAIN_POST_BOTTOM, CHAIN_POST_TOP, CHAIN_POST_SPACING = 0.055, 0.045, 2.5  # post radii (square, corner to centre)
CHAIN_BALL = 0.065  # radius of the ball on a post
CHAIN_LINK = 0.035  # thickness of the chain
CHAIN_HANG = 0.12  # the chain is hung this far below the post tops
CHAIN_SAG_PER_METRE, CHAIN_SAG_MAX_SHARE = 0.08, 0.45  # how far it sags, and at most this share of its hung height
CHAIN_PIECES = 3  # straight pieces a span of chain is drawn in (2 if the span is short)
RAIL_POST, RAIL_POST_SPACING, RAIL_DEPTH = 0.1, 2.0, 0.25  # guard rail
RAIL_THICK = 0.05
BOLLARD_SPACING = 1.5
MITRE_LIMIT = 2.0  # how far a sharp bend may stretch a solid strip's width

FENCE_COLOUR, GUARD_RAIL_COLOUR = "#23262a", "#8a8d8f"
HEDGE_COLOUR, HEDGE_TOP_SHADE = "#4f6b3a", 1.15
KERB_COLOUR = "#b3aea3"
TOP_SHADE = 1.05  # the top of a stone strip is a touch lighter than its sides
STONE_COLOURS = {
    "stone": "#a9a59b", "granite": "#a9a59b", "dry_stone": "#a9a59b", "concrete": "#b5b4ae", "brick": "#96604c",
}
KERB_COLOURS = {**STONE_COLOURS, "granite": KERB_COLOUR, "stone": KERB_COLOUR}


def _block(parts: _Parts, colour, centre, axes, half, skip=()) -> None:
    """An oriented box with right-handed unit `axes` (3, 3) and half sizes `half`; faces in `skip`
    (axis, sign) are left out."""
    positions, normals = [], []
    for k in range(3):
        for sign in (1, -1):
            if (k, sign) in skip:
                continue
            i, j = (k + 1) % 3, (k + 2) % 3
            if sign < 0:
                i, j = j, i  # keeps tangent_i x tangent_j pointing outward
            face = centre + sign * half[k] * axes[k]
            positions.append(face + _CORNERS[:, :1] * half[i] * axes[i] + _CORNERS[:, 1:] * half[j] * axes[j])
            normals.append(np.repeat((sign * axes[k])[None], 4, axis=0))
    if positions:
        parts.add(np.concatenate(positions), np.concatenate(normals), np.arange(len(positions))[:, None] * 4 + _QUAD, colour)


_UP = np.array([0.0, 0.0, 1.0])
_ENDS = ((0, 1), (0, -1))
_BOTTOM = ((2, -1),)


def _beam(parts, colour, a, b, across: float, tall: float, ends: bool = False) -> None:
    """A square-sectioned bar from 3D point `a` to `b`, `across` wide and `tall` thick. Its ends
    and its underside (nobody looks up at a rail) are left open."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    along = delta / length
    left = np.cross(_UP, along)
    left /= np.linalg.norm(left)
    axes = np.array([along, left, np.cross(along, left)])
    _block(parts, colour, (a + b) / 2, axes, np.array([length, across, tall]) / 2, _BOTTOM if ends else _ENDS + _BOTTOM)


def _post(parts, colour, x, y, height: float, size: float) -> None:
    """A square post standing on the ground, open at the bottom."""
    _block(parts, colour, np.array([x, y, height / 2]), np.eye(3), np.array([size, size, height]) / 2, _BOTTOM)


# --- Small solids, many at a time ----------------------------------------------
def _emit(parts: _Parts, colour, faces: np.ndarray) -> None:
    """Flat faces (m, k, 3), k = 3 or 4, each wound counter-clockwise seen from outside."""
    if not len(faces):
        return
    count, k = faces.shape[:2]
    if k == 4:
        normal = np.cross(faces[:, 2] - faces[:, 0], faces[:, 3] - faces[:, 1])
        indices = np.arange(count)[:, None] * 4 + _QUAD
    else:
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        indices = np.arange(count * 3)
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]
    parts.add(faces.reshape(-1, 3), np.repeat(normal, k, axis=0), indices, colour)


def _ring(xy: np.ndarray, z: float, radius: float, sides: int, rot: float = 0.0) -> np.ndarray:
    """(n, sides, 3): a horizontal ring of `sides` points round each (x, y), counter-clockwise from above."""
    angle = rot + 2 * math.pi * np.arange(sides) / sides
    offset = radius * np.column_stack([np.cos(angle), np.sin(angle)])
    ring = xy[:, None, :] + offset[None]
    return np.concatenate([ring, np.full(ring.shape[:2] + (1,), z)], axis=2)


def _frusta(parts, colour, xy, z0, z1, r0, r1, sides: int, rot: float = 0.0) -> None:
    """Upright prisms (open top and bottom) standing at each of `xy` (n, 2), radius r0 at z0 and r1 at z1."""
    if not len(xy):
        return
    lo, hi = _ring(xy, z0, r0, sides, rot), _ring(xy, z1, r1, sides, rot)
    nxt = np.roll(np.arange(sides), -1)
    _emit(parts, colour, np.stack([lo, lo[:, nxt], hi[:, nxt], hi], axis=2).reshape(-1, 4, 3))


def _cones(parts, colour, xy, z0, z1, radius, sides: int, rot: float = 0.0) -> None:
    """Pointed tops: a ring of `sides` at z0 rising to a point at z1 above each of `xy` (n, 2)."""
    if not len(xy):
        return
    base = _ring(xy, z0, radius, sides, rot)
    apex = np.concatenate([xy, np.full((len(xy), 1), z1)], axis=1)
    nxt = np.roll(np.arange(sides), -1)
    apexes = np.broadcast_to(apex[:, None, :], base.shape)
    _emit(parts, colour, np.stack([base, base[:, nxt], apexes], axis=2).reshape(-1, 3, 3))


def _balls(parts, colour, xy, z: float, radius: float) -> None:
    """A low-poly ball (an octahedron, eight faces) centred at height z above each of `xy` (n, 2)."""
    if not len(xy):
        return
    centre = np.concatenate([xy, np.full((len(xy), 1), z)], axis=1)
    signs = np.array([(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=float)
    corners = np.stack(
        [signs[:, 0, None] * [radius, 0, 0], signs[:, 1, None] * [0, radius, 0], signs[:, 2, None] * [0, 0, radius]], axis=1
    )  # (8, 3, 3), relative to the centre
    faces = centre[:, None, None, :] + corners[None]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.einsum("ij,ij->i", normal, corners.mean(axis=1)) < 0
    faces[:, flip] = faces[:, flip][:, :, ::-1]
    _emit(parts, colour, faces.reshape(-1, 3, 3))


def _stations(line: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the line: every vertex, and evenly spaced ones between, about `spacing` apart."""
    keep = np.r_[True, np.hypot(*np.diff(line, axis=0).T) > 1e-3]
    line = line[keep]
    points = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        count = max(1, round(float(np.hypot(*(b - a))) / spacing))  # a span up to half as long again stays whole
        t = (np.arange(1, count + 1) / count)[:, None]
        points.append(a + t * (b - a))
    points = np.concatenate(points)
    if len(points) > 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]  # a closed outline: one post where it meets itself
    return points


def _spans(line: np.ndarray, posts: np.ndarray):
    """Consecutive pairs of posts; a closed outline also joins its last post to its first."""
    pairs = list(zip(posts[:-1], posts[1:]))
    if len(posts) > 1 and len(line) > 2 and np.allclose(line[0], line[-1]):
        pairs.append((posts[-1], posts[0]))
    return pairs


def _segments(line: np.ndarray):
    keep = np.r_[True, np.hypot(*np.diff(line, axis=0).T) > 1e-3]
    line = line[keep]
    return zip(line[:-1], line[1:])


def _fence(parts, line, height, colour) -> None:
    for x, y in _stations(line, FENCE_POST_SPACING):
        _post(parts, colour, x, y, height, FENCE_POST)
    levels = [height - FENCE_RAIL_TALL / 2, min(FENCE_LOW_RAIL, height / 2)]
    if height >= FENCE_MIDDLE_FROM:
        levels.insert(1, height / 2)
    for a, b in _segments(line):
        for z in levels:
            _beam(parts, colour, (*a, z), (*b, z), FENCE_RAIL_ACROSS, FENCE_RAIL_TALL)


def _is_iron(material: str | None, height: float) -> bool:
    return (material or "") in IRON_MATERIALS and height <= IRON_MAX_HEIGHT


def _iron_fence(parts, line, height, colour) -> None:
    """Railings: pointed bars between a top and a bottom rail, on capped posts."""
    posts = _stations(line, FENCE_POST_SPACING)
    cap_from = height - IRON_POST_CAP
    _frusta(parts, colour, posts, 0.0, cap_from, IRON_POST_RADIUS, IRON_POST_RADIUS, 4)
    _cones(parts, colour, posts, cap_from, height, IRON_POST_RADIUS * 1.1, 4)  # a little wider: it closes the post
    top_rail = height - min(IRON_HEAD_MAX, IRON_HEAD_SHARE * height) - 0.05
    low_rail = min(IRON_LOW_RAIL, top_rail / 2)
    for a, b in _segments(line):
        for z in (top_rail, low_rail):
            _beam(parts, colour, (*a, z), (*b, z), IRON_RAIL_ACROSS, IRON_RAIL_TALL)
    bars = []
    for a, b in _spans(line, posts):
        along = b - a
        length = float(np.hypot(*along))
        room = length - 2 * IRON_POST_GAP
        if room < IRON_BAR_SPACING / 2:
            continue
        count = max(1, round(room / IRON_BAR_SPACING))
        t = (IRON_POST_GAP + (np.arange(count) + 0.5) * room / count) / length
        bars.append(a + t[:, None] * along)
    if bars:
        _cones(parts, colour, np.concatenate(bars), low_rail, height, IRON_BAR_RADIUS, 3, math.pi / 2)


def _chain(parts, line, height, colour) -> None:
    posts = _stations(line, CHAIN_POST_SPACING)
    shaft_top = height - CHAIN_BALL
    _frusta(parts, colour, posts, 0.0, shaft_top, CHAIN_POST_BOTTOM, CHAIN_POST_TOP, 4)
    _balls(parts, colour, posts, shaft_top, CHAIN_BALL)  # centred on the post's top, so it closes it
    hung = height - CHAIN_HANG
    # Between neighbouring posts the chain leaves the post just under the ball and hangs in a
    # curve, a parabola drawn in a few straight pieces.
    for a, b in _spans(line, posts):
        span = float(np.hypot(*(b - a)))
        sag = min(CHAIN_SAG_PER_METRE * span, CHAIN_SAG_MAX_SHARE * hung)
        pieces = CHAIN_PIECES if span > 1.5 else 2
        t = np.arange(pieces + 1) / pieces
        points = [(*(a + s * (b - a)), hung - sag * 4 * s * (1 - s)) for s in t]
        for p, q in zip(points[:-1], points[1:]):
            _beam(parts, colour, p, q, CHAIN_LINK, CHAIN_LINK)


def _guard_rail(parts, line, height, colour) -> None:
    for x, y in _stations(line, RAIL_POST_SPACING):
        _post(parts, colour, x, y, height, RAIL_POST)
    for a, b in _segments(line):
        _beam(parts, colour, (*a, height - RAIL_DEPTH / 2), (*b, height - RAIL_DEPTH / 2), 0.04, RAIL_DEPTH)


def _solid(builder, line, height, width, side, top, feature) -> None:
    """A strip of the given width and height centred on the line, with a top face."""
    zero = np.zeros(2)
    strip = LineString(line).buffer(width / 2, cap_style="flat", join_style="mitre", mitre_limit=MITRE_LIMIT, quad_segs=2)
    for polygon in polygons_of(strip):
        add_walls(builder, polygon, 0.0, height, zero, side, feature=feature)
        add_flat(builder, polygon, height, zero, top, feature=feature)


def _bollards(builder, line, height, colour, feature) -> None:
    string = LineString(line)
    count = max(1, math.ceil(string.length / BOLLARD_SPACING - 1e-9))
    for k in range(count + 1 if string.length > 1e-3 else 1):
        point = string.interpolate(k / count, normalized=True)
        add_object(builder, "bollard", point.x, point.y, height=height, colour=colour, seed=k, feature=feature)


def add_barrier(
    builder: MeshBuilder, kind: str, line, height: float, width: float,
    colour: str | None = None, material: str | None = None, feature: int = 0,
) -> None:
    """Add one barrier along `line` (n, 2), tile-relative metres. `colour` (hex) overrides the
    default for fences, chains, guard rails and bollards; walls, hedges and kerbs take
    theirs from `material`, and a fence's `material` decides whether it is railings."""
    line = np.asarray(line, dtype=float)[:, :2]
    if len(line) < 2:
        return
    if kind == "bollards":
        _bollards(builder, line, height, colour, feature)
        return
    if kind in ("wall", "retaining_wall", "hedge", "kerb"):
        if kind == "hedge":
            side, top = colour_bytes(HEDGE_COLOUR), colour_bytes(HEDGE_COLOUR, HEDGE_TOP_SHADE)
        else:
            palette = KERB_COLOURS if kind == "kerb" else STONE_COLOURS
            hexcode = palette.get((material or "").lower(), palette["stone"])
            side, top = colour_bytes(hexcode), colour_bytes(hexcode, TOP_SHADE)
        _solid(builder, line, height, width, side, top, feature)
        return
    parts = _Parts()
    if kind == "fence":
        draw = _iron_fence if _is_iron((material or "").lower(), height) else _fence
        draw(parts, line, height, colour_bytes(colour or FENCE_COLOUR))
    elif kind == "chain":
        _chain(parts, line, height, colour_bytes(colour or FENCE_COLOUR))
    elif kind == "guard_rail":
        _guard_rail(parts, line, height, colour_bytes(colour or GUARD_RAIL_COLOUR))
    else:
        raise ValueError(f"unknown barrier kind {kind!r}")
    if parts._count:
        parts.flush(builder, feature)

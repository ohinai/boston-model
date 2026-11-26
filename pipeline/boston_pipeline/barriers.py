"""Fences, walls, hedges, chains, kerbs, guard rails, rows of bollards and gates, in z-up tile metres.

Everything is low-poly and flat-shaded, with the ground at z = 0 and counter-clockwise
front faces. Faces nobody can see (the bottom of a post, the ends of a rail that
runs into posts, the underside of a rail) are not drawn, so a long fence stays cheap.

A fence is drawn in one of two ways. Where its tags say it is a metal railing (`material`
iron, steel or metal, up to 1.8 m tall) it is a wrought-iron picket fence: a row of thin
spear-headed bars between a top and a bottom rail, on posts with a cap. Any other fence
is the plain post-and-rail kind. A chain is ball-topped posts with a chain that sags
between them.

A gate (a way across the opening in a fence) is a granite pier at each end of the opening and
iron leaves hinged on the piers and swung back, so the whole mapped opening stays clear
(see `add_gate`).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

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


# --- Gates ------------------------------------------------------------------------------------
# The way is the opening between two fences (or a fence and a wall). It is drawn open: a granite
# pier just outside each end (its inner face on the end of the way, so the mapped opening stays
# clear however narrow it is) and the iron leaves hinged on the piers' faces and swung back beside
# the opening. A narrow opening has one leaf; a wide one has two, each at most GATE_LEAF_MAX long.
GATE_HEIGHT_MIN, GATE_HEIGHT_MAX = 0.9, 3.0  # leaf height: a mapped 5 m "gate" is a gateway, not a taller leaf
GATE_MIN_OPENING = 0.5  # a way shorter than this (or a ring) is not a gate: nothing is drawn
GATE_SINGLE_MAX = 1.3  # an opening up to this wide has one leaf
GATE_LEAF_MAX = 2.6  # longest leaf; a wider opening's leaves stand beside it and leave the middle open
GATE_SWING = math.radians(115)  # how far each leaf has swung from closed: a little past square to the fence
GATE_HINGE_OFFSET = 0.04  # a leaf's plane stands this far into the opening from the pier's inner face (its balls reach 0.09)
GATE_STILE = 0.04  # circumradius of a leaf's upright (a square bar)
GATE_BALL = 0.05  # radius of the ball on top of an upright
GATE_RAIL_ACROSS, GATE_RAIL_TALL = 0.035, 0.04
GATE_LOW_RAIL = 0.16  # centre of the bottom rail above the ground
GATE_BAR_SPACING, GATE_BAR_GAP = 0.16, 0.07  # bars are closer than a fence's, so a gate reads as a gate
GATE_PLANK, GATE_BATTEN = 0.045, 0.06  # thickness of a wooden leaf, and of the battens across it
WOOD_COLOUR = "#7a5a3a"
GATE_WOOD = ("wood", "timber")

PIER_WIDTH_RANGE = (0.3, 0.55)  # across the shaft: a wider opening gets a heavier pier
PIER_BASE, PIER_STEP = 0.28, 0.06  # height of the plinth, and of the step on it
PIER_PLINTH, PIER_STEP_SHARE, PIER_CAP = 1.14, 1.07, 1.2  # widths of the plinth, its step and the cap, as shares of the shaft's
PIER_ABOVE_LEAF = 0.3  # the shaft rises this far above the leaves
PIER_CORNICE, PIER_NECK = 0.12, 0.07  # heights of the projecting cap, and of the block the finial stands on
PIER_URN_FROM = 4.0  # an opening this wide, or a leaf this tall (and at least PIER_FINIAL_FROM wide), gets urns
PIER_URN_TALL = 2.0
PIER_FINIAL_FROM = 1.8  # a narrower opening gets a pyramid cap and no finial
PIER_SIDES = 6  # sides of the round finials
PIER_PANEL, PIER_PANEL_PROUD, PIER_PANEL_MARGIN = 0.62, 0.006, 0.22  # a panel's share of the shaft's width, how far it stands out, its gap to the shaft's ends
PIER_COLOURS = {"brick": STONE_COLOURS["brick"], "concrete": STONE_COLOURS["concrete"]}
PIER_STONE = "#a39b8b"  # warm grey granite: a shade darker than the walls', which read nearly white in the sun
# An urn, as (height, radius) on a unit pier 0.55 m wide: foot, stem, belly, shoulder, neck, lip, lid.
URN_PROFILE = (
    (0.0, 0.12), (0.04, 0.12), (0.07, 0.07), (0.17, 0.06), (0.27, 0.16), (0.36, 0.14),
    (0.44, 0.09), (0.47, 0.14), (0.52, 0.14),
)
URN_APEX = 0.6


@dataclass(frozen=True)
class GatePlan:
    """Where everything of a gate stands: the opening from `a` to `b` (x, y), `u` the unit vector
    along it, `n` its left normal, and which ends (0 is `a`, 1 is `b`) have a leaf."""

    a: np.ndarray
    b: np.ndarray
    u: np.ndarray
    n: np.ndarray
    opening: float
    height: float
    pier_width: float
    leaf_length: float
    hinges: tuple
    pier_top: float  # the top of the piers' shafts

    def end(self, k: int) -> np.ndarray:
        return self.b if k else self.a

    def inward(self, k: int) -> np.ndarray:
        """Unit vector along the opening, away from end `k`."""
        return -self.u if k else self.u

    def pier_centre(self, k: int) -> np.ndarray:
        """The pier stands outside the opening: its widest part, the cap, just touches the end of the way."""
        return self.end(k) - self.inward(k) * self.pier_width * PIER_CAP / 2

    def leaf(self, k: int, side: int) -> tuple:
        """(hinge point, unit direction toward the free end) of the leaf hinged at end `k`, swung
        toward the `side` (+1 left of a to b, -1 right) of the opening."""
        inward = self.inward(k)
        hinge = self.end(k) + inward * GATE_HINGE_OFFSET + side * self.n * (self.pier_width / 2 - GATE_HINGE_OFFSET)
        return hinge, inward * math.cos(GATE_SWING) + side * self.n * math.sin(GATE_SWING)


def gate_plan(line, height: float, fence: float = 0.0) -> GatePlan | None:
    """The layout of a gate along `line` (n, 2), or None for a way that is no gate: shorter than
    GATE_MIN_OPENING between its ends, which includes a ring. A bent way is taken as the straight
    opening between its ends. `fence` is the height of the tallest fence or wall that meets the
    gate: the piers are never lower than it."""
    line = np.asarray(line, dtype=float)[:, :2]
    if len(line) < 2 or not np.isfinite(line).all():
        return None
    a, b = line[0], line[-1]
    opening = float(np.hypot(*(b - a)))
    if opening < GATE_MIN_OPENING:
        return None
    u = (b - a) / opening
    height = min(max(height, GATE_HEIGHT_MIN), GATE_HEIGHT_MAX)
    pier = float(np.clip(0.28 + 0.1 * opening, *PIER_WIDTH_RANGE))
    if opening <= GATE_SINGLE_MAX:
        hinges, length = (0,), opening
    else:
        hinges, length = (0, 1), min(opening / 2, GATE_LEAF_MAX)
    top = min(max(height, fence), GATE_HEIGHT_MAX) + PIER_ABOVE_LEAF
    return GatePlan(a, b, u, np.array([-u[1], u[0]]), opening, height, pier, length, hinges, top)


def _disc(parts, colour, xy, z: float, radius: float, sides: int, rot: float = 0.0) -> None:
    """A flat cap facing up: a fan of `sides` triangles over each of `xy` (n, 2)."""
    if not len(xy):
        return
    ring = _ring(xy, z, radius, sides, rot)
    nxt = np.roll(np.arange(sides), -1)
    centre = np.concatenate([xy, np.full((len(xy), 1), z)], axis=1)
    centres = np.broadcast_to(centre[:, None, :], ring.shape)
    _emit(parts, colour, np.stack([centres, ring, ring[:, nxt]], axis=2).reshape(-1, 3, 3))


def _lathe(parts, colour, xy, z: float, profile, scale: float, apex: float, sides: int, rot: float) -> None:
    """A turned shape standing on `z` at each of `xy` (n, 2): rings from `profile` (height, radius)
    pairs, scaled; closed with a flat top, and a point `apex` above it where `apex` is not zero."""
    for (h0, r0), (h1, r1) in zip(profile[:-1], profile[1:]):
        _frusta(parts, colour, xy, z + h0 * scale, z + h1 * scale, r0 * scale, r1 * scale, sides, rot)
    top_h, top_r = profile[-1]
    if apex:
        _cones(parts, colour, xy, z + top_h * scale, z + apex * scale, top_r * scale, sides, rot)
    else:
        _disc(parts, colour, xy, z + top_h * scale, top_r * scale, sides, rot)


def _finial_kind(opening: float, height: float) -> str:
    if opening < PIER_FINIAL_FROM:
        return "cap"
    return "urn" if opening >= PIER_URN_FROM or height >= PIER_URN_TALL else "ball"


def _pier_stone(tags: dict | None) -> str:
    kinds = (tags or {}).get("material", "").lower().replace(" ", "").split(";")
    return next((PIER_COLOURS[k] for k in kinds if k in PIER_COLOURS), PIER_STONE)


def _pier(parts, plan: GatePlan, k: int, stone: str, wooden: bool) -> None:
    """A square pier: plinth, shaft, projecting cap and a finial (a pyramid, a ball or an urn). A
    wooden gate hangs on plain wooden posts instead."""
    width = plan.pier_width
    centre = plan.pier_centre(k)
    axes = np.array([[*plan.u, 0.0], [*plan.n, 0.0], [0.0, 0.0, 1.0]])
    yaw = math.atan2(plan.u[1], plan.u[0]) + math.pi / 4  # rotates a ring's corners onto the pier's
    xy = centre[None]
    top = plan.pier_top

    def slab(z0, z1, share, shade=1.0, skip=_BOTTOM, hexcode=stone):
        half = np.array([share * width, share * width, z1 - z0]) / 2
        _block(parts, colour_bytes(hexcode, shade), np.array([*centre, (z0 + z1) / 2]), axes, half, skip)

    if wooden:
        slab(0.0, plan.height + 0.12, 0.7, hexcode=WOOD_COLOUR)  # a plain post, its top a little proud
        slab(plan.height + 0.12, plan.height + 0.18, 0.8, shade=1.1, hexcode=WOOD_COLOUR, skip=_BOTTOM)
        return
    base = PIER_BASE + PIER_STEP
    slab(0.0, PIER_BASE, PIER_PLINTH, 0.9)
    slab(PIER_BASE, base, PIER_STEP_SHARE, 0.95)
    slab(base, top, 1.0, skip=((2, -1), (2, 1)))
    # A shallow panel on each face of the shaft, a shade darker, as cut stone piers have.
    panel = np.array([-0.5, 0.5]) * PIER_PANEL * width
    for direction in (plan.u, -plan.u, plan.n, -plan.n):
        right = np.array([-direction[1], direction[0]])  # to the right of someone looking at the face
        face = centre + direction * (width / 2 + PIER_PANEL_PROUD)
        a, b = face + right * panel[0], face + right * panel[1]
        z0, z1 = base + PIER_PANEL_MARGIN, top - PIER_PANEL_MARGIN
        _emit(parts, colour_bytes(stone, 0.88), np.array([[[*a, z0], [*b, z0], [*b, z1], [*a, z1]]]))
    slab(top, top + PIER_CORNICE, PIER_CAP, 1.08, skip=())
    kind = _finial_kind(plan.opening, plan.height)
    radius = width * PIER_CAP / 2 * math.sqrt(2)  # from the middle to a corner of the cornice
    if kind == "cap":
        _cones(parts, colour_bytes(stone, 1.05), xy, top + PIER_CORNICE, top + PIER_CORNICE + 0.6 * width, radius * 0.95, 4, yaw)
        return
    block = top + PIER_CORNICE + PIER_NECK
    slab(top + PIER_CORNICE, block, 0.85, 1.0, skip=_BOTTOM)
    scale = width / 0.55
    if kind == "urn":
        _lathe(parts, colour_bytes(stone, 0.97), xy, block, URN_PROFILE, scale, URN_APEX, PIER_SIDES, yaw)
    else:
        r = 0.19 * scale  # a ball: a stack of rings
        ball = tuple((z * r, f * r) for z, f in ((0.0, 0.62), (0.38, 0.95), (1.0, 1.0), (1.62, 0.95), (2.0, 0.62)))
        _lathe(parts, colour_bytes(stone, 0.97), xy, block, ball, 1.0, 0.25 * r, PIER_SIDES, yaw)


def _leaf(parts, colour, origin, direction, length: float, height: float, style: str) -> None:
    """One gate leaf standing on its hinge `origin` (x, y) and reaching `length` along `direction`,
    in the plane of the two. `style` is "railing" (bars between rails, like the iron fences),
    "boards" (a wooden gate) or "frame" (rails with a brace, for wire and other plain gates)."""
    o = np.asarray(origin, dtype=float)
    t = np.asarray(direction, dtype=float)
    e = o + t * length
    yaw = math.atan2(t[1], t[0])
    ends = np.array([o, e])
    low = GATE_LOW_RAIL
    top_rail = height - min(0.24, 0.14 * height) - 0.05
    if style == "boards":
        axes = np.array([[*t, 0.0], [-t[1], t[0], 0.0], [0.0, 0.0, 1.0]])
        z0, z1 = 0.1, height
        _block(parts, colour, np.array([*(o + e) / 2, (z0 + z1) / 2]), axes, np.array([length, GATE_PLANK, z1 - z0]) / 2, _BOTTOM)
        for z in (0.3, height - 0.3):
            _beam(parts, colour, (*o, z), (*e, z), GATE_BATTEN, 0.12)
        return
    _frusta(parts, colour, ends, 0.08, height - 2 * GATE_BALL, GATE_STILE, GATE_STILE, 4, yaw + math.pi / 4)
    _balls(parts, colour, ends, height - GATE_BALL, GATE_BALL)
    rails = [top_rail, low] + ([(top_rail + low) / 2] if height >= 1.3 else [])
    for z in rails:
        _beam(parts, colour, (*o, z), (*e, z), GATE_RAIL_ACROSS, GATE_RAIL_TALL)
    if style == "frame":
        _beam(parts, colour, (*o, low), (*e, top_rail), GATE_RAIL_ACROSS, GATE_RAIL_TALL)  # the brace
        return
    room = length - 2 * GATE_BAR_GAP
    if room <= 0:
        return
    count = max(1, round(room / GATE_BAR_SPACING))
    along = GATE_BAR_GAP + (np.arange(count) + 0.5) * room / count
    _cones(parts, colour, o + along[:, None] * t, low, height, IRON_BAR_RADIUS, 3, yaw + math.pi / 2)


def _leaf_style(material: str | None) -> str:
    material = (material or "").lower()
    if material in GATE_WOOD:
        return "boards"
    return "frame" if material == "wire" else "railing"


def add_gate(
    builder: MeshBuilder, line, height: float, colour: str | None = None, material: str | None = None,
    feature: int = 0, side: int = 1, tags: dict | None = None, piers: tuple = (True, True), fence: float = 0.0,
) -> None:
    """Add the gate in the opening along `line` (n, 2), tile-relative metres. The leaves are swung
    toward `side` (+1 the left of the way, -1 the right; see `gate_swing_side`). `piers` says which
    of the two ends gets a pier, and `fence` is the height of the tallest fence or wall that meets
    the gate (the piers rise at least that high). `colour` (hex) is the iron's; the piers are
    granite, or brick or concrete where `tags` say so. A way that is no gate (see `gate_plan`)
    adds nothing."""
    plan = gate_plan(line, height, fence)
    if plan is None:
        return
    parts = _Parts()
    wooden = (material or "").lower() in GATE_WOOD
    stone = _pier_stone(tags)
    for k in (0, 1):
        if piers[k]:
            _pier(parts, plan, k, stone, wooden)
    style = _leaf_style(material)
    iron = colour_bytes(WOOD_COLOUR if style == "boards" and not colour else colour or FENCE_COLOUR)
    for k in plan.hinges:
        origin, direction = plan.leaf(k, -1 if side < 0 else 1)
        _leaf(parts, iron, origin, direction, plan.leaf_length, plan.height, style)
    parts.flush(builder, feature)


def gate_swing_side(line, height: float, blocked=None, green=None) -> int:
    """Which side of a gate its leaves are swung to: +1 the left of the way, -1 the right. The
    leaf that stands in less of `blocked` (carriageways, paths, buildings, water: shapely
    geometry, or None) wins; where that is the same, the side with more `green` (lawn). On a tie it is +1."""
    plan = gate_plan(line, height)
    if plan is None:
        return 1
    score = {}
    for side in (1, -1):
        sweep = unary_union([
            LineString([hinge, hinge + direction * plan.leaf_length]).buffer(GATE_STILE * 4, cap_style="flat")
            for hinge, direction in (plan.leaf(k, side) for k in plan.hinges)
        ])
        score[side] = tuple(
            0.0 if geometry is None or geometry.is_empty else sweep.intersection(geometry).area for geometry in (blocked, green)
        )
    (block_left, green_left), (block_right, green_right) = score[1], score[-1]
    if abs(block_left - block_right) > 0.1:
        return 1 if block_left < block_right else -1
    return 1 if green_left >= green_right else -1


def join_gates(barriers: list) -> list:
    """The barriers with gates that meet end to end made one: a double gate is often mapped as one way
    for each leaf, and a pier between the two would stand in the middle of the opening."""
    barriers = list(barriers)
    while True:
        for i, j in itertools.combinations([k for k, b in enumerate(barriers) if b.kind == "gate"], 2):
            merged = _meet(barriers[i], barriers[j])
            if merged is not None:
                barriers[i] = merged
                del barriers[j]
                break
        else:
            return barriers


def _meet(first, second):
    """`first` and `second` as one gate if one's end is the other's start or end, and the line
    carries on from one to the other within 60 degrees; else None."""
    a, b = np.asarray(first.line.coords)[:, :2], np.asarray(second.line.coords)[:, :2]
    for head in (a, a[::-1]):  # `head` runs up to the join, `tail` on from it
        for tail in (b, b[::-1]):
            if np.hypot(*(head[-1] - tail[0])) > 1e-3:
                continue
            into, out = head[-1] - head[-2], tail[1] - tail[0]
            norm = np.hypot(*into) * np.hypot(*out)
            if norm < 1e-9 or float(into @ out) / norm < 0.5:
                continue
            return replace(
                first, line=LineString(np.concatenate([head, tail[1:]])), height=max(first.height, second.height),
                colour=first.colour or second.colour, material=first.material or second.material,
                tags={**second.tags, **first.tags},
            )
    return None


def add_barrier(
    builder: MeshBuilder, kind: str, line, height: float, width: float,
    colour: str | None = None, material: str | None = None, feature: int = 0,
) -> None:
    """Add one barrier along `line` (n, 2), tile-relative metres. `colour` (hex) overrides the
    default for fences, chains, guard rails, bollards and the iron of gates; walls, hedges and
    kerbs take theirs from `material`, and a fence's `material` decides whether it is railings."""
    line = np.asarray(line, dtype=float)[:, :2]
    if len(line) < 2:
        return
    if kind == "gate":
        add_gate(builder, line, height, colour, material, feature)
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

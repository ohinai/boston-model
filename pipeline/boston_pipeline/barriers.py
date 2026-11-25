"""Fences, walls, hedges, chains, kerbs, guard rails and rows of bollards, in z-up tile metres.

Everything is low-poly and flat-shaded, with the ground at z = 0 and counter-clockwise
front faces. Faces nobody can see (the bottom of a post, the ends of a rail that
runs into posts) are not drawn, so a long fence stays cheap.
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
CHAIN_POST, CHAIN_POST_SPACING = 0.09, 2.5
CHAIN_LINK = 0.025  # thickness of the chain
CHAIN_HANG = 0.08  # the chain is hung this far below the post tops
CHAIN_SAG = 0.12  # and sags this much further at the middle
CHAIN_SHOULDER = 0.2  # share of a span taken by each sloping end of the chain
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
    """A square-sectioned bar from 3D point `a` to `b`, `across` wide and `tall` thick."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    along = delta / length
    left = np.cross(_UP, along)
    left /= np.linalg.norm(left)
    axes = np.array([along, left, np.cross(along, left)])
    _block(parts, colour, (a + b) / 2, axes, np.array([length, across, tall]) / 2, () if ends else _ENDS)


def _post(parts, colour, x, y, height: float, size: float) -> None:
    """A square post standing on the ground, open at the bottom."""
    _block(parts, colour, np.array([x, y, height / 2]), np.eye(3), np.array([size, size, height]) / 2, _BOTTOM)


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


def _chain(parts, line, height, colour) -> None:
    posts = _stations(line, CHAIN_POST_SPACING)
    for x, y in posts:
        _post(parts, colour, x, y, height, CHAIN_POST)
    top = height - CHAIN_HANG
    # Between neighbouring posts the chain drops from the post tops to its lowest
    # point and runs level across the middle: three straight pieces.
    closed = len(line) > 2 and np.allclose(line[0], line[-1])
    pairs = list(zip(posts[:-1], posts[1:])) + ([(posts[-1], posts[0])] if closed and len(posts) > 1 else [])
    for a, b in pairs:
        shape = [(0.0, top), (CHAIN_SHOULDER, top - CHAIN_SAG), (1 - CHAIN_SHOULDER, top - CHAIN_SAG), (1.0, top)]
        points = [(*(a + t * (b - a)), z) for t, z in shape]
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
    theirs from `material`."""
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
        _fence(parts, line, height, colour_bytes(colour or FENCE_COLOUR))
    elif kind == "chain":
        _chain(parts, line, height, colour_bytes(colour or FENCE_COLOUR))
    elif kind == "guard_rail":
        _guard_rail(parts, line, height, colour_bytes(colour or GUARD_RAIL_COLOUR))
    else:
        raise ValueError(f"unknown barrier kind {kind!r}")
    if parts._count:
        parts.flush(builder, feature)

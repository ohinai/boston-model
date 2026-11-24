"""Footbridges: an arched deck with sides, underside and railings, in z-up tile metres."""

from __future__ import annotations

import numpy as np

from .mesh import MeshBuilder

STATION = 1.5  # target spacing of cross-sections along the centre line, metres
MIN_INTERVALS = 8  # even a very short bridge gets at least this many intervals
LIP = 0.03  # height of the deck's top at its ends, so it never z-fights the ground
DECK_DEPTH = 0.35  # thickness of the deck below its top surface
RAIL_INSET = 0.08  # centre of the railing from the deck edge
RAIL_HEIGHT = 1.1  # top of the posts above the deck top
POST_SIZE = 0.07  # square post side
RAIL_WIDTH, RAIL_THICK = 0.06, 0.05  # horizontal rail cross-section (across, tall)
RAIL_LEVELS = (0.25, 0.65, 1.1)  # centre height of each horizontal rail above the deck top
MAX_MITRE = 3.0  # cap on how much a sharp bend may stretch the deck width


def _height(s, length: float, rise: float):
    return LIP + rise * np.sin(np.pi * np.asarray(s) / length)


def deck_height(along: float, length: float, rise: float) -> float:
    """Height of the deck's top surface `along` metres from one end of a bridge."""
    return float(_height(along, length, rise))


def deck_profile(length: float, rise: float, samples: int) -> np.ndarray:
    """Heights of the deck's top surface at `samples` evenly spaced stations from one end to the other."""
    return _height(np.linspace(0.0, length, samples), length, rise)


def _emit(builder: MeshBuilder, quads: np.ndarray, colour, feature: int) -> None:
    """Add flat-shaded quads (k, 4, 3), each with a unit normal agreeing with its counter-clockwise winding."""
    if not len(quads):
        return
    normal = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]
    indices = np.arange(len(quads))[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])
    builder.add(quads.reshape(-1, 3), np.repeat(normal, 4, axis=0), indices, colour=colour, feature=feature)


def _strip(lt, rt, lb, rb) -> dict[str, np.ndarray]:
    """Top, bottom, left and right quads between consecutive cross-sections (left/right seen going forward)."""
    def pair(*corners):
        return np.stack(corners, axis=1)

    return {
        "top": pair(rt[:-1], rt[1:], lt[1:], lt[:-1]),
        "bottom": pair(lb[:-1], lb[1:], rb[1:], rb[:-1]),
        "left": pair(lt[:-1], lt[1:], lb[1:], lb[:-1]),
        "right": pair(rb[:-1], rb[1:], rt[1:], rt[:-1]),
    }


def _stations(line: np.ndarray):
    """Stations (m, 2) along the line, their distance s, and their (mitred) left offset vectors (m, 2)."""
    keep = np.r_[True, np.hypot(*np.diff(line, axis=0).T) > 1e-6]
    line = line[keep]
    delta = np.diff(line, axis=0)
    seg_len = np.hypot(delta[:, 0], delta[:, 1])
    step = min(STATION, seg_len.sum() / MIN_INTERVALS)
    pieces = np.maximum(1, np.ceil(seg_len / step - 1e-9).astype(int))
    seg_normal = np.column_stack([-delta[:, 1], delta[:, 0]]) / seg_len[:, None]

    points, owner, at_vertex = [], [], []
    for i, count in enumerate(pieces):
        t = (np.arange(count) / count)[:, None]
        points.append(line[i] + t * delta[i])
        owner += [i] * count
        at_vertex += [True] + [False] * (count - 1)
    points.append(line[-1:])
    owner.append(len(delta) - 1)
    at_vertex.append(True)
    points, owner, at_vertex = np.concatenate(points), np.array(owner), np.array(at_vertex)

    normal = seg_normal[owner].copy()
    for k in np.flatnonzero(at_vertex)[1:-1]:  # interior original vertices: mitre the adjacent normals
        before, after = seg_normal[owner[k] - 1], seg_normal[owner[k]]
        mitre = before + after
        mitre /= max(np.linalg.norm(mitre), 1e-9)
        normal[k] = mitre / max(float(mitre @ after), 1 / MAX_MITRE)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    return points, s, normal


def add_bridge(
    builder: MeshBuilder, line: np.ndarray, width: float, *,
    rise: float,
    deck_colour: np.ndarray, structure_colour: np.ndarray, rail_colour: np.ndarray,
    railings: bool = True, feature: int = 0,
) -> None:
    """A deck along `line` ((n, 2), tile-relative) arching `rise` metres over its mid-span."""
    points, s, normal = _stations(np.asarray(line, dtype=np.float64))
    length = s[-1]
    top = _height(s, length, rise)

    def at(offset: float, z) -> np.ndarray:
        """3D positions at a signed lateral offset (left positive) and height z."""
        return np.column_stack([points + normal * offset, np.broadcast_to(z, len(points))])

    half = width / 2
    lt, rt, lb, rb = at(half, top), at(-half, top), at(half, top - DECK_DEPTH), at(-half, top - DECK_DEPTH)
    deck = _strip(lt, rt, lb, rb)
    _emit(builder, deck["top"], deck_colour, feature)
    caps = np.stack([[lb[0], rb[0], rt[0], lt[0]], [lb[-1], lt[-1], rt[-1], rb[-1]]])
    _emit(builder, np.concatenate([deck["bottom"], deck["left"], deck["right"], caps]), structure_colour, feature)
    if not railings:
        return

    tangent = np.column_stack([normal[:, 1], -normal[:, 0]])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    lateral = normal / np.linalg.norm(normal, axis=1)[:, None]
    posts = np.unique(np.r_[np.arange(0, len(points), 2), len(points) - 1])
    corner = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) * POST_SIZE / 2  # (along, across), ccw from above
    for side in (1, -1):
        offset = side * (half - RAIL_INSET)
        for level in RAIL_LEVELS:
            c = at(offset, top + level)
            across = np.column_stack([normal * (RAIL_WIDTH / 2), np.zeros(len(c))])
            up = np.array([0, 0, RAIL_THICK / 2])
            strip = _strip(c + across + up, c - across + up, c + across - up, c - across - up)
            _emit(builder, np.concatenate(list(strip.values())), rail_colour, feature)
        base = points[posts] + lateral[posts] * offset
        ring = (base[:, None, :] + corner[None, :, 0:1] * tangent[posts][:, None, :]
                + corner[None, :, 1:2] * lateral[posts][:, None, :])  # (k, 4, 2)
        z0, z1 = top[posts], top[posts] + RAIL_HEIGHT
        low = np.concatenate([ring, np.broadcast_to(z0[:, None, None], (len(posts), 4, 1))], axis=2)
        high = np.concatenate([ring, np.broadcast_to(z1[:, None, None], (len(posts), 4, 1))], axis=2)
        nxt = np.roll(np.arange(4), -1)
        sides = np.stack([low, low[:, nxt], high[:, nxt], high], axis=2).reshape(-1, 4, 3)
        _emit(builder, np.concatenate([sides, high]), rail_colour, feature)

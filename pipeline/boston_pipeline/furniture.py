"""Street and park furniture, monuments and fountains as low-poly flat-shaded meshes.

Everything is in z-up metres with the ground at z = 0, relative to a tile origin,
counter-clockwise front faces and one normal per face (every face owns its vertices).
`add_object` is the single entry point; the helpers (`box`, `prism`, `pyramid`,
`disc`, `annulus`) are small enough to reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh import MeshBuilder, colour_bytes

# --- Dimensions (metres) -------------------------------------------------------
BENCH_LENGTH, BENCH_LENGTH_RANGE = 1.8, (1.0, 4.0)
BENCH_DEPTH, BENCH_SEAT_TOP, BENCH_SEAT_THICK = 0.45, 0.45, 0.05
BENCH_BACK_FROM, BENCH_BACK_TO, BENCH_BACK_THICK = 0.5, 0.88, 0.04
BENCH_SUPPORT_THICK, BENCH_SUPPORT_INSET = 0.06, 0.15  # end supports, distance from the ends

LAMP_POST_HEIGHT, LAMP_POST_RANGE = 4.9, (2.5, 14.0)
LAMP_BASE_RADIUS, LAMP_BASE_HEIGHT = 0.14, 0.7
LAMP_LANTERN_RADIUS, LAMP_LANTERN_HEIGHT, LAMP_CAP_HEIGHT = 0.2, 0.55, 0.12
LAMP_ARM_HEIGHT, LAMP_ARM_LENGTH = 8.0, 1.6
LAMP_HEAD = (0.6, 0.25, 0.12)  # along the arm, across it, thick

BIN_RADIUS, BIN_HEIGHT = 0.28, 0.95
FOUNTAIN_PEDESTAL_HEIGHT_RANGE = (0.8, 12.0)
FLAGPOLE_HEIGHT = 12.0
FLAGPOLE_BALL_RADIUS = 0.1

PLAQUE_SIZE = (0.6, 0.3, 0.5)  # wide, deep, tall
STELE_WIDTH, STELE_DEPTH, STELE_HEIGHT = 1.0, 0.28, 1.5
STELE_BASE = (1.3, 0.5, 0.2)
OBELISK_HEIGHT, OBELISK_BASE_WIDTH, OBELISK_BASE_HEIGHT = 6.0, 1.6, 0.6
OBELISK_SHAFT_WIDTHS = (1.0, 0.65)
COLUMN_HEIGHT, COLUMN_MIN_RADIUS = 8.0, 0.25
STATUE_HEIGHT, STATUE_PLINTH_SHARE, STATUE_MIN_PLINTH_TOTAL = 3.4, 0.45, 1.6
STATUE_FOOT_HEIGHT, STATUE_FOOT_EXTRA = 0.15, 0.2
MONUMENT_HEIGHT, MONUMENT_MIN_BASE = 9.0, 2.5
FOUNTAIN_DIAMETER, FOUNTAIN_DIAMETER_RANGE = 3.0, (1.2, 20.0)
FOUNTAIN_WALL, FOUNTAIN_WALL_HEIGHT, FOUNTAIN_WATER_Z = 0.25, 0.5, 0.38
FOUNTAIN_HEIGHT = 1.6

# --- Colours -------------------------------------------------------------------
WOOD, IRON = "#6b4a2f", "#2b2b2d"
LAMP_POST_COLOUR, LAMP_GLASS, LAMP_ARM_COLOUR, LAMP_HEAD_COLOUR = "#23302a", "#e9e2c8", "#8a8d8f", "#4a4d50"
BIN_COLOUR, BIN_OPENING = "#26332b", "#0f1411"
WATER_FOUNTAIN, FLAGPOLE_COLOUR = "#6f7275", "#d8d8d4"
PLAQUE_STONE, PLAQUE_BRONZE = "#8f8b84", "#5b4a34"
STELE_STONE, PLINTH_STONE, MONUMENT_STONE = "#a9a59b", "#b3aea3", "#b3aea3"
FOUNTAIN_STONE, FOUNTAIN_WATER = "#a9a59b", "#3f6f85"
FIGURE_COLOURS = {
    "bronze": "#4b4f44",
    "marble": "#d9d6cf",
    "granite": "#a9a59b",
    "stone": "#a9a59b",
    "steel": "#7d8286",
    "metal": "#7d8286",
}


# --- Geometry helpers ----------------------------------------------------------
class _Parts:
    """Collects coloured geometry and hands it to a MeshBuilder in one call."""

    def __init__(self) -> None:
        self._p: list[np.ndarray] = []
        self._n: list[np.ndarray] = []
        self._i: list[np.ndarray] = []
        self._c: list[np.ndarray] = []
        self._count = 0

    def add(self, positions, normals, indices, colour) -> None:
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        count = len(positions)
        self._p.append(positions)
        self._n.append(np.broadcast_to(np.asarray(normals, dtype=np.float64), (count, 3)))
        self._i.append(np.asarray(indices, dtype=np.int64).ravel() + self._count)
        self._c.append(np.broadcast_to(np.asarray(colour, dtype=np.uint8), (count, 4)))
        self._count += count

    def flush(self, builder: MeshBuilder, feature: int, base: float = 0.0) -> None:
        builder.add(
            np.concatenate(self._p) + np.array([0.0, 0.0, base]), np.concatenate(self._n), np.concatenate(self._i),
            colour=np.concatenate(self._c), feature=feature,
        )


_QUAD = np.array([0, 1, 2, 0, 2, 3])
_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])


def box(parts, colour, x, y, z0, z1, width, depth, yaw=0.0) -> None:
    """An oriented box. `width` runs along the right-hand side of `yaw` (degrees clockwise
    from north, the box's front), `depth` along its front."""
    t = math.radians(yaw)
    axes = np.array([(math.cos(t), -math.sin(t), 0.0), (math.sin(t), math.cos(t), 0.0), (0.0, 0.0, 1.0)])
    half = np.array([width, depth, z1 - z0]) / 2
    centre = np.array([x, y, (z0 + z1) / 2])
    positions, normals = [], []
    for k in range(3):
        for sign in (1, -1):
            i, j = (k + 1) % 3, (k + 2) % 3
            if sign < 0:
                i, j = j, i  # keeps tangent1 x tangent2 pointing outward
            face = centre + sign * half[k] * axes[k]
            positions.append(face + _CORNERS[:, :1] * half[i] * axes[i] + _CORNERS[:, 1:] * half[j] * axes[j])
            normals.append(np.repeat((sign * axes[k])[None], 4, axis=0))
    parts.add(np.concatenate(positions), np.concatenate(normals), (np.arange(6)[:, None] * 4 + _QUAD), colour)


def _ring(x, y, z, radius, sides, rot):
    """sides + 1 points (the last repeats the first) counter-clockwise around (x, y)."""
    angle = math.radians(rot) + 2 * math.pi * np.arange(sides + 1) / sides
    return np.column_stack([x + radius * np.cos(angle), y + radius * np.sin(angle), np.full(sides + 1, z)]), angle


def disc(parts, colour, x, y, z, radius, sides=8, rot=0.0, up=True) -> None:
    """A flat n-gon facing up (or down)."""
    ring = _ring(x, y, z, radius, sides, rot)[0][:-1]
    k = np.arange(1, sides - 1)
    tris = np.column_stack([np.zeros_like(k), k, k + 1])
    parts.add(ring, (0, 0, 1 if up else -1), tris if up else tris[:, ::-1], colour)


def annulus(parts, colour, x, y, z, r_outer, r_inner, sides=8, rot=0.0) -> None:
    """A flat upward-facing ring between two radii."""
    outer, inner = _ring(x, y, z, r_outer, sides, rot)[0], _ring(x, y, z, r_inner, sides, rot)[0]
    quads = np.stack([outer[:-1], outer[1:], inner[1:], inner[:-1]], axis=1)
    parts.add(quads.reshape(-1, 3), (0, 0, 1), np.arange(sides)[:, None] * 4 + _QUAD, colour)


def prism(parts, colour, x, y, z0, z1, r0, r1=None, sides=8, rot=0.0, top=False, bottom=False, inward=False) -> None:
    """A vertical n-sided prism; radii are circumradii and r1 != r0 tapers it.
    `inward` makes the side walls face the axis (the inside of a basin)."""
    r1 = r0 if r1 is None else r1
    lo, angle = _ring(x, y, z0, r0, sides, rot)
    hi = _ring(x, y, z1, r1, sides, rot)[0]
    quads = np.stack([lo[:-1], lo[1:], hi[1:], hi[:-1]], axis=1)
    mid = angle[:-1] + math.pi / sides
    slope = (r0 - r1) * math.cos(math.pi / sides) / (z1 - z0)
    normals = np.column_stack([np.cos(mid), np.sin(mid), np.full(sides, slope)])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    indices = np.arange(sides)[:, None, None] * 4 + (_QUAD.reshape(2, 3)[:, ::-1] if inward else _QUAD.reshape(2, 3))
    if inward:
        normals = -normals
    parts.add(quads.reshape(-1, 3), np.repeat(normals, 4, axis=0), indices, colour)
    if top:
        disc(parts, colour, x, y, z1, r1, sides, rot)
    if bottom:
        disc(parts, colour, x, y, z0, r0, sides, rot, up=False)


def _flat_triangles(parts, colour, triangles, centre) -> None:
    """Triangles (t, 3, 3) wound so each faces away from `centre`."""
    triangles = np.array(triangles, dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    flip = np.einsum("ij,ij->i", normals, triangles.mean(axis=1) - centre) < 0
    triangles[flip] = triangles[flip][:, ::-1]
    normals[flip] *= -1
    parts.add(triangles.reshape(-1, 3), np.repeat(normals, 3, axis=0), np.arange(len(triangles) * 3), colour)


def pyramid(parts, colour, x, y, z0, z1, r, sides=4, rot=0.0) -> None:
    """A pointed top: an n-gon base at z0 (circumradius r) rising to an apex at z1."""
    base = _ring(x, y, z0, r, sides, rot)[0]
    apex = np.broadcast_to([x, y, z1], (sides, 3))
    _flat_triangles(parts, colour, np.stack([base[:-1], base[1:], apex], axis=1), np.array([x, y, z0 - 1e3]))


def octahedron(parts, colour, x, y, z, r) -> None:
    signs = np.array([(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=float)
    corners = np.stack([signs[:, 0, None] * [r, 0, 0], signs[:, 1, None] * [0, r, 0], signs[:, 2, None] * [0, 0, r]], axis=1)
    _flat_triangles(parts, colour, corners + [x, y, z], np.array([x, y, z]))


def _at(x, y, direction, u, v):
    """World position of a point `u` to the right of and `v` ahead of (x, y) facing `direction`."""
    t = math.radians(direction)
    return x + u * math.cos(t) + v * math.sin(t), y - u * math.sin(t) + v * math.cos(t)


def _square_rot(direction):
    """`rot` for a 4-sided prism/pyramid whose edges are parallel to a box facing `direction`."""
    return 45.0 - direction


# --- Kinds ---------------------------------------------------------------------
@dataclass
class _Opts:
    height: float | None
    size: float | None
    colour: str | None
    material: str | None
    variant: str | None
    shade: float  # small per-object lightness variation

    def paint(self, default: str) -> np.ndarray:
        """The object's main material: the stated colour exactly, else the default with variation."""
        return colour_bytes(self.colour) if self.colour else colour_bytes(default, self.shade)

    def figure(self) -> np.ndarray:
        if self.colour:
            return colour_bytes(self.colour)
        return colour_bytes(FIGURE_COLOURS.get((self.material or "bronze").lower(), FIGURE_COLOURS["bronze"]), self.shade)

    def tall(self, default: float, lo: float = 0.3, hi: float = 200.0) -> float:
        return min(max(self.height if self.height is not None else default, lo), hi)


def _bench(p, x, y, d, o: _Opts) -> None:
    length = min(max(o.size if o.size is not None else BENCH_LENGTH, BENCH_LENGTH_RANGE[0]), BENCH_LENGTH_RANGE[1])
    wood, iron = o.paint(WOOD), colour_bytes(IRON)
    seat_bottom = BENCH_SEAT_TOP - BENCH_SEAT_THICK
    sx, sy = _at(x, y, d, 0, 0)
    box(p, wood, sx, sy, seat_bottom, BENCH_SEAT_TOP, length, BENCH_DEPTH, d)
    rear = -BENCH_DEPTH / 2 + BENCH_BACK_THICK / 2  # the sitter looks along `d`, so the back is behind them
    backless = o.variant == "backless"
    if not backless:
        bx, by = _at(x, y, d, 0, rear)
        box(p, wood, bx, by, BENCH_BACK_FROM, BENCH_BACK_TO, length, BENCH_BACK_THICK, d)
    for side in (-1, 1):
        u = side * (length / 2 - BENCH_SUPPORT_INSET)
        ex, ey = _at(x, y, d, u, 0)
        box(p, iron, ex, ey, 0.0, seat_bottom, BENCH_SUPPORT_THICK, BENCH_DEPTH * 0.9, d)
        if not backless:  # a rear post carrying the backrest
            px, py = _at(x, y, d, u, rear)
            box(p, iron, px, py, seat_bottom, BENCH_BACK_TO - 0.02, BENCH_SUPPORT_THICK, BENCH_BACK_THICK, d)


def _lamp(p, x, y, d, o: _Opts) -> None:
    if o.variant == "arm":
        top = o.tall(LAMP_ARM_HEIGHT, 2.5, 14.0)
        pole = o.paint(LAMP_ARM_COLOUR)
        prism(p, pole, x, y, 0.0, top, 0.10, 0.06, 8, top=True)
        ax, ay = _at(x, y, d, 0, LAMP_ARM_LENGTH / 2)
        box(p, pole, ax, ay, top - 0.40, top - 0.32, 0.08, LAMP_ARM_LENGTH, d)
        hx, hy = _at(x, y, d, 0, LAMP_ARM_LENGTH - LAMP_HEAD[0] / 2 + 0.1)
        box(p, colour_bytes(LAMP_HEAD_COLOUR), hx, hy, top - 0.52, top - 0.40, LAMP_HEAD[1], LAMP_HEAD[0], d)
        return
    top = o.tall(LAMP_POST_HEIGHT, *LAMP_POST_RANGE)
    cap_z = top - LAMP_CAP_HEIGHT
    lantern_z = cap_z - LAMP_LANTERN_HEIGHT
    post = o.paint(LAMP_POST_COLOUR)
    prism(p, post, x, y, 0.0, LAMP_BASE_HEIGHT, LAMP_BASE_RADIUS, sides=8, top=True)
    prism(p, post, x, y, LAMP_BASE_HEIGHT, lantern_z, 0.08, 0.05, 8)
    prism(p, colour_bytes(LAMP_GLASS), x, y, lantern_z, cap_z, LAMP_LANTERN_RADIUS, sides=6, bottom=True)
    pyramid(p, post, x, y, cap_z, top, LAMP_LANTERN_RADIUS + 0.06, 6)


def _waste_basket(p, x, y, d, o: _Opts) -> None:
    top = o.tall(BIN_HEIGHT, 0.4, 2.0)
    # Body widens slightly toward the rim; a dark disc sits just above the rim to read as the opening.
    prism(p, o.paint(BIN_COLOUR), x, y, 0.0, top - 0.01, BIN_RADIUS, BIN_RADIUS + 0.02, 8, top=True)
    disc(p, colour_bytes(BIN_OPENING), x, y, top, BIN_RADIUS - 0.08, 6)


def _drinking_water(p, x, y, d, o: _Opts) -> None:
    top = o.tall(0.9, 0.5, 1.5)
    box(p, o.paint(WATER_FOUNTAIN), x, y, 0.0, top, 0.32, 0.32, d)
    box(p, colour_bytes(WATER_FOUNTAIN, 1.25), x, y, top, top + 0.06, 0.5, 0.5, d)


def _flagpole(p, x, y, d, o: _Opts) -> None:
    top = o.tall(FLAGPOLE_HEIGHT, 3.0, 60.0)
    colour = o.paint(FLAGPOLE_COLOUR)
    ball = FLAGPOLE_BALL_RADIUS
    prism(p, colour, x, y, 0.0, top - ball, 0.07, 0.04, 8)
    octahedron(p, colour_bytes("#c9a24a"), x, y, top - ball, ball)


def _plaque(p, x, y, d, o: _Opts) -> None:
    w, deep, tall = PLAQUE_SIZE
    box(p, o.paint(PLAQUE_STONE), x, y, 0.0, tall, w, deep, d)
    fx, fy = _at(x, y, d, 0, deep / 2 + 0.01)
    box(p, colour_bytes(PLAQUE_BRONZE), fx, fy, tall * 0.2, tall * 0.8, w * 0.7, 0.02, d)


def _stele(p, x, y, d, o: _Opts) -> None:
    top = o.tall(STELE_HEIGHT, 0.6)
    stone = o.paint(STELE_STONE)
    bw, bd, bh = STELE_BASE
    box(p, stone, x, y, 0.0, bh, bw, bd, d)
    box(p, stone, x, y, bh, top, STELE_WIDTH, STELE_DEPTH, d)


def _obelisk(p, x, y, d, o: _Opts) -> None:
    top = o.tall(OBELISK_HEIGHT, 2.0)
    k = min(max(top / OBELISK_HEIGHT, 0.6), 2.5)  # footprint grows with very tall obelisks
    stone = o.paint(MONUMENT_STONE)
    base_h = min(OBELISK_BASE_HEIGHT, 0.2 * top)
    tip_h = 0.1 * top
    rot, root2 = _square_rot(d), math.sqrt(2) / 2
    box(p, stone, x, y, 0.0, base_h, OBELISK_BASE_WIDTH * k, OBELISK_BASE_WIDTH * k, d)
    w0, w1 = (w * k for w in OBELISK_SHAFT_WIDTHS)
    prism(p, stone, x, y, base_h, top - tip_h, w0 * root2, w1 * root2, 4, rot)
    pyramid(p, stone, x, y, top - tip_h, top, w1 * root2, 4, rot)


def _column(p, x, y, d, o: _Opts) -> None:
    top = o.tall(COLUMN_HEIGHT, 1.5)
    r = max(top / 22, COLUMN_MIN_RADIUS)
    step, cap = min(0.18, 0.04 * top), min(0.25, 0.06 * top)
    stone = o.paint(MONUMENT_STONE)
    box(p, stone, x, y, 0.0, step, 4.2 * r, 4.2 * r, d)
    box(p, stone, x, y, step, 2 * step, 3.2 * r, 3.2 * r, d)
    prism(p, stone, x, y, 2 * step, top - cap, r, 0.88 * r, 10)
    box(p, stone, x, y, top - cap, top, 2.6 * r, 2.6 * r, d)


def _plinth(p, x, y, d, o: _Opts, total: float) -> float:
    """A stone plinth under a statue (when tall enough). Returns the height the figure starts at."""
    if total < STATUE_MIN_PLINTH_TOTAL:
        return 0.0
    top = STATUE_PLINTH_SHARE * total
    width = 0.36 * total
    stone = colour_bytes(PLINTH_STONE, o.shade)
    box(p, stone, x, y, 0.0, STATUE_FOOT_HEIGHT, width + STATUE_FOOT_EXTRA, width + STATUE_FOOT_EXTRA, d)
    box(p, stone, x, y, STATUE_FOOT_HEIGHT, top, width, width, d)
    return top


def _statue(p, x, y, d, o: _Opts) -> None:
    total = o.tall(STATUE_HEIGHT, 0.2)
    z0 = _plinth(p, x, y, d, o, total)
    fh, colour = total - z0, o.figure()
    # Abstract standing form: a tapering six-sided body under a small eight-sided head.
    prism(p, colour, x, y, z0, z0 + 0.8 * fh, 0.15 * fh, 0.10 * fh, 6, top=True)
    prism(p, colour, x, y, z0 + 0.8 * fh, total, 0.075 * fh, sides=8, top=True)


def _sculpture(p, x, y, d, o: _Opts) -> None:
    total = o.tall(STATUE_HEIGHT, 0.2)
    z0 = _plinth(p, x, y, d, o, total)
    fh, colour = total - z0, o.figure()
    box(p, colour, x, y, z0, z0 + 0.55 * fh, 0.30 * fh, 0.30 * fh, d)
    box(p, colour, x, y, z0 + 0.55 * fh, total, 0.22 * fh, 0.22 * fh, d + 30)


def _monument(p, x, y, d, o: _Opts) -> None:
    top = o.tall(MONUMENT_HEIGHT, 2.0)
    width = max(MONUMENT_MIN_BASE, 0.45 * top)
    step, cap = min(0.3, 0.05 * top), 0.06 * top
    stone = o.paint(MONUMENT_STONE)
    for i, share in enumerate((1.0, 0.88, 0.76)):
        box(p, stone, x, y, i * step, (i + 1) * step, width * share, width * share, d)
    root2 = math.sqrt(2) / 2
    prism(p, stone, x, y, 3 * step, top - cap, 0.6 * width * root2, 0.5 * width * root2, 4, _square_rot(d))
    box(p, colour_bytes(MONUMENT_STONE, 1.08), x, y, top - cap, top, 0.55 * width, 0.55 * width, d)


def _fountain(p, x, y, d, o: _Opts) -> None:
    diameter = min(max(o.size if o.size is not None else FOUNTAIN_DIAMETER, FOUNTAIN_DIAMETER_RANGE[0]), FOUNTAIN_DIAMETER_RANGE[1])
    radius = diameter / 2
    wall = min(FOUNTAIN_WALL, radius / 8)
    sides, stone = 12, o.paint(FOUNTAIN_STONE)
    inner = radius - wall
    prism(p, stone, x, y, 0.0, FOUNTAIN_WALL_HEIGHT, radius, sides=sides)
    prism(p, stone, x, y, 0.0, FOUNTAIN_WALL_HEIGHT, inner, sides=sides, inward=True)
    annulus(p, stone, x, y, FOUNTAIN_WALL_HEIGHT, radius, inner, sides)
    disc(p, colour_bytes(FOUNTAIN_WATER), x, y, FOUNTAIN_WATER_Z, inner, sides)
    top = o.tall(FOUNTAIN_HEIGHT, *FOUNTAIN_PEDESTAL_HEIGHT_RANGE)
    pr = max(0.12, 0.12 * radius)
    if o.variant == "nozzle":
        prism(p, stone, x, y, 0.0, top, pr * 0.8, pr * 0.6, 8, top=True)
        return
    prism(p, stone, x, y, 0.0, top, pr, pr * 0.7, 8, top=True)
    bowl = min(0.12, 0.1 * top)
    prism(p, stone, x, y, top - 0.15 * top - bowl, top - 0.15 * top, pr * 1.3, pr * 2.8, 8, top=True)


_KINDS = {
    "bench": _bench,
    "lamp": _lamp,
    "waste_basket": _waste_basket,
    "drinking_water": _drinking_water,
    "flagpole": _flagpole,
    "plaque": _plaque,
    "stele": _stele,
    "obelisk": _obelisk,
    "column": _column,
    "statue": _statue,
    "sculpture": _sculpture,
    "monument": _monument,
    "fountain": _fountain,
}
KINDS: tuple[str, ...] = tuple(_KINDS)


def add_object(
    builder: MeshBuilder,
    kind: str,
    x: float,
    y: float,
    *,
    direction: float = 0.0,
    height: float | None = None,
    size: float | None = None,
    colour: str | None = None,
    material: str | None = None,
    variant: str | None = None,
    seed: int = 0,
    feature: int = 0,
    base: float = 0.0,
) -> None:
    """Add one piece of furniture at (x, y) (tile-relative metres) facing `direction`
    (degrees clockwise from north). Dimensions are in metres; None means the default.
    `base` is the height it stands at: 0 for the ground, more on a bridge pier."""
    if kind not in _KINDS:
        raise ValueError(f"unknown furniture kind {kind!r}; expected one of {', '.join(KINDS)}")
    shade = 1.0 + (np.random.default_rng(seed & 0xFFFFFFFF).random() - 0.5) * 0.08  # +-4%
    parts = _Parts()
    _KINDS[kind](parts, x, y, direction, _Opts(height, size, colour, material, variant, shade))
    parts.flush(builder, feature, base)

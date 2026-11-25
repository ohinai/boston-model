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
# The park lamp: a slim black cast-iron post on a flared foot, a collar, an opal globe and a small cap.
# Six sides everywhere (about 60 triangles a lamp: the pilot has over a thousand of them).
LAMP_FOOT_RADIUS, LAMP_FOOT_HEIGHT = 0.20, 0.8  # the flared base, from its widest to where the shaft starts
LAMP_SHAFT_RADIUS = (0.065, 0.05)  # at the foot, under the collar
LAMP_COLLAR_RADIUS, LAMP_COLLAR_HEIGHT = 0.10, 0.12
LAMP_GLOBE_RADIUS, LAMP_GLOBE_BELOW, LAMP_GLOBE_ABOVE = 0.19, 0.13, 0.19  # widest radius; height under and over it
LAMP_CAP_RADIUS, LAMP_CAP_HEIGHT = 0.07, 0.07
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

# Statues. A statue with no stated height is 3.4 m, pedestal included: a figure a little over life size on a
# pedestal about as tall as itself. Nothing here is measured; these are proportions for "a statue of this kind".
STATUE_HEIGHT, STATUE_MIN_PLINTH_TOTAL = 3.4, 1.6
STATUE_PEDESTAL_SHARE = 0.44  # of the total height, for a standing figure
STATUE_FIGURE_MAX = 3.4  # a standing bronze is never taller than this; a taller monument gets a taller pedestal
STATUE_PEDESTAL_WIDTH, STATUE_PEDESTAL_MAX_WIDTH = 0.34, 2.6  # the die: of the total height, up to this many metres (the footing is wider)
EQUESTRIAN_HEIGHT, EQUESTRIAN_PEDESTAL_SHARE = 6.0, 0.42  # a rider needs a bigger statue than a walker
EQUESTRIAN_PEDESTAL_LENGTH, EQUESTRIAN_PEDESTAL_WIDTH = 0.46, 0.28  # of the total height, along and across the horse
STATUE_GROUND_HEIGHT = 1.2  # a statue the data says stands on the ground (`support=ground`), no pedestal
ANIMAL_LENGTH = 0.9  # nose to tail of an animal statue (`animal=*`) when neither `length` nor `height` is given
MONUMENT_HEIGHT, MONUMENT_MIN_BASE = 9.0, 2.5
FOUNTAIN_DIAMETER, FOUNTAIN_DIAMETER_RANGE = 3.0, (1.2, 20.0)
FOUNTAIN_WALL, FOUNTAIN_WALL_HEIGHT, FOUNTAIN_WATER_Z = 0.25, 0.5, 0.38
FOUNTAIN_HEIGHT = 1.6
BOLLARD_HEIGHT, BOLLARD_RANGE = 0.9, (0.3, 2.0)
BOLLARD_RADIUS, BOLLARD_CAP_RADIUS, BOLLARD_CAP_HEIGHT = 0.09, 0.105, 0.07
BLOCK_SIZE, BLOCK_HEIGHT = 0.6, 0.5

# --- Colours -------------------------------------------------------------------
WOOD, IRON = "#6b4a2f", "#2b2b2d"
LAMP_POST_COLOUR, LAMP_GLASS, LAMP_ARM_COLOUR, LAMP_HEAD_COLOUR = "#1c1e20", "#efece0", "#8a8d8f", "#4a4d50"
BIN_COLOUR, BIN_OPENING = "#26332b", "#0f1411"
WATER_FOUNTAIN, FLAGPOLE_COLOUR = "#6f7275", "#d8d8d4"
PLAQUE_STONE, PLAQUE_BRONZE = "#8f8b84", "#5b4a34"
STELE_STONE, PLINTH_STONE, MONUMENT_STONE = "#a9a59b", "#7f786f", "#b3aea3"  # the plinth is warm grey granite
FOUNTAIN_STONE, FOUNTAIN_WATER = "#a9a59b", "#3f6f85"
BOLLARD_COLOUR, BLOCK_STONE = "#23262a", "#a9a59b"
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


def _frame(x, y, direction, base=0.0, ahead=0.0):
    """A function from (u, v, z) to world (x, y, z): u to the right of and v ahead of (x, y) facing
    `direction`, z up from `base`. `ahead` moves the origin that far along `direction` first."""
    t = math.radians(direction)
    right = np.array([math.cos(t), -math.sin(t), 0.0])
    front = np.array([math.sin(t), math.cos(t), 0.0])
    origin = np.array([x, y, base]) + ahead * front

    def to_world(u, v, z):
        return origin + u * right + v * front + z * np.array([0.0, 0.0, 1.0])

    return to_world


def limb(parts, colour, to_world, a, b, r0, r1=None, sides=4, flat=1.0, closed=True) -> None:
    """A tapered bar from `a` to `b` (local (u, v, z) points, see `_frame`): circumradius r0 at a and r1 at b.
    The cross-section is an ellipse's n-gon: `r` wide across the figure's right-hand side (where the bar allows)
    and `r * flat` the other way. Flat-shaded. Used for the limbs, necks and tails of figures."""
    r1 = r0 if r1 is None else r1
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    axis = b - a
    axis /= np.linalg.norm(axis)
    side = np.array([1.0, 0.0, 0.0])
    if abs(axis @ side) > 0.95:
        side = np.array([0.0, 1.0, 0.0])
    e1 = side - (side @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    angle = 2 * math.pi * (np.arange(sides) + 0.5) / sides
    ring = np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * flat * e2  # (sides, 3), unit-ish
    lo, hi = a + r0 * ring, a + (b - a) + r1 * ring
    world = lambda pts: np.array([to_world(*q) for q in pts])  # noqa: E731
    lo, hi = world(lo), world(hi)
    nxt = np.roll(np.arange(sides), -1)
    triangles = [np.stack([lo, lo[nxt], hi[nxt]], axis=1), np.stack([lo, hi[nxt], hi], axis=1)]
    if closed:
        fan = np.arange(1, sides - 1)
        triangles.append(np.stack([lo[np.zeros_like(fan)], lo[fan], lo[fan + 1]], axis=1))
        triangles.append(np.stack([hi[np.zeros_like(fan)], hi[fan], hi[fan + 1]], axis=1))
    _flat_triangles(parts, colour, np.concatenate(triangles), world([(a + b) / 2])[0])


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
    post, glass = o.paint(LAMP_POST_COLOUR), colour_bytes(LAMP_GLASS)
    cap_z = top - LAMP_CAP_HEIGHT
    globe_top = cap_z
    globe_mid = globe_top - LAMP_GLOBE_ABOVE
    globe_bottom = globe_mid - LAMP_GLOBE_BELOW
    collar_bottom = globe_bottom - LAMP_COLLAR_HEIGHT
    prism(p, post, x, y, 0.0, LAMP_FOOT_HEIGHT, LAMP_FOOT_RADIUS, LAMP_SHAFT_RADIUS[0], 6)  # flared foot
    prism(p, post, x, y, LAMP_FOOT_HEIGHT, collar_bottom, LAMP_SHAFT_RADIUS[0], LAMP_SHAFT_RADIUS[1], 6)
    neck = LAMP_COLLAR_RADIUS * 0.8
    prism(p, post, x, y, collar_bottom, globe_bottom, LAMP_COLLAR_RADIUS, neck, 6)
    prism(p, glass, x, y, globe_bottom, globe_mid, neck, LAMP_GLOBE_RADIUS, 6, bottom=True)
    prism(p, glass, x, y, globe_mid, globe_top, LAMP_GLOBE_RADIUS, LAMP_CAP_RADIUS, 6)
    pyramid(p, post, x, y, cap_z, top, LAMP_CAP_RADIUS, 6)


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


def _bollard(p, x, y, d, o: _Opts) -> None:
    top = o.tall(BOLLARD_HEIGHT, *BOLLARD_RANGE)
    colour = o.paint(BOLLARD_COLOUR)
    cap = min(BOLLARD_CAP_HEIGHT, 0.3 * top)
    prism(p, colour, x, y, 0.0, top - cap, BOLLARD_RADIUS, sides=8)
    prism(p, colour, x, y, top - cap, top, BOLLARD_CAP_RADIUS, sides=8, top=True)


def _block(p, x, y, d, o: _Opts) -> None:
    box(p, o.paint(BLOCK_STONE), x, y, 0.0, o.tall(BLOCK_HEIGHT, 0.2, 2.0), BLOCK_SIZE, BLOCK_SIZE, d)


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


# A pedestal's courses, bottom to top: (footprint relative to the die, from, to as fractions of its height, shade).
# A footing, a base, the tall die that carries the inscription, and a two-step cap that overhangs the die.
PEDESTAL_COURSES = (
    (1.26, 0.00, 0.05, 0.90),
    (1.12, 0.05, 0.18, 0.94),
    (0.88, 0.18, 0.84, 1.00),
    (0.99, 0.84, 0.93, 1.07),
    (0.92, 0.93, 1.00, 1.10),
)


def _plinth(p, x, y, d, o: _Opts, total: float, *, height=None, length=None, width=None) -> float:
    """A stepped granite pedestal under a statue (when tall enough). `length` runs along the way the figure
    faces and `width` across it; both are the die's, the footing is wider. Returns the height it ends at."""
    if total < STATUE_MIN_PLINTH_TOTAL:
        return 0.0
    height = STATUE_PEDESTAL_SHARE * total if height is None else height
    square = min(STATUE_PEDESTAL_WIDTH * total, STATUE_PEDESTAL_MAX_WIDTH)
    length = square if length is None else length
    width = square if width is None else width
    for scale, lo, hi, shade in PEDESTAL_COURSES:
        box(p, colour_bytes(PLINTH_STONE, o.shade * shade), x, y, lo * height, hi * height, width * scale, length * scale, d)
    return height


def _standing_figure(p, x, y, d, z0, fh, colour) -> None:
    """A person in a long coat standing with arms at the sides: legs, a flared coat, shoulders, a head.
    It is a placeholder of the right kind and size, not a likeness. About 130 triangles."""
    to_world = _frame(x, y, d, z0)
    bar = lambda a, b, r0, r1=None, **k: limb(p, colour, to_world, a, b, r0, r1, **k)  # noqa: E731
    prism(p, colour, x, y, z0, z0 + 0.035 * fh, 0.17 * fh, sides=6, top=True)  # the bronze's own base
    for side in (-1, 1):
        bar((side * 0.035 * fh, 0, 0.03 * fh), (side * 0.035 * fh, 0, 0.46 * fh), 0.045 * fh, 0.05 * fh)  # legs
        bar((side * 0.135 * fh, 0, 0.80 * fh), (side * 0.15 * fh, 0.01 * fh, 0.50 * fh), 0.045 * fh, 0.035 * fh)  # arms
    bar((0, 0, 0.30 * fh), (0, 0, 0.60 * fh), 0.125 * fh, 0.095 * fh, sides=6)  # the coat's skirt
    bar((0, 0, 0.58 * fh), (0, 0, 0.82 * fh), 0.13 * fh, 0.20 * fh, flat=0.55)  # waist to shoulders
    bar((0, 0, 0.80 * fh), (0, 0, 0.86 * fh), 0.035 * fh)  # neck
    bar((0, 0, 0.84 * fh), (0, 0, fh), 0.055 * fh, 0.048 * fh, sides=6)  # head


EQUESTRIAN_WITHERS = 0.56  # horse's height at the shoulder, of the whole bronze (hat to hoof)


def _equestrian_figure(p, x, y, d, z0, fh, colour) -> None:
    """A horse standing with one foreleg raised, and a rider in a cocked hat with a sword at his side.
    Proportions are a typical bronze rider's, not any particular statue's. About 280 triangles."""
    h = EQUESTRIAN_WITHERS * fh  # the horse's withers
    # The horse stretches from its tail (-0.64 h) to its nose (0.95 h): centre that on the pedestal.
    to_world = _frame(x, y, d, z0, ahead=-0.155 * h)
    bar = lambda a, b, r0, r1=None, **k: limb(p, colour, to_world, a, b, r0, r1, **k)  # noqa: E731
    at = lambda u, v: to_world(u, v, 0.0)[:2]  # noqa: E731

    cx, cy = at(0, 0.155 * h)
    box(p, colour, cx, cy, z0, z0 + 0.03 * h, 0.56 * h, 1.62 * h, d)  # the bronze's own base
    for v, depth, wide, lo, hi in ((0.24 * h, 0.50 * h, 0.44 * h, 0.42 * h, 1.00 * h), (-0.23 * h, 0.46 * h, 0.46 * h, 0.45 * h, 0.96 * h)):
        bx, by = at(0, v)
        box(p, colour, bx, by, z0 + lo, z0 + hi, wide, depth, d)  # chest and withers; rump and loins
    for side in (-1, 1):
        bar((side * 0.15 * h, -0.33 * h, 0.56 * h), (side * 0.15 * h, -0.38 * h, 0.0), 0.11 * h, 0.065 * h)  # hind legs
    bar((0.15 * h, 0.36 * h, 0.55 * h), (0.15 * h, 0.36 * h, 0.0), 0.105 * h, 0.06 * h)  # standing foreleg
    bar((-0.15 * h, 0.36 * h, 0.55 * h), (-0.15 * h, 0.50 * h, 0.34 * h), 0.105 * h, 0.07 * h)  # raised foreleg
    bar((-0.15 * h, 0.50 * h, 0.34 * h), (-0.15 * h, 0.44 * h, 0.12 * h), 0.07 * h, 0.06 * h)
    bar((0, 0.40 * h, 0.80 * h), (0, 0.62 * h, 1.28 * h), 0.25 * h, 0.14 * h, flat=0.75)  # neck
    bar((0, 0.60 * h, 1.30 * h), (0, 0.90 * h, 0.99 * h), 0.14 * h, 0.07 * h, flat=0.85)  # head
    bar((0, -0.45 * h, 0.92 * h), (0, -0.60 * h, 0.42 * h), 0.05 * h, 0.035 * h)  # tail

    seat, behind = 0.98 * h, -0.02 * h  # where the rider sits: height, and along the horse
    for side in (-1, 1):
        knee = (side * 0.20 * h, behind + 0.20 * h, 0.80 * h)
        bar((side * 0.08 * fh, behind, seat + 0.01 * fh), knee, 0.055 * fh, 0.045 * fh)  # thigh
        bar(knee, (side * 0.21 * h, behind + 0.16 * h, 0.50 * h), 0.045 * fh, 0.035 * fh)  # boot
        bar((side * 0.12 * fh, behind, seat + 0.24 * fh), (side * 0.09 * fh, behind + 0.12 * h, seat + 0.12 * fh), 0.035 * fh, 0.03 * fh)
    bar((0, behind, seat), (0, behind, seat + 0.26 * fh), 0.12 * fh, 0.17 * fh, flat=0.55)  # torso
    bar((0, behind, seat + 0.24 * fh), (0, behind, seat + 0.30 * fh), 0.035 * fh)  # neck
    top = seat + 0.28 * fh
    bar((0, behind, top), (0, behind, 0.92 * fh), 0.055 * fh, 0.048 * fh, sides=6)  # head
    hx, hy = at(0, behind)
    box(p, colour, hx, hy, z0 + 0.915 * fh, z0 + 0.94 * fh, 0.22 * fh, 0.12 * fh, d)  # hat brim
    box(p, colour, hx, hy, z0 + 0.94 * fh, z0 + fh, 0.17 * fh, 0.085 * fh, d)  # and crown
    bar((-0.12 * fh, behind, seat + 0.03 * fh), (-0.24 * h, -0.58 * h, 0.40 * h), 0.014 * fh, 0.010 * fh)  # sword


def _animal(p, x, y, d, o: _Opts) -> None:
    """A low animal statue (a duck, a tortoise): body, neck and head, standing on the ground. Sized by
    `length` (nose to tail) if stated, else by `height`, else ANIMAL_LENGTH. About 45 triangles."""
    nose_to_tail, height_per_length = 0.82, 0.55
    if o.size:
        length = o.size / nose_to_tail
    elif o.height:
        length = o.height / height_per_length
    else:
        length = ANIMAL_LENGTH / nose_to_tail
    length = min(max(length, 0.15), 10.0)
    to_world = _frame(x, y, d, 0.0, ahead=0.01 * length)
    colour = o.figure()
    bar = lambda a, b, r0, r1=None, **k: limb(p, colour, to_world, a, b, r0, r1, **k)  # noqa: E731
    k = length
    bar((0, -0.42 * k, 0.16 * k), (0, 0.12 * k, 0.23 * k), 0.16 * k, 0.25 * k, sides=6, flat=0.85)
    bar((0, 0.10 * k, 0.30 * k), (0, 0.24 * k, 0.48 * k), 0.12 * k, 0.075 * k)
    bar((0, 0.23 * k, 0.48 * k), (0, 0.40 * k, 0.44 * k), 0.085 * k, 0.04 * k)


def _statue(p, x, y, d, o: _Opts) -> None:
    if o.variant == "animal":
        return _animal(p, x, y, d, o)
    equestrian, grounded = o.variant == "equestrian", o.variant == "ground"
    total = o.tall(EQUESTRIAN_HEIGHT if equestrian else STATUE_GROUND_HEIGHT if grounded else STATUE_HEIGHT, 0.2)
    colour = o.figure()
    if grounded or total < STATUE_MIN_PLINTH_TOTAL:
        z0 = 0.0
    elif equestrian:
        z0 = _plinth(
            p, x, y, d, o, total, height=EQUESTRIAN_PEDESTAL_SHARE * total,
            length=EQUESTRIAN_PEDESTAL_LENGTH * total, width=EQUESTRIAN_PEDESTAL_WIDTH * total,
        )
    else:  # a standing figure is at most STATUE_FIGURE_MAX tall: a taller monument has a taller pedestal
        z0 = _plinth(p, x, y, d, o, total, height=max(total - STATUE_FIGURE_MAX, STATUE_PEDESTAL_SHARE * total))
    (_equestrian_figure if equestrian else _standing_figure)(p, x, y, d, z0, total - z0, colour)


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
    "bollard": _bollard,
    "block": _block,
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

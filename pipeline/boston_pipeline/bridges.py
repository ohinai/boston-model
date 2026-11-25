"""Footbridges: an arched deck with sides, underside and railings, in z-up tile metres.

A bridge tagged `bridge:structure=suspension` (the Public Garden's footbridge over the
lagoon) is drawn as one: lattice railings, granite pylons carrying globe lamps, cables
and stone abutments. Every other footbridge keeps the plain look.
"""

from __future__ import annotations

import numpy as np

from .mesh import MeshBuilder, colour_bytes

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

PLAIN, SUSPENSION = "plain", "suspension"

# The suspension look. Where things stand comes from the data (the way, its width, the lamps
# mapped at its pylons); these are the proportions of the style, chosen to read like a
# nineteenth-century park footbridge: granite pylons, light painted ironwork, white globes.
GRANITE, GRANITE_BASE = "#aaa69c", "#66686a"  # pylon above the deck; the pier below it, in the water
IRONWORK = "#9cc8c3"  # railings, cables and lamp posts: pale blue-green paint
GLOBE = "#f3f1ea"
SUSPENSION_DECK_DEPTH = 0.55  # a deeper girder than the plain bridge's slab
PYLON_PIER = 1.7  # side of the dark stone pier under a pylon
PYLON_PLINTH, PYLON_PLINTH_TOP = 1.5, 0.45  # side, and top above the deck
PYLON_SHAFT, PYLON_SHAFT_TOP = 1.1, 2.0
PYLON_CORNICE, PYLON_CORNICE_TOP = 1.4, 2.25
PYLON_CAP, PYLON_TOP = 1.0, 2.6  # the cap tapers from CAP to half of it
PYLON_STAY = 2.1  # height above the deck where the cables run into the pylon
LAMP_POST, GLOBE_RADIUS = 0.1, 0.3
LAMP_HEIGHT = 4.9  # above the deck, when the data gives no height for the lamp
ABUTMENT_LENGTH = 1.5  # stone block under each end of the deck, along the bridge
LATTICE_PITCH, LATTICE_BAR = 0.6, 0.035
LATTICE_FOOT, LATTICE_HEAD = 0.23, 1.07  # the lattice fills the space between the two rails
SUSPENSION_RAILS = ((0.16, 0.1, 0.07), (RAIL_HEIGHT, 0.08, 0.07))  # (height, across, tall) of the foot and top rails
CABLE, HANGER, HANGER_SPACING = 0.07, 0.03, 1.8
CABLE_CLEAR = 0.12  # the main cable dips to this far above the top rail at mid-span
STAY_SAG = 0.15  # how far an end stay hangs below its straight line


def bridge_style(tags: dict) -> str:
    """`bridge:structure=suspension` selects the suspension look; every other bridge is plain."""
    return SUSPENSION if tags.get("bridge:structure") == "suspension" else PLAIN


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
    style: str = PLAIN, towers=(), water_z: float = -0.6,
) -> None:
    """A deck along `line` ((n, 2), tile-relative) arching `rise` metres over its mid-span.

    `style` is PLAIN or SUSPENSION. A suspension bridge stands a pylon at each of `towers`, each
    (distance along the line, side: +1 left or -1 right, height of the lamp on it above the deck or
    None), and its piers go down to `water_z`. With no towers it has lattice railings and abutments only.
    """
    points, s, normal = _stations(np.asarray(line, dtype=np.float64))
    length = s[-1]
    top = _height(s, length, rise)
    suspension = style == SUSPENSION
    depth = SUSPENSION_DECK_DEPTH if suspension else DECK_DEPTH

    def at(offset: float, z) -> np.ndarray:
        """3D positions at a signed lateral offset (left positive) and height z."""
        return np.column_stack([points + normal * offset, np.broadcast_to(z, len(points))])

    half = width / 2
    lt, rt, lb, rb = at(half, top), at(-half, top), at(half, top - depth), at(-half, top - depth)
    deck = _strip(lt, rt, lb, rb)
    _emit(builder, deck["top"], deck_colour, feature)
    caps = np.stack([[lb[0], rb[0], rt[0], lt[0]], [lb[-1], lt[-1], rt[-1], rb[-1]]])
    _emit(builder, np.concatenate([deck["bottom"], deck["left"], deck["right"], caps]), structure_colour, feature)
    if not railings:
        return

    tangent = np.column_stack([normal[:, 1], -normal[:, 0]])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    lateral = normal / np.linalg.norm(normal, axis=1)[:, None]
    if suspension:
        frame = _Frame(points, s, normal, tangent, lateral, length, rise)
        _lattice_railings(builder, frame, half, feature)
        _abutments(builder, frame, half, water_z, feature)
        _pylons_and_cables(builder, frame, half, towers, water_z, feature)
        return
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


# --- The suspension look ---------------------------------------------------------------------


def _emit_triangles(builder: MeshBuilder, tris: np.ndarray, colour, feature: int) -> None:
    """Add flat-shaded triangles (k, 3, 3), counter-clockwise seen from outside."""
    if not len(tris):
        return
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]
    builder.add(tris.reshape(-1, 3), np.repeat(normal, 3, axis=0), np.arange(len(tris) * 3), colour=colour, feature=feature)


_UP = np.array([0.0, 0.0, 1.0])
_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)  # counter-clockwise from above


def _block(builder, colour, centre, axes, half, feature, skip=()) -> None:
    """An oriented box with right-handed unit `axes` (3, 3) and half sizes `half`; faces in `skip`
    (axis, sign) are left out."""
    quads = []
    for k in range(3):
        for sign in (1, -1):
            if (k, sign) in skip:
                continue
            i, j = (k + 1) % 3, (k + 2) % 3
            if sign < 0:
                i, j = j, i  # keeps axis i x axis j pointing outward
            face = centre + sign * half[k] * axes[k]
            quads.append(face + _CORNERS[:, :1] * half[i] * axes[i] + _CORNERS[:, 1:] * half[j] * axes[j])
    _emit(builder, np.array(quads), colour, feature)


def _beam(builder, colour, a, b, across: float, tall: float, feature: int, reference=_UP, ends: bool = False) -> None:
    """A bar from 3D point `a` to `b`, `across` wide and `tall` thick, its ends left open unless `ends`.
    `reference` fixes which way is up for the bar's cross-section (use the lateral axis for a vertical bar)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return
    along = (b - a) / length
    left = np.cross(reference, along)
    left /= np.linalg.norm(left)
    axes = np.array([along, left, np.cross(along, left)])
    _block(builder, colour, (a + b) / 2, axes, np.array([length, across, tall]) / 2, feature, () if ends else ((0, 1), (0, -1)))


def _ribbon(builder, colour, a, b, width: float, lateral, feature: int) -> None:
    """A thin flat bar from `a` to `b` in the vertical plane of a railing, drawn from both sides.
    `lateral` is the unit vector across the bridge."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return
    side = np.cross((b - a) / length, lateral) * (width / 2)
    front = np.array([a - side, b - side, b + side, a + side])
    _emit(builder, np.array([front, front[::-1]]), colour, feature)


def _frustum(builder, colour, centre, t, l, z0: float, z1: float, half0: float, half1: float, feature: int, top: bool = True) -> None:
    """A square tower from height z0 to z1 whose sides run along unit vectors `t` and `l` (t x l = up):
    half-side `half0` at the foot, `half1` at the head. The bottom is left open."""
    def ring(half, z):
        xy = np.asarray(centre)[None, :] + _CORNERS[:, :1] * half * np.asarray(t) + _CORNERS[:, 1:] * half * np.asarray(l)
        return np.column_stack([xy, np.full(4, z)])

    low, high = ring(half0, z0), ring(half1, z1)
    nxt = np.roll(np.arange(4), -1)
    quads = [np.array([low[i], low[nxt[i]], high[nxt[i]], high[i]]) for i in range(4)]
    if top:
        quads.append(high)
    _emit(builder, np.array(quads), colour, feature)


def _icosphere() -> tuple[np.ndarray, np.ndarray]:
    """A unit icosphere split once: (vertices (42, 3), counter-clockwise triangles (80, 3))."""
    phi = (1 + 5 ** 0.5) / 2
    v = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
         (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    cache: dict[tuple[int, int], int] = {}

    def middle(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in cache:
            m = verts[i] + verts[j]
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]

    faces = []
    for a, b, c in f:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts), np.array(faces)


_SPHERE = _icosphere()


def _globe(builder, colour, centre, radius: float, feature: int) -> None:
    verts, faces = _SPHERE
    _emit_triangles(builder, np.asarray(centre) + verts[faces] * radius, colour, feature)


class _Frame:
    """Where a bridge is at any distance along it: its centre, the lateral offset vector (a unit
    vector on a straight bridge), the direction of travel and the height of the deck's top."""

    def __init__(self, points, s, normal, tangent, lateral, length: float, rise: float):
        self.points, self.s, self.normal, self.tangent, self.lateral = points, s, normal, tangent, lateral
        self.length, self.rise = length, rise

    def _at(self, table, a: float) -> np.ndarray:
        return np.array([np.interp(a, self.s, table[:, 0]), np.interp(a, self.s, table[:, 1])])

    def centre(self, a: float) -> np.ndarray:
        return self._at(self.points, a)

    def offset(self, a: float) -> np.ndarray:
        """Vector from the centre line to the left edge of a unit-wide bridge (longer than 1 round a bend)."""
        return self._at(self.normal, a)

    def unit_lateral(self, a: float) -> np.ndarray:
        v = self._at(self.lateral, a)
        return v / np.linalg.norm(v)

    def unit_tangent(self, a: float) -> np.ndarray:
        v = self._at(self.tangent, a)
        return v / np.linalg.norm(v)

    def top(self, a: float) -> float:
        return float(_height(a, self.length, self.rise))

    def point(self, a: float, lateral_offset: float, z: float) -> np.ndarray:
        return np.array([*(self.centre(a) + self.offset(a) * lateral_offset), z])


def _lattice_railings(builder, frame: _Frame, half: float, feature: int) -> None:
    """Both sides: a foot rail, a top rail, posts, and a diamond lattice of thin bars between them."""
    iron = colour_bytes(IRONWORK)
    length = frame.length
    marks = np.unique(np.r_[np.arange(0.0, length, STATION), length])
    run = LATTICE_HEAD - LATTICE_FOOT  # the bars climb 45 degrees
    for side in (1, -1):
        offset = side * (half - RAIL_INSET)
        for height, across, tall in SUSPENSION_RAILS:
            # Each rail is a run of straight beams, one per station interval.
            for a, b in zip(marks[:-1], marks[1:]):
                _beam(builder, iron, frame.point(a, offset, frame.top(a) + height), frame.point(b, offset, frame.top(b) + height),
                      across, tall, feature, ends=True)
        for a in marks[::2]:
            foot = frame.point(a, offset, frame.top(a))
            _beam(builder, iron, foot, foot + np.array([0, 0, RAIL_HEIGHT]), POST_SIZE, POST_SIZE, feature,
                  reference=np.array([*frame.unit_lateral(a), 0.0]), ends=True)
        for start in np.arange(0.0, length - run + 1e-9, LATTICE_PITCH):
            end = start + run
            lat = np.array([*frame.unit_lateral(start), 0.0])
            for lo, hi in ((start, end), (end, start)):
                _ribbon(builder, iron, frame.point(lo, offset, frame.top(lo) + LATTICE_FOOT),
                        frame.point(hi, offset, frame.top(hi) + LATTICE_HEAD), LATTICE_BAR, lat, feature)


def _abutments(builder, frame: _Frame, half: float, water_z: float, feature: int) -> None:
    """A block of dark stone under each end of the deck, from the water up to the girder."""
    stone = colour_bytes(GRANITE_BASE)
    for a, toward in ((0.0, 1), (frame.length, -1)):
        mid = a + toward * ABUTMENT_LENGTH / 2
        top = min(frame.top(a), frame.top(a + toward * ABUTMENT_LENGTH)) - SUSPENSION_DECK_DEPTH
        if top <= water_z + 0.05:
            continue
        t, l = frame.unit_tangent(mid), frame.unit_lateral(mid)
        centre = np.array([*frame.centre(mid), (top + water_z) / 2])
        axes = np.array([[*t, 0.0], [*l, 0.0], [0.0, 0.0, 1.0]])
        sizes = np.array([ABUTMENT_LENGTH / 2, half + 0.1, (top - water_z) / 2])
        _block(builder, stone, centre, axes, sizes, feature, skip=((2, 1), (2, -1)))


def _pylons_and_cables(builder, frame: _Frame, half: float, towers, water_z: float, feature: int) -> None:
    """Pylons with a globe lamp on each, and the cables strung between them and down to the ends."""
    if not towers:
        return
    granite, base_stone, iron, globe = (colour_bytes(c) for c in (GRANITE, GRANITE_BASE, IRONWORK, GLOBE))
    # Pylons come in pairs, one each side at the same place along the bridge: line each pair up.
    groups: list[list[float]] = []
    for along, _, _ in sorted(towers, key=lambda tower: tower[0]):
        if groups and abs(along - np.mean(groups[-1])) < 1.5:
            groups[-1].append(along)
        else:
            groups.append([along])
    stations = [float(np.mean(group)) for group in groups]
    lamp_height = {}
    for along, side, height in towers:
        lamp_height[(int(np.argmin([abs(a - along) for a in stations])), side)] = height
    sides = sorted({side for _, side, _ in towers})
    attach = []
    for index, a in enumerate(stations):
        t, l = frame.unit_tangent(a), frame.unit_lateral(a)
        deck = frame.top(a)
        for side in sides:
            xy = frame.centre(a) + frame.offset(a) * (side * half)
            _frustum(builder, base_stone, xy, t, l, water_z, deck - LIP, PYLON_PIER / 2, PYLON_PIER / 2, feature)
            _frustum(builder, granite, xy, t, l, deck - LIP, deck + PYLON_PLINTH_TOP, PYLON_PLINTH / 2, PYLON_PLINTH / 2, feature)
            _frustum(builder, granite, xy, t, l, deck + PYLON_PLINTH_TOP, deck + PYLON_SHAFT_TOP, PYLON_SHAFT / 2, PYLON_SHAFT / 2,
                     feature, top=False)
            _frustum(builder, granite, xy, t, l, deck + PYLON_SHAFT_TOP, deck + PYLON_CORNICE_TOP, PYLON_CORNICE / 2,
                     PYLON_CORNICE / 2, feature)
            _frustum(builder, granite, xy, t, l, deck + PYLON_CORNICE_TOP, deck + PYLON_TOP, PYLON_CAP / 2, PYLON_CAP / 4, feature)
            lamp = max(lamp_height.get((index, side)) or LAMP_HEIGHT, PYLON_TOP + 4 * GLOBE_RADIUS)
            _beam(builder, iron, [*xy, deck + PYLON_TOP], [*xy, deck + lamp - GLOBE_RADIUS], LAMP_POST, LAMP_POST, feature,
                  reference=np.array([*l, 0.0]), ends=True)
            _globe(builder, globe, [*xy, deck + lamp - GLOBE_RADIUS], GLOBE_RADIUS, feature)
        attach.append(deck + PYLON_STAY)
    for side in sides:
        offset = side * (half - RAIL_INSET)

        def cable(a0, z0, a1, z1, sag, count):
            """Points along a cable from (a0, z0) to (a1, z1) that dips `sag` metres below the straight line at mid-span."""
            return np.array([
                frame.point(a0 + (a1 - a0) * u, offset, z0 + (z1 - z0) * u - 4 * sag * u * (1 - u))
                for u in np.linspace(0.0, 1.0, count + 1)
            ])

        def draw(points):
            for p, q in zip(points[:-1], points[1:]):
                _beam(builder, iron, p, q, CABLE, CABLE, feature, ends=True)

        for i in range(len(stations) - 1):
            a0, a1, z0, z1 = stations[i], stations[i + 1], attach[i], attach[i + 1]
            sag = max(0.2, (z0 + z1) / 2 - (frame.top((a0 + a1) / 2) + RAIL_HEIGHT + CABLE_CLEAR))
            draw(cable(a0, z0, a1, z1, sag, max(6, int((a1 - a0) / 1.2))))
            # Hangers from the cable down to the top rail.
            for a in np.arange(a0 + HANGER_SPACING / 2, a1, HANGER_SPACING):
                u = (a - a0) / (a1 - a0)
                z = z0 + (z1 - z0) * u - 4 * sag * u * (1 - u)
                bottom = frame.top(a) + RAIL_HEIGHT
                if z - bottom > 0.2:
                    _beam(builder, iron, frame.point(a, offset, z), frame.point(a, offset, bottom), HANGER, HANGER, feature,
                          reference=np.array([*frame.unit_lateral(a), 0.0]), ends=True)
        # From the outermost pylons the cables sweep down to the ends of the deck.
        for index, end in ((0, 0.0), (len(stations) - 1, frame.length)):
            a = stations[index]
            if abs(a - end) > 1.0:
                draw(cable(a, attach[index], end, frame.top(end) + RAIL_HEIGHT + 0.05, STAY_SAG, max(4, int(abs(a - end) / 1.2))))

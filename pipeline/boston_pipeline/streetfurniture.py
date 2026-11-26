"""Street furniture that stands beside a carriageway: traffic signals, stop signs, bus stops, post boxes and bicycle racks.

Two halves.

Shapes (`SHAPES`, drawn through `furniture.add_object`): low-poly, flat-shaded and z-up like everything in furniture.py.
Nothing here is measured; each is a plausible thing of its kind and size, standing at (x, y) and facing `direction`.

Placement (`street_objects`): OpenStreetMap says very little about where such a thing stands. A signal node is on the
road's centre line and says only "forward" or "backward" along it; a stop sign likewise; a bus stop or post box stands
somewhere on the pavement. So this half reads the road network: it puts each pole on the verge beside the carriageway, on
the right of the traffic it serves, turned to face that traffic. A node that is on no road gets no geometry, because a
guess at which way it faces would be worse than leaving it out.

Every size is in metres. The counts in the comments are for the pilot area (November 2026 data).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from .features import Object, Scene, Way
from .furniture import _frame, _Opts, box, face, limb, prism
from .mesh import colour_bytes

# --- Dimensions (metres) -------------------------------------------------------
# A vehicle signal head: a black housing with a red, an amber and a green lens, each under a visor.
HEAD_WIDTH, HEAD_DEPTH, HEAD_HEIGHT = 0.32, 0.26, 0.95
HEAD_LENS = 0.095  # circumradius of a lens
HEAD_VISOR = 0.18  # how far a visor stands out
# A post-mounted head (the usual one on a minor street): the pole is 3.7 m and the head bottoms out at 2.6 m.
SIGNAL_POST_HEIGHT, SIGNAL_POST_RANGE = 3.7, (3.2, 5.0)
SIGNAL_HEAD_ABOVE_GROUND = 2.6
# A mast arm (an arterial street, 9 m wide or more): the arm runs out over the lanes and the heads hang from it. Whatever
# the pole's stated height, the heads bottom out above 4.6 m, so nobody walking can bump one.
MAST_HEIGHT, MAST_RANGE = 6.4, (6.0, 9.0)
MAST_ARM_BELOW_TOP = 0.35
MAST_HEAD_CLEARANCE = 4.6
MAST_REACH, MAST_REACH_RANGE = 4.5, (2.5, 14.0)
# A pedestrian head: a small black box with an orange hand and a white walker, on a slim pole.
PED_HEIGHT, PED_RANGE = 3.0, (2.6, 4.0)
PED_BOX = (0.44, 0.22, 0.46)  # width, depth, height
# A stop sign: a 0.76 m octagon whose lower edge is 2.2 m above the ground.
STOP_RADIUS, STOP_CENTRE, STOP_POLE = 0.41, 2.6, 3.0
# A US Postal Service collection box.
POST_BOX_WIDTH, POST_BOX_DEPTH, POST_BOX_BODY, POST_BOX_DOME = 0.66, 0.56, 0.82, 0.30
# Bicycle racks: inverted-U hoops 0.6 m wide and 0.85 m tall, a bicycle at each side of a hoop.
HOOP_WIDTH, HOOP_HEIGHT, HOOP_CHAMFER, HOOP_SPACING, HOOP_TUBE = 0.6, 0.85, 0.17, 0.75, 0.022
HOOP_MAX = 6
BOLLARD_POST, BOLLARD_RING = 0.6, 0.17  # a post and a ring on top of it (the ring is 0.6 to 0.94 m)
# A bus stop: a sign on a pole, and where the data says there is one, a shelter open on the side toward the road.
BUS_POLE, BUS_PLATE = 2.7, (0.50, 0.62)  # plate width and height; the plate's lower edge is at 1.95 m
SHELTER_LENGTH, SHELTER_DEPTH, SHELTER_HEIGHT, SHELTER_BACK = 3.2, 1.5, 2.5, 0.25
SHELTER_DEPTHS = (SHELTER_DEPTH, 1.2, 0.9)  # tried in turn; a bus stop's shelter is drawn with the first that clears the buildings

# --- Colours -------------------------------------------------------------------
SIGNAL_POLE, SIGNAL_HOUSING = "#2f3336", "#1b1d1f"
LAMP_RED, LAMP_AMBER, LAMP_GREEN = "#cf3028", "#e3a92a", "#2ea862"
PED_HAND, PED_WALKER = "#e8772e", "#ecebe2"
STOP_RED, STOP_WHITE, STOP_BACK = "#b8231c", "#f4f2ea", "#7d8183"
POST_BOX_BLUE, POST_BOX_BROWN, POST_BOX_DOOR, POST_BOX_SLOT = "#274b8f", "#4b3626", "#bfc4ca", "#15171a"
RACK_STEEL, RACK_BLACK = "#5d6368", "#2d3033"
BUS_BLUE, BUS_WHITE = "#1f4e91", "#f1f1ec"
SHELTER_FRAME, SHELTER_GLASS, SHELTER_ROOF, SHELTER_BENCH = "#3a3f44", "#7f9ea8", "#2f3336", "#6b4a2f"


# --- Geometry helpers ----------------------------------------------------------
def _panel(parts, colour, to_world, corners, toward, both=False) -> None:
    """A flat polygon given by local (u, v, z) corners (see `furniture._frame`), facing the local direction `toward`
    (and, if `both`, the other way too)."""
    world = [to_world(*c) for c in corners]
    direction = to_world(*toward) - to_world(0.0, 0.0, 0.0)
    face(parts, colour, world, direction)
    if both:
        face(parts, colour, world, -direction)


def _ring(centre, radius, sides, rotation=0.0):
    """Corners of a regular polygon in the local (u, z) plane around `centre`, a local (u, v, z) point."""
    cu, cv, cz = centre
    angle = rotation + 2 * math.pi * np.arange(sides) / sides
    return [(cu + radius * math.cos(a), cv, cz + radius * math.sin(a)) for a in angle]


# --- Traffic signals -----------------------------------------------------------
def _head(p, W, d, u, v, z0, housing) -> None:
    """A vehicle signal head with its rear face at `v`, centred on `u`, from height z0, facing +v of the frame."""
    x, y = W(u, v + HEAD_DEPTH / 2, 0.0)[:2]
    box(p, housing, x, y, z0, z0 + HEAD_HEIGHT, HEAD_WIDTH, HEAD_DEPTH, d)
    front = v + HEAD_DEPTH
    for colour, share in ((LAMP_RED, 0.82), (LAMP_AMBER, 0.50), (LAMP_GREEN, 0.18)):
        zc = z0 + HEAD_HEIGHT * share
        _panel(p, colour_bytes(colour), W, _ring((u, front + 0.004, zc), HEAD_LENS, 6, math.pi / 6), (0, 1, 0))
        top = zc + HEAD_LENS + 0.02  # the visor over the lens, seen from below and from above
        _panel(
            p, housing, W,
            [(u - 0.12, front, top), (u + 0.12, front, top), (u + 0.12, front + HEAD_VISOR, top - 0.03), (u - 0.12, front + HEAD_VISOR, top - 0.03)],
            (0, 0, -1), both=True,
        )


def _signal(p, x, y, d, o: _Opts) -> None:
    W = _frame(x, y, d)
    pole = o.paint(SIGNAL_POLE)
    housing = colour_bytes(SIGNAL_HOUSING)
    variant = o.variant or "post"
    if variant == "pedestrian":
        top = o.tall(PED_HEIGHT, *PED_RANGE)
        prism(p, pole, x, y, 0.0, top, 0.05, 0.045, 6, top=True)
        w, deep, tall = PED_BOX
        z0 = top - 0.6
        bx, by = W(0.0, deep / 2 + 0.03, 0.0)[:2]
        box(p, housing, bx, by, z0, z0 + tall, w, deep, d)
        front = 0.03 + deep + 0.004
        for u, colour in ((-0.1, PED_HAND), (0.1, PED_WALKER)):
            c = colour_bytes(colour)
            _panel(p, c, W, [(u - 0.075, front, z0 + 0.07), (u + 0.075, front, z0 + 0.07), (u + 0.075, front, z0 + tall - 0.07), (u - 0.075, front, z0 + tall - 0.07)], (0, 1, 0))
        return
    if variant == "mast":
        top = o.tall(MAST_HEIGHT, *MAST_RANGE)
        reach = min(max(o.size if o.size is not None else MAST_REACH, MAST_REACH_RANGE[0]), MAST_REACH_RANGE[1])
        prism(p, pole, x, y, 0.0, top, 0.13, 0.08, 6, top=True)
        arm = top - MAST_ARM_BELOW_TOP
        limb(p, pole, W, (0.0, 0.0, arm), (reach, 0.0, arm + 0.12), 0.065, 0.04, sides=4)
        z0 = max(arm - 0.03 - HEAD_HEIGHT, MAST_HEAD_CLEARANCE)  # the head's top meets the arm
        for share in ((0.55,) if reach < 5.5 else (0.40, 0.82)):
            _head(p, W, d, reach * share, 0.0, z0, housing)
        return
    top = o.tall(SIGNAL_POST_HEIGHT, *SIGNAL_POST_RANGE)
    prism(p, pole, x, y, 0.0, top, 0.075, 0.065, 6, top=True)
    _head(p, W, d, 0.0, 0.06, min(SIGNAL_HEAD_ABOVE_GROUND, top - HEAD_HEIGHT - 0.1), housing)


def _stop_sign(p, x, y, d, o: _Opts) -> None:
    W = _frame(x, y, d)
    prism(p, o.paint(SIGNAL_POLE), x, y, 0.0, STOP_POLE, 0.04, 0.035, 6, top=True)
    corners = lambda r, v: _ring((0.0, v, STOP_CENTRE), r, 8, math.pi / 8)  # noqa: E731  flat top and bottom
    _panel(p, colour_bytes(STOP_BACK), W, corners(STOP_RADIUS, 0.045), (0, -1, 0))
    _panel(p, colour_bytes(STOP_WHITE), W, corners(STOP_RADIUS, 0.051), (0, 1, 0))
    _panel(p, colour_bytes(STOP_RED), W, corners(STOP_RADIUS * 0.88, 0.057), (0, 1, 0))
    bar = STOP_RADIUS * 0.58  # the white bar stands for the word
    _panel(p, colour_bytes(STOP_WHITE), W, [(-bar, 0.063, STOP_CENTRE - 0.045), (bar, 0.063, STOP_CENTRE - 0.045), (bar, 0.063, STOP_CENTRE + 0.045), (-bar, 0.063, STOP_CENTRE + 0.045)], (0, 1, 0))


# --- Post boxes, bicycle racks, bus stops --------------------------------------
def _post_box(p, x, y, d, o: _Opts) -> None:
    """A collection box: a body, a half-round top, a pull-down door and a slot, the front facing `d`."""
    W = _frame(x, y, d)
    blue = o.paint(POST_BOX_BROWN if o.variant == "ups" else POST_BOX_BLUE)
    hw, hd = POST_BOX_WIDTH / 2, POST_BOX_DEPTH / 2
    box(p, blue, x, y, 0.0, POST_BOX_BODY, POST_BOX_WIDTH, POST_BOX_DEPTH, d)
    arc = [(-hd * math.cos(a), POST_BOX_BODY + POST_BOX_DOME * math.sin(a)) for a in np.linspace(0, math.pi, 5)]  # (v, z), rear to front
    for (v0, z0), (v1, z1) in zip(arc[:-1], arc[1:]):  # the half-round top, in four strips; each faces out, away from the arc's centre
        _panel(p, blue, W, [(-hw, v0, z0), (hw, v0, z0), (hw, v1, z1), (-hw, v1, z1)], (0.0, -(z1 - z0), v1 - v0))
    for side in (-1, 1):  # the two ends of the dome
        _panel(p, blue, W, [(side * hw, v, z) for v, z in arc], (side, 0, 0))
    door, slot = colour_bytes(POST_BOX_DOOR), colour_bytes(POST_BOX_SLOT)
    front = hd + 0.004
    _panel(p, door, W, [(-0.25, front, 0.26), (0.25, front, 0.26), (0.25, front, 0.70), (-0.25, front, 0.70)], (0, 1, 0))
    _panel(p, slot, W, [(-0.20, front + 0.003, 0.58), (0.20, front + 0.003, 0.58), (0.20, front + 0.003, 0.63), (-0.20, front + 0.003, 0.63)], (0, 1, 0))


def _hoop(p, W, colour, v) -> None:
    """One inverted-U rack at `v`, in the plane of u and z."""
    half, c = HOOP_WIDTH / 2, HOOP_CHAMFER
    path = [(-half, 0.0), (-half, HOOP_HEIGHT - c), (-half + c, HOOP_HEIGHT), (half - c, HOOP_HEIGHT), (half, HOOP_HEIGHT - c), (half, 0.0)]
    for (u0, z0), (u1, z1) in zip(path[:-1], path[1:]):
        limb(p, colour, W, (u0, v, z0), (u1, v, z1), HOOP_TUBE, sides=3, closed=False)


def _bicycle_parking(p, x, y, d, o: _Opts) -> None:
    """A rack of hoops one behind another along `d` (each hoop lies across `d`), or a post with a ring on it, or for
    lockers and sheds a plain box. `size` is the capacity."""
    W = _frame(x, y, d)
    capacity = o.size if o.size else 2.0
    kind = o.variant or "stands"
    if kind in ("shed", "lockers"):
        box(p, o.paint(RACK_STEEL), x, y, 0.0, 1.3, 2.4, 0.9, d)
        return
    count = int(min(max(round(capacity / 2), 1), HOOP_MAX))
    if kind == "bollard":  # a post with a ring to lock to; one per two bicycles
        steel = o.paint(RACK_BLACK)
        for i in range(count):
            v = (i - (count - 1) / 2) * 1.2
            px, py = W(0.0, v, 0.0)[:2]
            prism(p, steel, px, py, 0.0, BOLLARD_POST, 0.045, 0.04, 6, top=True)
            ring = [(BOLLARD_RING * math.cos(a), v, BOLLARD_POST + BOLLARD_RING + BOLLARD_RING * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
            for a, b in zip(ring, ring[1:] + ring[:1]):
                limb(p, steel, W, a, b, 0.018, sides=3, closed=False)
        return
    steel = o.paint(RACK_STEEL)
    for i in range(count):
        _hoop(p, W, steel, (i - (count - 1) / 2) * HOOP_SPACING)


def _bus_stop(p, x, y, d, o: _Opts) -> None:
    """A sign on a pole at (x, y), and with `shelter` in the variant a shelter behind it, open on the side that faces `d`
    (toward the road). The sign faces to the left of `d`, toward the bus coming along a road with the stop on its right,
    or to the right with `flip` in the variant. `bench` puts a bench in the shelter; `size` is the shelter's depth. About 24
    triangles for the sign, 92 with a shelter and bench."""
    W = _frame(x, y, d)
    flags = set((o.variant or "").split("+"))
    side = 1.0 if "flip" in flags else -1.0
    blue, white = o.paint(BUS_BLUE), colour_bytes(BUS_WHITE)
    prism(p, colour_bytes(RACK_STEEL), x, y, 0.0, BUS_POLE, 0.04, 0.035, 6, top=True)
    w, h = BUS_PLATE
    z0 = BUS_POLE - 0.08 - h
    u = side * 0.052
    plate = lambda inset, a, b: [(u + side * inset, -w / 2 * a, z0 + h * b[0]), (u + side * inset, w / 2 * a, z0 + h * b[0]), (u + side * inset, w / 2 * a, z0 + h * b[1]), (u + side * inset, -w / 2 * a, z0 + h * b[1])]  # noqa: E731
    _panel(p, colour_bytes(STOP_BACK), W, plate(0.0, 1, (0, 1)), (-side, 0, 0))  # its back
    _panel(p, blue, W, plate(0.0, 1, (0, 1)), (side, 0, 0))
    _panel(p, white, W, plate(0.006, 0.8, (0.08, 0.2)), (side, 0, 0))  # a white band
    _panel(p, white, W, [(u + side * 0.006, c[0], z0 + h * c[1]) for c in ((0.0, 0.88), (0.1, 0.79), (0.0, 0.70), (-0.1, 0.79))], (side, 0, 0))
    if "shelter" not in flags:
        return
    frame, glass = colour_bytes(SHELTER_FRAME), colour_bytes(SHELTER_GLASS)
    depth = o.size if o.size else SHELTER_DEPTH  # placement narrows a shelter that would not clear the building behind it
    half, front, rear = SHELTER_LENGTH / 2, -0.25, -0.25 - depth
    cx, cy = W(0.0, (front + rear) / 2, 0.0)[:2]
    box(p, colour_bytes(SHELTER_ROOF), cx, cy, SHELTER_HEIGHT, SHELTER_HEIGHT + 0.1, SHELTER_LENGTH + 0.2, depth + 0.2, d)
    for u_post in (-half, half):
        for v_post in (front, rear):
            px, py = W(u_post, v_post, 0.0)[:2]
            prism(p, frame, px, py, 0.0, SHELTER_HEIGHT, 0.055, sides=4, rot=45.0 - d)
    sill = SHELTER_BACK
    _panel(p, glass, W, [(-half, rear, sill), (half, rear, sill), (half, rear, SHELTER_HEIGHT), (-half, rear, SHELTER_HEIGHT)], (0, 1, 0), both=True)
    for u_side in (-half, half):
        _panel(p, glass, W, [(u_side, rear, sill), (u_side, front, sill), (u_side, front, SHELTER_HEIGHT), (u_side, rear, SHELTER_HEIGHT)], (1, 0, 0), both=True)
    if "bench" in flags:
        bx, by = W(0.0, rear + 0.3, 0.0)[:2]
        box(p, colour_bytes(SHELTER_BENCH), bx, by, 0.0, 0.46, SHELTER_LENGTH * 0.6, 0.45, d)


SHAPES = {
    "signal": _signal,
    "stop_sign": _stop_sign,
    "post_box": _post_box,
    "bicycle_parking": _bicycle_parking,
    "bus_stop": _bus_stop,
}
STREET_KINDS: tuple[str, ...] = tuple(SHAPES)


# --- Placement -----------------------------------------------------------------
KERB = 0.6  # a pole stands this far outside the carriageway's edge, on the pavement
REACH = {"signal": 3.0, "stop_sign": 3.0, "bus_stop": 30.0, "post_box": 30.0, "bicycle_parking": 30.0, "lamp": 25.0}
CLEAR_OF_BUILDINGS = 0.3  # a pole is not put closer to a wall than this
SHELTER_MAPPED = 8.0  # a bus stop this close to a shelter that is mapped as a building (amenity=shelter) is not given another
MIN_ARM = 6.0  # a road shorter than this beyond a junction is a stub, not an approach
ARM_MERGE = 25.0  # approaches within this many degrees of one another are one
CROSS_STREET = (60.0, 135.0)  # an arm this many degrees from another is a cross street to it
STOP_LINE = 1.2  # a signal pole at a junction stands this far beyond the cross street's kerb
MAX_ARMS = 5
MAST_WIDTH = 9.0  # a carriageway this wide or wider gets its signals on mast arms
MIN_ROAD_WIDTH = 5.0  # a signal is not put on a narrower road than this (alleys, drives)
NO_FURNITURE = {"motorway", "motorway_link", "track"}
CROSSING_LENGTH = (4.0, 40.0)  # a signalised crossing way is a street crossing, not a stub or a ramp
CROSSING_SETBACKS, CROSSING_SIDE = (0.7, 1.5, 2.3), 1.7  # a pedestrian head stands this far beyond a crossing's end, this far to one side
POLE_SEPARATION = 1.0  # two poles of the same kind are not placed closer than this
PED_BESIDE_POLE = 1.4  # a pedestrian head this close to a vehicle signal's pole would be the same pole twice
SIGNALISED = {"traffic_signals", "pedestrian_signals"}  # `crossing` values of a crossing that has its own signals


def _bearing(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360


def _step(x: float, y: float, bearing: float, distance: float) -> tuple[float, float]:
    t = math.radians(bearing)
    return x + distance * math.sin(t), y + distance * math.cos(t)


def _apart(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _oneway(tags: dict) -> int:
    """+1 where traffic runs only along the way's direction, -1 only against it, 0 both ways."""
    value = (tags.get("oneway") or "").strip().lower()
    if value in ("yes", "true", "1") or tags.get("junction") == "roundabout":
        return 1
    return -1 if value in ("-1", "reverse") else 0


def _tangent(line, along: float, forward: bool = True) -> float:
    """Which way a line runs at a point, in degrees clockwise from north (the other way if `forward` is false)."""
    a, b = line.interpolate(max(0.0, along - 1.5)), line.interpolate(min(line.length, along + 1.5))
    bearing = _bearing(b.x - a.x, b.y - a.y)
    return bearing if forward else (bearing + 180) % 360


class _Streets:
    """Spatial lookups over the scene's carriageways and buildings."""

    def __init__(self, scene: Scene) -> None:
        self.roads = [w for w in scene.ways if w.surface == "road" and w.kind not in NO_FURNITURE and w.line.length > 0.5]
        self._lines = STRtree([w.line for w in self.roads]) if self.roads else None
        self._cover = [w.line.buffer(w.width / 2, quad_segs=2) for w in self.roads]
        self._cover_tree = STRtree(self._cover) if self.roads else None
        footprints = [b.polygon for b in scene.buildings if b.min_height < 1.0]
        self._walls = STRtree(footprints) if footprints else None
        # Bus shelters (and other shelters) that are mapped as buildings: a bus stop beside one needs no shelter drawn.
        shelters = [b.polygon for b in scene.buildings if b.tags.get("amenity") == "shelter"]
        self._shelters = STRtree(shelters) if shelters else None

    def near(self, x: float, y: float, reach: float) -> list[tuple[Way, float, float]]:
        """The roads within `reach` of a point, nearest first, each with the distance along it of the point nearest to
        (x, y) and the distance from (x, y) to it."""
        if self._lines is None:
            return []
        point = Point(x, y)
        found = [(self.roads[i], self.roads[i].line.distance(point)) for i in self._lines.query(point, predicate="dwithin", distance=reach)]
        found.sort(key=lambda item: (item[1], item[0].id))
        return [(way, way.line.project(point), distance) for way, distance in found]

    def on_road(self, x: float, y: float) -> bool:
        if self._cover_tree is None:
            return False
        point = Point(x, y)
        return any(self._cover[i].contains(point) for i in self._cover_tree.query(point))

    def in_building(self, x: float, y: float, margin: float = 0.0) -> bool:
        if self._walls is None:
            return False
        point = Point(x, y)
        hits = self._walls.query(point, predicate="dwithin", distance=margin) if margin else self._walls.query(point, predicate="within")
        return len(hits) > 0

    def shelter_near(self, x: float, y: float) -> bool:
        """Whether a shelter mapped as a building stands within SHELTER_MAPPED of the point."""
        return self._shelters is not None and len(self._shelters.query(Point(x, y), predicate="dwithin", distance=SHELTER_MAPPED)) > 0

    def clear(self, corners: list[tuple[float, float]]) -> bool:
        """Whether a rectangle (its four corners in order) stands clear of every building and off the carriageways."""
        rect = Polygon(corners)
        if self._walls is not None and len(self._walls.query(rect, predicate="intersects")):
            return False
        ring = corners + [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(corners, corners[1:] + corners[:1])]
        return not any(self.on_road(*c) for c in ring)

    def stand(self, x: float, y: float, travel: float, half_width: float) -> tuple[float, float] | None:
        """Where a pole stands on the verge to the right of traffic heading `travel`, along a road whose centre line
        passes through (x, y): beside the carriageway, off every road, clear of walls; None where there is no such spot."""
        right = (travel + 90) % 360
        edge = half_width + KERB
        for offset in (edge, edge + 0.5, edge + 1.0, edge + 1.6, edge + 2.4, edge - 0.35):
            px, py = _step(x, y, right, offset)
            if not self.on_road(px, py) and not self.in_building(px, py, CLEAR_OF_BUILDINGS):
                return px, py
        return None


class _Arm:
    """A road running away from a point: which way, how far, whether traffic comes in along it, and whether it lies
    ahead of the point in its way's direction."""

    def __init__(self, way: Way, bearing: float, length: float, inbound: bool, ahead: bool) -> None:
        self.way, self.bearing, self.length, self.inbound, self.ahead = way, bearing, length, inbound, ahead


def _arms(streets: _Streets, x: float, y: float) -> list[_Arm]:
    arms = []
    for way, along, _ in streets.near(x, y, REACH["signal"]):
        if way.width < MIN_ROAD_WIDTH:
            continue
        one = _oneway(way.tags)
        for ahead in (True, False):
            room = way.line.length - along if ahead else along
            if room < MIN_ARM:
                continue
            # Traffic comes in along the arm ahead of the point if it runs against the way's direction, along the one behind if with it.
            arms.append(_Arm(way, _tangent(way.line, along, ahead), room, one <= 0 if ahead else one >= 0, ahead))
    return arms


def _merge(arms: list[_Arm]) -> list[_Arm]:
    """One arm for each way out of the junction: of two within ARM_MERGE degrees, the wider road's."""
    kept: list[_Arm] = []
    for arm in sorted(arms, key=lambda a: (-a.way.width, a.way.id, a.bearing)):
        if all(_apart(arm.bearing, other.bearing) >= ARM_MERGE for other in kept):
            kept.append(arm)
    return kept


def _signals(scene: Scene, streets: _Streets) -> list[Object]:
    """A pole and head for each way traffic comes in to a signal node. A node with `traffic_signals:direction` is on one
    road and serves one way along it; a node without it serves every approach to the junction it stands at."""
    placed: list[Object] = []
    for point in scene.signals:
        if point.tags.get("traffic_signals") == "ramp_meter":
            continue
        arms = _arms(streets, point.x, point.y)
        tag = (point.tags.get("traffic_signals:direction") or "").strip().lower()
        if tag in ("forward", "backward") and arms:
            # The head faces traffic going `tag` along the nearest road, which comes from the arm behind (forward) or ahead (backward).
            arms = [a for a in arms if a.way is arms[0].way and a.ahead == (tag == "backward")]
            stopped = False
        else:
            arms = _merge([a for a in arms if a.inbound])[:MAX_ARMS]
            stopped = True
        for arm in arms:
            way = arm.way
            cross = [a.way.width for a in arms if CROSS_STREET[0] <= _apart(a.bearing, arm.bearing) <= CROSS_STREET[1]]
            back = min(0.5 * max(cross) + STOP_LINE, max(arm.length - 3.0, 0.0)) if stopped and cross else 0.0
            bx, by = _step(point.x, point.y, arm.bearing, back)
            spot = streets.stand(bx, by, (arm.bearing + 180) % 360, way.width / 2)
            if spot is None:
                continue
            mast = way.width >= MAST_WIDTH
            # A mast arm runs out to the middle of a two-way road and across all of a one-way one.
            reach = math.hypot(spot[0] - bx, spot[1] - by) + (way.width / 2 if _oneway(way.tags) else 0.0)
            placed.append(Object(point.id, spot[0], spot[1], "signal", None, arm.bearing, None, reach if mast else None, None, None, "mast" if mast else "post", point.tags))
    return placed


def _crossing_heads(scene: Scene, streets: _Streets) -> list[Object]:
    """Pedestrian signal heads: two for each footway that crosses a street and is tagged as having signals, one at each
    end, looking across to the other end."""
    placed: list[Object] = []
    for way in scene.ways:
        tags = way.tags
        if way.surface != "path" or tags.get("footway") != "crossing":
            continue
        if tags.get("crossing") not in SIGNALISED and tags.get("crossing:signals") not in ("yes", "traffic_signals"):
            continue
        a, b = way.line.coords[0], way.line.coords[-1]
        if not CROSSING_LENGTH[0] <= way.line.length <= CROSSING_LENGTH[1] or math.hypot(b[0] - a[0], b[1] - a[1]) < 0.75 * CROSSING_LENGTH[0]:
            continue
        for start, end in ((a, b), (b, a)):
            across = _bearing(end[0] - start[0], end[1] - start[1])
            spots = (
                _step(*_step(start[0], start[1], (across + 180) % 360, back), (across + 90 * side) % 360, CROSSING_SIDE)
                for back in CROSSING_SETBACKS for side in (1, -1)
            )
            spot = next((s for s in spots if not streets.on_road(*s) and not streets.in_building(*s, CLEAR_OF_BUILDINGS)), None)
            if spot is not None:
                placed.append(Object(way.id, spot[0], spot[1], "signal", None, across, None, None, None, None, "pedestrian", tags))
    return placed


def _spread(things: list[Object], separation: float = POLE_SEPARATION) -> list[Object]:
    """The things in order, leaving out any that stands within `separation` of one already kept."""
    kept: list[Object] = []
    cells: dict[tuple[int, int], list[Object]] = {}
    for thing in things:
        cell = (int(thing.x // 4), int(thing.y // 4))
        if any(
            math.hypot(other.x - thing.x, other.y - thing.y) < separation
            for i in (-1, 0, 1) for j in (-1, 0, 1) for other in cells.get((cell[0] + i, cell[1] + j), ())
        ):
            continue
        cells.setdefault(cell, []).append(thing)
        kept.append(thing)
    return kept


def _stand_stop_sign(thing: Object, streets: _Streets) -> Object | None:
    """A stop sign on the right of the traffic it is for, facing it. None if the node is on no road, or the data does not
    say which way along the road the sign applies (`direction=forward` or `backward`)."""
    tag = (thing.tags.get("direction") or "").strip().lower()
    found = streets.near(thing.x, thing.y, REACH["stop_sign"])
    if not found or tag not in ("forward", "backward"):
        return None
    way, along, _ = found[0]
    facing = _tangent(way.line, along, tag == "backward")  # toward the traffic: back along forward traffic, ahead of backward
    foot = way.line.interpolate(along)
    spot = streets.stand(foot.x, foot.y, (facing + 180) % 360, way.width / 2)
    return None if spot is None else replace(thing, x=spot[0], y=spot[1], direction=facing)


def _pavement_thing(thing: Object, streets: _Streets) -> Object | None:
    """A bus stop, post box or rack: where the data puts it, turned by the road beside it."""
    if streets.in_building(thing.x, thing.y):
        return None
    found = streets.near(thing.x, thing.y, REACH[thing.kind])
    if not found:
        return None if thing.kind == "bus_stop" else thing  # a sign with no road to face has no meaning; the others stand anyway
    way, along, distance = found[0]
    foot = way.line.interpolate(along)
    road = _tangent(way.line, along, True)
    toward = _bearing(foot.x - thing.x, foot.y - thing.y) if distance > 0.3 else (road + 90) % 360  # toward the road
    if thing.direction is not None and thing.kind != "bus_stop":
        return thing
    if thing.kind == "post_box":
        return replace(thing, direction=(toward + 180) % 360)  # the door faces the pavement, away from the road
    if thing.kind == "bicycle_parking":
        # A long rack stands along the kerb, in a row; one hoop or two stand across it, so a bicycle lies along the kerb.
        return replace(thing, direction=road if (thing.size or 2.0) >= 6 else toward)
    # A bus stop: the buses come along with the stop on their right; on a one-way street, the way it runs.
    one = _oneway(way.tags)
    flip = False
    if one:
        coming_from = (road + 180) % 360 if one == 1 else road  # the sign faces the oncoming bus
        flip = _apart((coming_from - toward) % 360, 90) < 45  # a sign faces to the left of `toward` unless it is to the right
    variant, depth = thing.variant or "", None
    if "shelter" in variant:
        # The shelter is already there if the map has it as a building; else the deepest that clears the walls, else the sign alone.
        depth = None if streets.shelter_near(thing.x, thing.y) else _shelter_depth(thing, toward, streets)
        if depth is None:
            variant = ""
    return replace(thing, direction=toward, variant=variant + ("+flip" if flip else ""), size=depth)


def _shelter_depth(thing: Object, toward: float, streets: _Streets) -> float | None:
    """The depth of the deepest shelter that fits behind a bus stop sign (SHELTER_DEPTHS), or None."""
    half = SHELTER_LENGTH / 2 + 0.1
    for depth in SHELTER_DEPTHS:
        rear = 0.25 + depth + 0.1  # the roof overhangs by 0.1 m
        corners = [_step(*_step(thing.x, thing.y, toward, -v), (toward + 90) % 360, u) for u, v in ((-half, 0.15), (half, 0.15), (half, rear), (-half, rear))]
        if streets.clear(corners):
            return depth
    return None


def _lamp_arm(thing: Object, streets: _Streets) -> Object:
    """A lamp on a bracket arm that has no `direction`: its arm reaches out over the nearest road."""
    found = streets.near(thing.x, thing.y, REACH["lamp"])
    if not found:
        return thing
    way, along, distance = found[0]
    foot = way.line.interpolate(along)
    return replace(thing, direction=_bearing(foot.x - thing.x, foot.y - thing.y)) if distance > 0.5 else thing


def street_objects(scene: Scene) -> list[Object]:
    """Every object in the scene, with the street furniture worked out from the road network (see the module docstring):
    signals made from the scene's signal nodes and signalised crossings; stop signs, bus stops, post boxes and racks put
    where they stand and turned; lamps on arms turned toward the road. Everything else is passed through."""
    streets = _Streets(scene)
    out: list[Object] = []
    for thing in scene.objects:
        if thing.kind == "stop_sign":
            thing = _stand_stop_sign(thing, streets)
        elif thing.kind in ("bus_stop", "post_box", "bicycle_parking"):
            thing = _pavement_thing(thing, streets)
        elif thing.kind == "lamp" and thing.variant == "arm" and thing.direction is None:
            thing = _lamp_arm(thing, streets)
        if thing is not None:
            out.append(thing)
    poles = _spread([*_spread(_signals(scene, streets)), *_spread(_crossing_heads(scene, streets))], PED_BESIDE_POLE)
    return out + poles

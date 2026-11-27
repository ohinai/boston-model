"""Pitches, courts and diamonds that read as what they are: painted lines, goals, hoops, nets, bases and backstops.

A `leisure=pitch` area is a loose polygon with a `sport`. Everything here is interpretation of that one fact: where the
polygon is and what sport it is decide which standard layout is fitted to it, centred, turned to the polygon's long side and
scaled down (never up) to fit. The standard sizes are those of the sport, as a tree's size follows from its species; none is
measured from a photograph (only where a baseball diamond stands is, see below). Where a layout does not fit (a polygon much smaller than the sport needs, a shape that is not
the sport's, an area that is covered by something else) nothing is drawn: a bare pitch is better than a wrong one.

What is drawn, and the rule for each (lines are white, 8 to 11 cm wide, 5 cm above the ground in the `markings` layer, which a walker
passes through; dirt is a flat patch 4 cm above the ground in the same layer):

* tennis: the court with singles sidelines, service lines and centre marks; the net hangs between two posts (white tape over
  bands of dark mesh in the `markings` layer, which a walker passes through, and two slim solid posts, kind `net`). One court
  per polygon: a polygon big enough for several courts is left bare.
* basketball: the court (the 84 by 50 foot court of a school or park, or the 94 by 50 foot one where the polygon is that long)
  with keys, free-throw circles, three-point arcs and the centre circle, and a hoop (kind `hoop`) at each end. A half court (a
  polygon under 21 m long, or `hoops=1`) has one hoop, at an end the data does not name.
* soccer: touchlines, halfway line, centre circle, penalty and goal areas and arcs, and a goal (kind `goal`) at each end.
* American football: sidelines, goal lines, a line every five yards and a goalpost (kind `goalpost`) on each end line.
* volleyball and beach volleyball: the court and its net.
* baseball and softball: the foul lines out to the fence, home plate, batter's boxes, bases, pitcher's rubber and mound, the
  infield dirt (an arc round a grass infield, with base paths) and a backstop (kind `backstop`). The hard part is where home
  plate is and which way the diamond faces: the outline of the outfield does not say (see `fan_seed`), so a diamond is drawn
  only where the infield is mapped as a polygon of its own inside the pitch, or where the project tags `home_plate` and
  `direction` say (fitted to the infield dirt in the NAIP photograph and set through fixes/sports/). Any other is left bare.

A pitch with no `sport`, a `multi` pitch, one of more than three sports, and one that is covered, indoors or mostly under a
building or a road are left bare. A pitch listed with several sports gets the first one it fits. Where pitches overlap the
smaller one shows (the project's rule for grass areas, see tiles._by_size) and lines are drawn only where the pitch shows.

Project tags (not OpenStreetMap's), set through files in fixes/sports/ from the NAIP photograph (USGS, 7 July 2023): on a baseball
or softball pitch `home_plate` ("latitude,longitude"), `direction` (degrees clockwise from north, from home plate toward second
base), `base_distance` (60 ft, 70 ft or 90 ft; 60 ft if absent) and `infield=dirt` (no grass inside the base paths).
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from . import world
from .features import Object, Scene, parse_direction, parse_length, parse_number, polygons_of
from .furniture import _frame, annulus, cuboid, face, limb, prism
from .mesh import MeshBuilder, add_flat, colour_bytes

# --- Appearance ----------------------------------------------------------------------------------------------------------
LINE_COLOUR = "#f2f2ee"
DIRT = "#a58c68"  # an infield of natural dirt, as NAIP (7 July 2023) shows it: pale tan to brown
DIRT_TURF = "#a85a3e"  # the red clay colour of an artificial-turf field's infield
NET_DARK, NET_TAPE = "#2d3033", "#eeeeea"
# Heights above the ground. The viewer's depth buffer cannot tell two surfaces apart that are closer than about 4 cm at 400 m (see
# markings.MARK_LIFT), so dirt lies as high as road paint does, and the lines on it a centimetre above that: far enough not to
# flicker, too little to see as a step (and a walker passes straight through the whole `markings` layer anyway).
DIRT_LIFT = 0.04
LINE_LIFT = 0.05
BASE_LIFT = 0.06  # bases and plates stand a hair above the lines

# --- Standard sizes (metres) ---------------------------------------------------------------------------------------------
YARD = 0.9144
FOOT = 0.3048
TENNIS = (23.77, 10.97)  # doubles court; singles sidelines 1.37 m inside, service line 6.40 m from the net
TENNIS_NET_POST, TENNIS_NET_HEIGHT, TENNIS_POST_HEIGHT = 0.914, 0.914, 1.07
BASKET_HS = (25.6, 15.24)  # 84 by 50 feet; the 94-foot court is 28.65 m
BASKET_LONG = 28.65
VOLLEY = (18.0, 9.0)
BEACH = (16.0, 8.0)
SOCCER = (105.0, 68.0)
FOOTBALL = (120 * YARD, 160 * FOOT)  # 120 yards with the end zones, 53 1/3 yards (160 feet) wide
BASES = {90: 90 * FOOT, 70: 70 * FOOT, 60: 60 * FOOT}  # base path length by name
PITCH_DISTANCE = {90: 60.5 * FOOT, 70: 53 * FOOT, 60: 46 * FOOT}  # rubber to the back point of home plate
SOFTBALL_PITCH = 43 * FOOT

# What is drawn for a pitch that a polygon is smaller than: at least this share of the standard size, else nothing.
MIN_SCALE = 0.7
MAX_SPORTS = 3  # a pitch with more sports than this is a multi-purpose field
NOT_DRAWN = {"multi"}
MIN_VISIBLE = 0.5  # a pitch less than this share showing (the rest under a building, road, path or water) gets nothing
ONE_COURT_MAX_AREA = 900.0  # a tennis polygon bigger than this holds several courts: left bare
DIAMOND_MARGIN = 8.0  # a diamond is painted this far beyond its polygon (see plan)

LINE_WIDTH = {"tennis": 0.08, "basketball": 0.08, "volleyball": 0.08, "soccer": 0.11, "american_football": 0.10, "baseball": 0.10}


# --- Geometry helpers ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """A court's own axes: its centre, and the unit vector along its long side. `a` runs along it, `b` across (to its left)."""

    x: float
    y: float
    ux: float
    uy: float

    def pt(self, a: float, b: float) -> tuple[float, float]:
        return (self.x + a * self.ux - b * self.uy, self.y + a * self.uy + b * self.ux)

    def heading(self, da: float, db: float) -> float:
        """Degrees clockwise from north of the direction da along the court and db across it."""
        dx, dy = da * self.ux - db * self.uy, da * self.uy + db * self.ux
        return math.degrees(math.atan2(dx, dy)) % 360.0


def _rect(f: Frame, a0: float, a1: float, b0: float, b1: float) -> Polygon:
    return Polygon([f.pt(a0, b0), f.pt(a1, b0), f.pt(a1, b1), f.pt(a0, b1)])


def _outline(f: Frame, a0: float, a1: float, b0: float, b1: float, w: float) -> Polygon:
    """The outline of a rectangle as a band of width w inside it."""
    return _rect(f, a0, a1, b0, b1).difference(_rect(f, a0 + w, a1 - w, b0 + w, b1 - w))


def _seg(f: Frame, a0: float, b0: float, a1: float, b1: float, w: float) -> Polygon:
    return LineString([f.pt(a0, b0), f.pt(a1, b1)]).buffer(w / 2, cap_style="flat", join_style="mitre")


def _ring(x: float, y: float, r: float, w: float, quad: int = 8) -> Polygon:
    return Point(x, y).buffer(r + w / 2, quad_segs=quad).difference(Point(x, y).buffer(max(r - w / 2, 0.0), quad_segs=quad))


def _disc(x: float, y: float, r: float, quad: int = 4) -> Polygon:
    return Point(x, y).buffer(r, quad_segs=quad)


def _unit(v) -> np.ndarray:
    n = math.hypot(v[0], v[1])
    return np.asarray(v, dtype=float) / n if n > 1e-9 else np.zeros(2)


# --- What a layout is ----------------------------------------------------------------------------------------------------
@dataclass
class Layout:
    """What one pitch gets: paint (polygon, colour, lift above the ground), nets (x0, y0, x1, y1, height) and things (kind, x, y,
    heading, size, height, variant) that become objects."""

    marks: list = field(default_factory=list)
    nets: list = field(default_factory=list)
    things: list = field(default_factory=list)
    sport: str = ""

    def line(self, geom, colour: str = LINE_COLOUR, lift: float = LINE_LIFT) -> None:
        if geom is not None and not geom.is_empty:
            self.marks.append((geom, colour, lift))


@dataclass
class Pitch:
    """A pitch polygon and what is known about it."""

    ident: str
    index: int  # in scene.areas["green"]
    polygon: Polygon
    tags: dict
    cover: str
    centre: tuple[float, float]
    ux: float
    uy: float
    length: float  # of the minimum rotated rectangle
    width: float
    visible: object = None  # the part of the pitch that shows (not under a smaller pitch, a road, a path, water, a building)
    children: list = field(default_factory=list)  # polygons of the same game inside it (the infield of a baseball field)

    def frame(self, length: float, width: float) -> Frame:
        """The frame of a court `length` by `width`, turned to the pitch's long side and moved a little (up to 4 m) to cover as
        much of the polygon as it can."""
        best, best_area = (0.0, 0.0), -1.0
        vx, vy = -self.uy, self.ux
        for da in (0.0, -1.5, 1.5, -3.0, 3.0, -4.5, 4.5):
            for db in (0.0, -1.5, 1.5, -3.0, 3.0):
                f = Frame(self.centre[0] + da * self.ux + db * vx, self.centre[1] + da * self.uy + db * vy, self.ux, self.uy)
                area = self.polygon.intersection(_rect(f, -length / 2, length / 2, -width / 2, width / 2)).area
                if area > best_area + 1e-6:
                    best, best_area = (da, db), area
        da, db = best
        return Frame(self.centre[0] + da * self.ux + db * vx, self.centre[1] + da * self.uy + db * vy, self.ux, self.uy)


# --- Tennis and volleyball ------------------------------------------------------------------------------------------------
def _tennis(p: Pitch) -> Layout | None:
    length, width = TENNIS
    if p.polygon.area > ONE_COURT_MAX_AREA:
        return None  # several courts in one polygon: the layout of each is not known
    k = min(1.0, p.length / length, p.width / width)
    if k < MIN_SCALE:
        return None
    f, w = p.frame(length * k, width * k), LINE_WIDTH["tennis"]
    hl, hw = length * k / 2, width * k / 2
    out = Layout(sport="tennis")
    out.line(unary_union([
        _outline(f, -hl, hl, -hw, hw, w),
        _seg(f, -hl, 4.115 * k, hl, 4.115 * k, w), _seg(f, -hl, -4.115 * k, hl, -4.115 * k, w),
        _seg(f, 6.40 * k, -4.115 * k, 6.40 * k, 4.115 * k, w), _seg(f, -6.40 * k, -4.115 * k, -6.40 * k, 4.115 * k, w),
        _seg(f, -6.40 * k, 0, 6.40 * k, 0, w),
        _seg(f, hl - 0.2, 0, hl, 0, w), _seg(f, -hl, 0, -hl + 0.2, 0, w),
    ]))
    reach = hw + TENNIS_NET_POST
    x0, y0 = f.pt(0, -reach)
    x1, y1 = f.pt(0, reach)
    out.nets.append((x0, y0, x1, y1, TENNIS_NET_HEIGHT))
    mx, my = f.pt(0, 0)
    out.things.append(("net", mx, my, f.heading(1, 0), 2 * reach, TENNIS_POST_HEIGHT, "tennis"))
    return out


def _volleyball(p: Pitch, beach: bool = False) -> Layout | None:
    length, width = BEACH if (beach or (p.cover == "dirt" and p.length < 22)) else VOLLEY
    beach = (length, width) == BEACH
    k = min(1.0, p.length / length, p.width / width)
    if k < MIN_SCALE or p.polygon.area > 600:
        return None
    f, w = p.frame(length * k, width * k), LINE_WIDTH["volleyball"]
    hl, hw = length * k / 2, width * k / 2
    out = Layout(sport="beachvolleyball" if beach else "volleyball")
    parts = [_outline(f, -hl, hl, -hw, hw, w), _seg(f, 0, -hw, 0, hw, w)]
    if not beach:
        parts += [_seg(f, 3.0 * k, -hw, 3.0 * k, hw, w), _seg(f, -3.0 * k, -hw, -3.0 * k, hw, w)]
    out.line(unary_union(parts))
    reach = hw + (0.7 if beach else 0.5)
    x0, y0 = f.pt(0, -reach)
    x1, y1 = f.pt(0, reach)
    out.nets.append((x0, y0, x1, y1, 2.43))
    mx, my = f.pt(0, 0)
    out.things.append(("net", mx, my, f.heading(1, 0), 2 * reach, 2.55, "volleyball"))
    return out


# --- Basketball -----------------------------------------------------------------------------------------------------------
LANE_WIDTH, LANE_LENGTH, FT_RADIUS, HOOP_FROM_BASELINE = 3.66, 5.79, 1.83, 1.575
BOARD_FROM_BASELINE = 1.22  # the backboard's face stands this far inside the baseline


def _basket_end(f: Frame, sign: int, length: float, width: float, s: float, w: float, college: bool) -> list:
    """The key, free-throw circle and three-point arc at the end of the court at a = sign * length / 2 (s scales the court down)."""
    def at(t: float, b: float) -> tuple[float, float]:  # t metres in from the baseline
        return f.pt(sign * (length / 2 - t), b)

    key_w, key_l = LANE_WIDTH * s, LANE_LENGTH * s
    parts = [
        _seg(f, sign * length / 2, -key_w / 2, sign * (length / 2 - key_l), -key_w / 2, w),
        _seg(f, sign * length / 2, key_w / 2, sign * (length / 2 - key_l), key_w / 2, w),
        _seg(f, sign * (length / 2 - key_l), -key_w / 2 - w / 2, sign * (length / 2 - key_l), key_w / 2 + w / 2, w),
    ]
    cx, cy = at(key_l, 0.0)
    parts.append(_ring(cx, cy, FT_RADIUS * s, w, 6))
    hx, hy = at(HOOP_FROM_BASELINE * s, 0.0)
    radius = (6.75 if college else 6.02) * s
    margin = width / 2 - 0.9 * s  # the three-point line stays this far from a sideline, and is straight where the circle would pass it
    inside = _rect(f, -length / 2, length / 2, -margin, margin)
    arc = _ring(hx, hy, radius, w, 8).intersection(inside)
    parts.append(arc)
    if radius > margin:
        t_meet = HOOP_FROM_BASELINE * s + math.sqrt(radius**2 - margin**2)
        for b in (-margin, margin):
            parts.append(_seg(f, sign * length / 2, b, sign * (length / 2 - t_meet), b, w))
    return parts


def _basketball(p: Pitch) -> Layout | None:
    hoops = p.tags.get("hoops")
    half = p.length < 21.0 or hoops == "1"
    out = Layout(sport="basketball")
    w = LINE_WIDTH["basketball"]
    if not half:
        if p.polygon.area > 700 or p.length > 36.0 or p.width > 18.5:
            return None  # more than one court: the layout is not known (a court with a few metres of run-off at each end is one)
        length = BASKET_LONG if p.length >= 27.0 else BASKET_HS[0]  # a polygon that long holds the 94-foot court, else the 84-foot one
        length = min(length, p.length + 1.0)
        width = min(BASKET_HS[1], p.width + 0.5)
        s = min(1.0, length / BASKET_HS[0], width / BASKET_HS[1])
        if s < MIN_SCALE:
            return None
        f = p.frame(length, width)
        parts = [_outline(f, -length / 2, length / 2, -width / 2, width / 2, w), _seg(f, 0, -width / 2, 0, width / 2, w)]
        cx, cy = f.pt(0, 0)
        parts.append(_ring(cx, cy, FT_RADIUS * s, w, 6))
        for sign in (-1, 1):
            parts += _basket_end(f, sign, length, width, s, w, length > 27.0)
        out.line(unary_union(parts))
        for sign in (-1, 1):
            x, y = f.pt(sign * (length / 2 - BOARD_FROM_BASELINE), 0.0)
            out.things.append(("hoop", x, y, f.heading(-sign, 0), None, None, None))
        return out
    # A half court: the hoop's end is not in the data, so it is the end a hash of the id picks.
    if p.polygon.area < 70 or p.width < 8.0:
        return None
    length, width = min(p.length, 15.0), min(p.width, BASKET_HS[1])
    s = min(1.0, length / 14.0, width / BASKET_HS[1])
    if s < 0.55:
        return None
    f = p.frame(length, width)
    sign = 1 if zlib.crc32(p.ident.encode()) & 1 else -1
    parts = [_outline(f, -length / 2, length / 2, -width / 2, width / 2, w)]
    parts += _basket_end(f, sign, length, width, s, w, False)
    out.line(unary_union(parts))
    x, y = f.pt(sign * (length / 2 - BOARD_FROM_BASELINE), 0.0)
    out.things.append(("hoop", x, y, f.heading(-sign, 0), None, None, None))
    return out


# --- Soccer ---------------------------------------------------------------------------------------------------------------
def _soccer(p: Pitch) -> Layout | None:
    length, width = min(p.length, SOCCER[0]), min(p.width, SOCCER[1])
    k = min(1.0, length / SOCCER[0], width / SOCCER[1])
    if p.polygon.area < 600 or length < 36 or width < 20:
        return None
    if length / width < 1.2 and p.polygon.area > 2500:
        return None  # a square field is not a soccer pitch
    f, w = p.frame(length, width), LINE_WIDTH["soccer"]
    hl, hw = length / 2, width / 2
    # The boxes and circles keep their regulation size unless the pitch is well under it (a 65 m wide field has a full centre circle).
    kk = 1.0 if k >= 0.88 else max(k, 0.45)
    parts = [_outline(f, -hl, hl, -hw, hw, w), _seg(f, 0, -hw, 0, hw, w)]
    cx, cy = f.pt(0, 0)
    parts += [_ring(cx, cy, 9.15 * kk, w, 8), _disc(cx, cy, 0.2)]
    pen_d, pen_w, goal_d, goal_w, mark, arc = 16.5 * kk, 20.16 * kk, 5.5 * kk, 9.16 * kk, 11.0 * kk, 9.15 * kk
    pen_w, goal_w = min(pen_w, hw - 0.3), min(goal_w, hw - 0.3)
    for sign in (-1, 1):
        parts.append(_outline(f, *sorted((sign * hl, sign * (hl - pen_d))), -pen_w, pen_w, w))
        parts.append(_outline(f, *sorted((sign * hl, sign * (hl - goal_d))), -goal_w, goal_w, w))
        mx, my = f.pt(sign * (hl - mark), 0)
        parts.append(_disc(mx, my, 0.18))
        outside = _rect(f, *sorted((sign * (hl - pen_d), sign * (hl - pen_d - arc - 1))), -hw, hw)
        parts.append(_ring(mx, my, arc, w, 8).intersection(outside))
    out = Layout(sport="soccer")
    out.line(unary_union(parts))
    goal_width = 7.32 if length >= 90 else 5.5 if length >= 55 else 3.66
    for sign in (-1, 1):
        x, y = f.pt(sign * hl, 0)
        out.things.append(("goal", x, y, f.heading(-sign, 0), goal_width, 2.44 if goal_width > 6 else 2.0 if goal_width > 4 else 1.83, "soccer"))
    return out


# --- American football ----------------------------------------------------------------------------------------------------
def _football(p: Pitch) -> Layout | None:
    length, width = FOOTBALL
    k = min(1.0, p.length / length, p.width / width)
    if k < 0.88 or p.length / p.width < 1.6:
        return None
    k = min(k, 1.0)
    f, w = p.frame(length * k, width * k), LINE_WIDTH["american_football"]
    hl, hw = length * k / 2, width * k / 2
    goal = 100 * YARD * k / 2
    parts = [_outline(f, -hl, hl, -hw, hw, w), _seg(f, goal, -hw, goal, hw, w), _seg(f, -goal, -hw, -goal, hw, w)]
    for n in range(1, 20):  # a line every five yards between the goal lines
        a = -goal + n * 5 * YARD * k
        parts.append(_seg(f, a, -hw, a, hw, w if n != 10 else w * 1.4))
    out = Layout(sport="american_football")
    out.line(unary_union(parts))
    for sign in (-1, 1):
        x, y = f.pt(sign * hl, 0)
        out.things.append(("goalpost", x, y, f.heading(-sign, 0), 5.64, 9.1, None))
    return out


# --- Baseball and softball ------------------------------------------------------------------------------------------------
@dataclass
class Diamond:
    home: np.ndarray  # the back point of home plate
    axis: np.ndarray  # unit vector from home plate toward second base
    base: float  # length of a base path
    pitch: float  # rubber to the back point of home plate
    reach: float  # distance from home plate to the fence
    how: str  # "tags" or "infield": how it was worked out
    skin: bool = False  # the whole infield is dirt, with no grass inside the base paths


def _straight_runs(polygon: Polygon, min_length: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """The straight sides of a polygon: (start, end, length) of each run of edges that turn less than 9 degrees from one
    another, longest first, those at least `min_length` long."""
    ring = polygon.simplify(0.35).exterior
    pts = np.asarray(ring.coords)[:-1]
    n = len(pts)
    if n < 4:
        return []
    delta = np.roll(pts, -1, axis=0) - pts
    length = np.hypot(delta[:, 0], delta[:, 1])
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    turn = np.abs((heading - np.roll(heading, 1) + np.pi) % (2 * np.pi) - np.pi)
    first = int(np.argmax(turn))
    runs: list[list] = []
    for k in range(n):
        i = (first + k) % n
        if length[i] < 1e-6:
            continue
        if runs:
            chord = runs[-1][1] - runs[-1][0]
            current = math.atan2(chord[1], chord[0])
            if abs((heading[i] - current + math.pi) % (2 * math.pi) - math.pi) < math.radians(9.0):
                runs[-1][1] = pts[(i + 1) % n]
                runs[-1][2] += length[i]
                continue
        runs.append([pts[i], pts[(i + 1) % n], float(length[i])])
    found = [(a, b, float(math.hypot(*(b - a)))) for a, b, _ in runs]
    return sorted((r for r in found if r[2] >= min_length), key=lambda r: -r[2])


def fan_seed(polygon: Polygon) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """A first guess at home plate and the diamond's axis from the shape of the outline alone: (home, axis, reach, foul), or None.

    OpenStreetMap maps the outfield as a polygon shaped like a fan: two straight sides along the foul lines and the fence a curve
    across the far side. The two longest straight sides that meet at a right angle (80 to 100 degrees) near their ends, with the
    polygon lying between them and about the size of a quarter circle, give the axis (their bisector) and a home plate (where
    their lines cross); `reach` is how far the farthest vertex is from it, and `foul` the length of the shorter side.

    That is only a seed, and not good enough to draw from. The sides are the fences, which stand a few metres outside the foul
    lines, so where they cross is behind home plate: 0 m behind it at Amory Playground, 11 m at Briggs Field and 14 m at
    Boston Common's Little League field, measured against USGS NAIP (7 July 2023). Nor does the fence say how big the diamond
    is (Cusick Field's is a 60-foot diamond in a 113 m fan). So `fit_diamond` draws a diamond only from a measurement of the
    infield: an infield polygon in the map, or home plate and direction in tags (set through a file in fixes/, fitted to the
    infield dirt in the photograph).
    """
    far = np.asarray(polygon.exterior.coords)
    runs = _straight_runs(polygon, max(14.0, 0.2 * math.sqrt(polygon.area)))
    best, best_score = None, 0.0
    for i, (a0, a1, la) in enumerate(runs):
        for b0, b1, lb in runs[i + 1 :]:
            ua, ub = _unit(a1 - a0), _unit(b1 - b0)
            if abs(float(ua @ ub)) > 0.24:  # not at about a right angle
                continue
            try:
                s, _t = np.linalg.solve(np.array([ua, -ub]).T, b0 - a0)
            except np.linalg.LinAlgError:
                continue
            home = a0 + ua * s
            sa = np.array([(a0 - home) @ ua, (a1 - home) @ ua])
            sb = np.array([(b0 - home) @ ub, (b1 - home) @ ub])
            if sa.min() * sa.max() < -4.0 or sb.min() * sb.max() < -4.0:  # home plate is inside one of the sides
                continue
            if min(abs(sa).min(), abs(sb).min()) > 14.0:  # the sides do not come near their meeting point
                continue
            va = ua * (1.0 if sa.mean() > 0 else -1.0)
            vb = ub * (1.0 if sb.mean() > 0 else -1.0)
            axis = _unit(va + vb)
            if float(va @ vb) < -0.25 or float(va @ axis) < 0.55:
                continue
            reach = float(np.hypot(*(far - home).T).max())
            if reach < 45.0 or not 0.7 <= polygon.area / (math.pi * reach * reach / 4) <= 1.35:
                continue
            if float(_unit(np.array(polygon.centroid.coords[0]) - home) @ axis) < math.cos(math.radians(22)):
                continue
            left = np.array([-axis[1], axis[0]])
            wedge = Polygon([home - axis * 3.0, home + (axis + left) * (reach + 5), home + axis * (reach + 6), home + (axis - left) * (reach + 5)])
            if polygon.intersection(wedge).area / polygon.area < 0.9:  # the polygon must lie between the two foul lines
                continue
            if min(la, lb) > best_score:
                best, best_score = (home, axis, reach, min(la, lb)), min(la, lb)
    return best


def _base_override(tags: dict) -> int | None:
    """The `base_distance` tag as the nearest standard base path (90, 70 or 60 feet), or None."""
    value = parse_length(tags.get("base_distance"))
    if value is None or not 15.0 <= value <= 30.0:
        return None
    return min(BASES, key=lambda name: abs(BASES[name] - value))


def _tagged_home(tags: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(home plate, axis) from the project tags `home_plate` ("latitude,longitude") and `direction` (degrees clockwise from north
    that home plate looks toward second base), or None unless both are there and make sense."""
    where = str(tags.get("home_plate", "")).split(",")
    bearing = parse_direction(tags.get("direction"))
    if len(where) != 2 or bearing is None:
        return None
    lat, lon = parse_number(where[0]), parse_number(where[1])
    if lat is None or lon is None:
        return None
    x, y = (float(v) for v in world.to_local(lon, lat))
    t = math.radians(bearing)
    return np.array([x, y]), np.array([math.sin(t), math.cos(t)])


def fit_diamond(p: Pitch, sport: str) -> Diamond | None:
    """Where home plate is and which way a baseball or softball diamond faces, or None where that is not known.

    Two sources, the first better than the second:

    * the pitch carries the project tags `home_plate` ("latitude,longitude"), `direction` (degrees clockwise from north, from home
      plate toward second base), and optionally `base_distance` (60 ft, 70 ft or 90 ft: 60 if absent) and `infield=dirt` (no
      grass inside the base paths). Set through a file in fixes/ from a fit to the infield dirt in USGS NAIP (7 July 2023);
    * the infield is mapped as a polygon of its own inside the pitch (the grass square inside the base paths, as at Fenway Park):
      its sides run along the base paths, and home plate is the corner at the end of the diagonal that points away from where
      the outfield reaches farthest.

    A diamond from neither is not drawn (see `fan_seed` for why the outline of the outfield is not enough).
    """
    polygon = p.polygon
    name = _base_override(p.tags)
    tagged = _tagged_home(p.tags)
    if tagged is not None:
        home, axis = tagged
        base = name or 60
        reach = float(np.hypot(*(np.asarray(polygon.exterior.coords) - home).T).max())
        return Diamond(home, axis, BASES[base], _pitch_distance(base, sport), reach, "tags", p.tags.get("infield") == "dirt")
    for child in p.children:  # an infield polygon of its own
        rect = child.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        pts = np.column_stack([xs, ys])[:4]
        e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
        l1, l2 = float(math.hypot(*e1)), float(math.hypot(*e2))
        if not (16.5 <= min(l1, l2) and max(l1, l2) <= 31.0 and min(l1, l2) / max(l1, l2) >= 0.85 and child.area / rect.area >= 0.8):
            continue
        side, centre = (l1 + l2) / 2, pts.mean(axis=0)
        u1, u2 = _unit(e1), _unit(e2)
        best, best_reach = None, -1.0
        far = np.asarray(polygon.exterior.coords)
        for d in (_unit(u1 + u2), _unit(u1 - u2), _unit(-u1 + u2), _unit(-u1 - u2)):
            reach = float(((far - centre) @ d).max())
            if reach > best_reach:
                best, best_reach = d, reach
        base = name or min(BASES, key=lambda n: abs(BASES[n] - (side + 0.8)))
        home = centre - best * BASES[base] / math.sqrt(2)
        return Diamond(home, best, BASES[base], _pitch_distance(base, sport), best_reach + BASES[base] / math.sqrt(2), "infield")
    return None


def _pitch_distance(base: int, sport: str) -> float:
    return SOFTBALL_PITCH if sport == "softball" and base == 60 else PITCH_DISTANCE[base]


# How far the infield dirt reaches from the pitching rubber, by base path (feet) and whether the grass inside the base paths is
# dirt too. Measured on USGS NAIP (7 July 2023): 29 m (95 feet) at Amory Playground (90-foot bases, grass infield); 15.9 m at Boston
# Common's Little League field (60-foot bases, grass infield); 18.7 m at Cusick Field (60-foot, all dirt); about 22.5 m at the
# Soldiers Field softball diamond (70-foot, all dirt). The other two are in between or guesses (70-foot with grass, 90-foot all dirt).
SKIN_RADIUS = {(60, False): 15.9, (60, True): 18.7, (70, False): 22.0, (70, True): 22.5, (90, False): 29.0, (90, True): 31.8}


def diamond_skin(at, b: float, pitch: float, full: bool = False) -> Polygon:
    """The infield dirt of a diamond with bases `b` metres apart and the rubber `pitch` metres from home plate: an arc round the
    rubber (SKIN_RADIUS), clipped to fair ground, with the grass inside the base paths cut out (unless `full`), a circle of dirt
    at home plate and at the mound. `at(x, y)` is a point x to the right of and y ahead of home plate, as `_diamond` defines it."""
    s2, reach = math.sqrt(2.0), 80.0
    mound = at(0.0, pitch)
    wedge = Polygon([at(0, -0.7), at(-reach, reach - 0.7), at(reach, reach - 0.7)])
    radius = SKIN_RADIUS[(min(BASES, key=lambda n: abs(BASES[n] - b)), bool(full))]
    arc = _disc(mound[0], mound[1], radius, 12).intersection(wedge)
    if not full:
        grass = Polygon([at(0, 0), at(b / s2, b / s2), at(0, b * s2), at(-b / s2, b / s2)]).buffer(-0.058 * b, join_style="round", quad_segs=4)
        arc = arc.difference(grass)
    return unary_union([arc, _disc(*at(0, 0.5), 0.135 * b, 6), _disc(mound[0], mound[1], max(1.8, 0.1 * b), 6)])


def _diamond(p: Pitch, sport: str) -> Layout | None:
    d = fit_diamond(p, sport)
    if d is None:
        return None
    out = Layout(sport=sport)
    ey = d.axis
    ex = np.array([ey[1], -ey[0]])  # to the right of someone at home plate facing second base: toward first base
    h = d.home

    def at(x: float, y: float) -> tuple[float, float]:
        return tuple(h + x * ex + y * ey)

    b = d.base
    s2 = math.sqrt(2.0)
    lift_dirt, w = DIRT_LIFT, LINE_WIDTH["baseball"]
    dirt = DIRT_TURF if p.cover == "turf" else DIRT
    whole_dirt = p.cover == "dirt"
    if not whole_dirt:
        out.line(diamond_skin(at, b, d.pitch, d.skin), dirt, lift_dirt)
    # Foul lines from home plate to the fence, and the batter's boxes.
    run = min(d.reach + 1.0, 130.0)
    parts = [LineString([at(0, 0), at(-run / s2, run / s2)]).buffer(w / 2, cap_style="flat"), LineString([at(0, 0), at(run / s2, run / s2)]).buffer(w / 2, cap_style="flat")]
    for side in (-1, 1):
        x0, x1 = (0.216 + 0.15, 0.216 + 0.15 + 1.22)
        lo, hi = (x0, x1) if side > 0 else (-x1, -x0)
        box_ = Polygon([at(lo, 0.22 - 0.915), at(hi, 0.22 - 0.915), at(hi, 0.22 + 0.915), at(lo, 0.22 + 0.915)])
        parts.append(box_.difference(box_.buffer(-0.07, join_style="mitre")))
    out.line(unary_union(parts))
    # Home plate, the bases and the rubber.
    plate = Polygon([at(0, 0), at(0.216, 0.216), at(0.216, 0.432), at(-0.216, 0.432), at(-0.216, 0.216)])
    pieces = [plate]
    d1, d2 = _unit(ex + ey), _unit(ey - ex)  # along the base paths: a base is a 38 cm square with its sides on them
    for bx, by in ((b / s2 - 0.19, b / s2), (0.0, b * s2), (0.19 - b / s2, b / s2)):  # first, second, third, inside the foul line
        cx, cy = at(bx, by)
        pieces.append(Polygon([(cx + 0.19 * (sa * d1[0] + sb * d2[0]), cy + 0.19 * (sa * d1[1] + sb * d2[1])) for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]))
    pieces.append(Polygon([at(-0.305, d.pitch), at(0.305, d.pitch), at(0.305, d.pitch + 0.15), at(-0.305, d.pitch + 0.15)]))  # the rubber
    out.line(unary_union(pieces), LINE_COLOUR, BASE_LIFT)
    # The backstop stands behind home plate, facing the field.
    back = min(max(0.33 * b + 2.0, 7.0), 11.0)
    out.things.append(("backstop", float(h[0]), float(h[1]), math.degrees(math.atan2(ey[0], ey[1])) % 360.0, back, 4.6, None))
    return out


# --- What a pitch is ------------------------------------------------------------------------------------------------------
FITTERS = {
    "tennis": _tennis,
    "basketball": _basketball,
    "soccer": _soccer,
    "american_football": _football,
    "volleyball": _volleyball,
    "beachvolleyball": lambda p: _volleyball(p, beach=True),
    "baseball": lambda p: _diamond(p, "baseball"),
    "softball": lambda p: _diamond(p, "softball"),
}
DIAMOND_SPORTS = {"baseball", "softball"}


def pitch_sports(tags: dict) -> list[str]:
    """The sports a pitch is for, in the order the tag lists them; [] for a pitch left bare (no sport, `multi`, too many)."""
    sports = [s.strip().lower() for s in str(tags.get("sport", "")).split(";") if s.strip()]
    if not sports or sports[0] in NOT_DRAWN or len(sports) > MAX_SPORTS:
        return []
    return sports


def _drawable(tags: dict) -> bool:
    if tags.get("leisure") != "pitch":
        return False
    if tags.get("covered") == "yes" or tags.get("location") in ("rooftop", "indoor", "underground") or tags.get("description") == "bullpen":
        return False
    return True


@dataclass
class SportsPlan:
    """Everything drawn for the pitches of the district: paint in world metres, nets, and the objects (goals, hoops, ...)."""

    polygons: list = field(default_factory=list)
    colours: list = field(default_factory=list)
    lifts: list = field(default_factory=list)
    nets: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    tree: STRtree | None = None
    counts: Counter = field(default_factory=Counter)  # pitches drawn, by sport
    skipped: Counter = field(default_factory=Counter)  # pitches left bare, by reason
    drawn: dict = field(default_factory=dict)  # pitch id (and its position in the list) -> sport

    def finish(self) -> SportsPlan:
        self.tree = STRtree(self.polygons) if self.polygons else None
        return self


class _Blockers:
    """What stands on a pitch's ground and so hides its lines: buildings, carriageways, paths, water, planted beds."""

    def __init__(self, scene: Scene):
        self.polygons: list[Polygon] = []
        for name in ("road", "path", "water", "bed"):
            self.polygons += [a.polygon for a in scene.areas[name]]
        self.polygons += [b.polygon for b in scene.buildings if b.min_height < 1.0]
        self.tree = STRtree(self.polygons) if self.polygons else None
        self.ways = [w for w in scene.ways if not w.bridge and w.kind not in ("steps",) and w.surface in ("road", "path", "rail")]
        self.way_tree = STRtree([w.line for w in self.ways]) if self.ways else None
        self.barriers = [b for b in scene.barriers if b.kind in ("fence", "wall", "retaining_wall")]
        self.barrier_tree = STRtree([b.line for b in self.barriers]) if self.barriers else None

    def over(self, area) -> list:
        """The polygons of everything on the ground in `area` that hides it."""
        found = [self.polygons[int(i)] for i in self.tree.query(area, predicate="intersects")] if self.tree is not None else []
        if self.way_tree is not None:
            for i in self.way_tree.query(area, predicate="intersects"):
                way = self.ways[int(i)]
                found.append(way.line.buffer(way.width / 2, quad_segs=3))
        return found

    def fences_near(self, line) -> list:
        if self.barrier_tree is None:
            return []
        return [self.barriers[int(i)] for i in self.barrier_tree.query(line, predicate="dwithin", distance=2.5)]


def _rect_of(polygon: Polygon) -> tuple[tuple[float, float], float, float, float, float]:
    rect = polygon.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    p = list(zip(xs, ys))
    e1, e2 = (p[1][0] - p[0][0], p[1][1] - p[0][1]), (p[2][0] - p[1][0], p[2][1] - p[1][1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    (ux, uy), length, width = ((e1[0] / l1, e1[1] / l1), l1, l2) if l1 >= l2 else ((e2[0] / l2, e2[1] / l2), l2, l1)
    c = rect.centroid
    return (c.x, c.y), ux, uy, length, width


def plan(scene: Scene) -> SportsPlan:
    """The lines, nets and equipment of every pitch in the scene (see the module docstring). Computed once per scene."""
    global _last
    if not hasattr(scene, "areas"):  # not a scene (extras.py's own tests hand its modules a bare object): nothing to draw
        return SportsPlan()
    if _last is not None and _last[0] is scene:
        return _last[1]
    out = SportsPlan()
    greens = scene.areas["green"]
    pitches = [(i, a) for i, a in enumerate(greens) if _drawable(a.tags) and a.polygon.area >= 60.0 and pitch_sports(a.tags)]
    if not pitches:
        _last = (scene, out.finish())
        return out
    tree = STRtree([a.polygon for a in greens])
    blockers = _Blockers(scene)
    seen: Counter = Counter()
    for index, area in pitches:
        polygon = area.polygon
        sports = pitch_sports(area.tags)
        wanted = {s for s in sports if s in DIAMOND_SPORTS}
        children, smaller = [], []
        for j in tree.query(polygon, predicate="intersects"):
            j = int(j)
            other = greens[j]
            if j == index or (other.polygon.area, j) >= (polygon.area, index):
                continue
            if wanted and other.polygon.within(polygon.buffer(1.0)) and wanted & set(pitch_sports(other.tags)):
                children.append(other.polygon)  # the infield of this diamond, drawn on with it
                continue
            smaller.append(other.polygon)
        blocked = unary_union(smaller + blockers.over(polygon.buffer(DIAMOND_MARGIN)))
        visible = polygon.difference(blocked) if not blocked.is_empty else polygon
        if not visible.is_valid:
            visible = shapely.make_valid(visible)
        if visible.area < MIN_VISIBLE * polygon.area:
            out.skipped["mostly covered"] += 1
            continue
        centre, ux, uy, length, width = _rect_of(polygon)
        pitch = Pitch(area.id, index, polygon, area.tags, area.cover, centre, ux, uy, length, width, visible, children)
        layout = None
        for sport in sports:
            fit = FITTERS.get(sport)
            layout = fit(pitch) if fit else None
            if layout is not None:
                break
        if layout is None:
            out.skipped[f"no layout ({sports[0]})"] += 1
            continue
        seen[area.id] += 1
        ident = area.id if seen[area.id] == 1 else f"{area.id}#{seen[area.id]}"
        # Paint is cut to what shows. A diamond's home plate and backstop may stand just outside the polygon (the outline stops at the
        # fence round them), so a diamond is painted on a margin round the polygon too, still out of the way of everything that hides it.
        show = visible
        if layout.sport in DIAMOND_SPORTS:
            show = unary_union([polygon, *children]).buffer(DIAMOND_MARGIN - 1.0).difference(blocked)
        for geom, colour, lift in layout.marks:
            for piece in polygons_of(geom.intersection(show)):
                if piece.area >= 0.002:
                    out.polygons.append(piece)
                    out.colours.append(colour)
                    out.lifts.append(lift)
        for net in layout.nets:
            if show.contains(Point((net[0] + net[2]) / 2, (net[1] + net[3]) / 2)):
                out.nets.append(net)
        for number, (kind, x, y, heading, size, height, variant) in enumerate(layout.things):
            if kind == "backstop":
                line = _backstop_line(x, y, heading, size)
                if blockers.fences_near(line):  # the map already has a fence round home plate
                    out.skipped["backstop (a fence is mapped there)"] += 1
                    continue
            out.objects.append(
                Object(f"{ident}:{kind}{number}", x, y, kind, None, heading, height, size, None, None, variant, {"sport": layout.sport, "pitch": ident})
            )
        out.counts[layout.sport] += 1
        out.drawn[ident] = layout.sport
    out.finish()
    _last = (scene, out)
    return out


_last: tuple | None = None


def objects(scene: Scene) -> list[Object]:
    """Goals, hoops, net posts, goalposts and backstops (see extras.py)."""
    return plan(scene).objects


# --- Painting the plan into a tile ---------------------------------------------------------------------------------------
_colours: dict[str, np.ndarray] = {}


def _colour(hex_code: str) -> np.ndarray:
    if hex_code not in _colours:
        _colours[hex_code] = colour_bytes(hex_code)
    return _colours[hex_code]


def add_marks(builder: MeshBuilder, sports: SportsPlan | None, tile_box: Polygon, origin) -> None:
    """The lines, dirt and nets of the pitches in a tile, into the `markings` layer (which has colours): cut to the tile; a net
    is drawn whole, in the tile of its middle."""
    if sports is None:
        return
    if sports.tree is not None:
        for i in sorted(int(i) for i in sports.tree.query(tile_box, predicate="intersects")):
            piece = sports.polygons[i].intersection(tile_box)
            colour = _colour(sports.colours[i])
            for polygon in polygons_of(piece):
                if polygon.area >= 0.002:
                    add_flat(builder, polygon, sports.lifts[i], origin, colour)
    x0, y0, x1, y1 = tile_box.bounds
    for net in sports.nets:
        mx, my = (net[0] + net[2]) / 2, (net[1] + net[3]) / 2
        if x0 <= mx < x1 and y0 <= my < y1:
            _add_net(builder, net, origin)


NET_BAND, NET_PITCH = 0.05, 0.12  # a net is drawn as bands of this height every NET_PITCH metres (a mesh that can be seen through)


def _add_net(builder: MeshBuilder, net: tuple, origin) -> None:
    """A net: a white tape along its top over bands of dark mesh, every one seen from both sides (4 triangles each; 36 for a tennis
    net). Bands, not a panel, so that a net is not a wall: the court beyond it shows between them."""
    a = np.array(net[:2]) - origin
    b = np.array(net[2:4]) - origin
    top = net[4]
    along = np.array([b[0] - a[0], b[1] - a[1], 0.0])
    front = np.cross(along, [0.0, 0.0, 1.0])
    front /= max(np.linalg.norm(front), 1e-9)
    bands = [(top - 0.07, top, NET_TAPE)]
    z = top - 0.07 - 0.05
    while z > 0.05:
        bands.append((max(z - NET_BAND, 0.05), z, NET_DARK))
        z -= NET_PITCH
    for z0, z1, colour in bands:
        corners = np.array([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)])
        builder.add(corners, front, [0, 1, 2, 0, 2, 3], colour=_colour(colour))
        builder.add(corners, -front, [0, 2, 1, 0, 3, 2], colour=_colour(colour))


# --- The objects ----------------------------------------------------------------------------------------------------------
POST_IRON, BOARD, RIM, ROPE = "#33383b", "#e3e1da", "#d4561f", "#e9e9e4"
GOAL_WHITE, GOAL_YELLOW = "#ecece8", "#e6c31d"
BACKSTOP_IRON = "#3b4144"


def _panel(p, colour, W, pts) -> None:
    """A flat panel seen from both sides; `pts` are (u, v, z) points of a convex polygon."""
    pts = np.asarray([W(*q) for q in pts])
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    normal /= np.linalg.norm(normal)
    face(p, colour, pts, normal)
    face(p, colour, pts, -normal)


def _hoop(p, x, y, d, o) -> None:
    """A basketball goal: a pole behind a steel arm, a backboard 1.83 by 1.07 m with its face 1.22 m inside the baseline, a rim
    3.05 m up with a short net. (x, y) is the foot of the backboard on the court, `d` the way it faces. About 80 triangles."""
    W = _frame(x, y, d)
    steel, board, rim, rope = colour_bytes(POST_IRON), colour_bytes(BOARD), colour_bytes(RIM), colour_bytes(ROPE)
    cuboid(p, steel, W, -0.08, 0.08, -1.75, -1.59, 0.0, 3.35, "tfrlR")
    cuboid(p, steel, W, -0.05, 0.05, -1.59, -0.05, 3.2, 3.32, "tblR")
    cuboid(p, board, W, -0.915, 0.915, -0.05, 0.0, 2.9, 3.97, "tbfrlR")
    cuboid(p, steel, W, -0.05, 0.05, 0.0, 0.17, 3.02, 3.07, "tbflR")
    rx, ry, _ = W(0.0, 0.39, 0.0)
    prism(p, rim, rx, ry, 3.02, 3.08, 0.23, sides=6)
    annulus(p, rim, rx, ry, 3.08, 0.23, 0.2, sides=6)
    prism(p, rope, rx, ry, 2.7, 3.02, 0.2, 0.12, sides=6)


def _strips(p, colour, W, corners, count: int, thick: float = 0.035) -> None:
    """A panel of net drawn as `count` thin bands from one edge of the quadrilateral `corners` (four (u, v, z) points in order) to the
    opposite one, seen from both sides: it reads as netting and leaves the view through it open. 4 triangles a band."""
    a, b, c, d = (np.asarray(q, dtype=float) for q in corners)  # a-b the bottom edge, d-c the top
    for k in range(count):
        t0, t1 = (k + 0.5) / count - thick, (k + 0.5) / count + thick
        lo0, lo1 = a + (d - a) * t0, b + (c - b) * t0
        hi0, hi1 = a + (d - a) * t1, b + (c - b) * t1
        _panel(p, colour, W, [lo0, lo1, hi1, hi0])


def _goal(p, x, y, d, o) -> None:
    """A soccer goal on the goal line at (x, y) facing `d` (into the pitch): two posts, a crossbar, slanted back supports and a net
    of bands (back, top and sides; see `_strips`). `size` is its width and `height` the crossbar's height. About 150 triangles."""
    width = o.size if o.size is not None else 7.32
    height = o.height if o.height is not None else 2.44
    depth = min(2.0, 0.28 * width + 0.3)
    W = _frame(x, y, d)
    frame, net = colour_bytes(GOAL_WHITE), colour_bytes(ROPE, 0.85)
    r = 0.06
    for side in (-1, 1):
        u = side * width / 2
        cuboid(p, frame, W, u - r, u + r, -r, r, 0.0, height, "tfrlR")
        limb(p, frame, W, (u, 0.0, height - 0.06), (u, -depth, 0.0), 0.035, 0.03, sides=4, closed=False)
    cuboid(p, frame, W, -width / 2 - r, width / 2 + r, -r, r, height - 2 * r, height, "tbfr")
    low = 0.35 * height
    left, right = -width / 2, width / 2
    _strips(p, net, W, [(left, -depth, low), (right, -depth, low), (right, 0.0, height), (left, 0.0, height)], 4)  # the roof of the net
    _strips(p, net, W, [(left, -depth, 0.0), (right, -depth, 0.0), (right, -depth, low), (left, -depth, low)], 5)  # the back
    for u in (left, right):
        _strips(p, net, W, [(u, -depth, 0.0), (u, 0.0, 0.0), (u, 0.0, height), (u, -depth, low)], 4)  # the sides


def _goalpost(p, x, y, d, o) -> None:
    """A football goalpost on the end line at (x, y) facing `d` (into the field): a post, a crossbar 3.05 m up and two uprights.
    `size` is the space between the uprights. About 40 triangles."""
    spread = o.size if o.size is not None else 5.64
    top = o.height if o.height is not None else 9.1
    W = _frame(x, y, d)
    yellow = colour_bytes(GOAL_YELLOW)
    cuboid(p, yellow, W, -0.08, 0.08, -0.45, -0.29, 0.0, 3.0, "tfrlR")
    cuboid(p, yellow, W, -0.05, 0.05, -0.45, 0.0, 2.95, 3.05, "tblR")
    cuboid(p, yellow, W, -spread / 2 - 0.04, spread / 2 + 0.04, -0.04, 0.04, 3.05, 3.17, "tbfr")
    for side in (-1, 1):
        u = side * spread / 2
        cuboid(p, yellow, W, u - 0.04, u + 0.04, -0.04, 0.04, 3.17, top, "tfrlR")


def _net_posts(p, x, y, d, o) -> None:
    """The two posts of a tennis or volleyball net: `size` apart, `height` tall, the net running across `d`. (The net itself is
    drawn in the `markings` layer, which a walker passes through.) About 20 triangles."""
    spread = o.size if o.size is not None else 12.8
    height = o.height if o.height is not None else TENNIS_POST_HEIGHT
    W = _frame(x, y, d)
    steel = colour_bytes(POST_IRON)
    for side in (-1, 1):
        u = side * spread / 2
        cuboid(p, steel, W, u - 0.05, u + 0.05, -0.05, 0.05, 0.0, height, "tfrlR")


ARC = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0)  # where posts stand round the backstop, degrees from straight behind home plate
WING = 4.5  # the wings of a backstop run on this far past the arc


def _backstop_points(back: float) -> list[tuple[float, float]]:
    """The posts of a backstop as (u to the right of home plate, v ahead of it) facing the field: a curve of radius `back` behind
    home plate, and at each end a wing that runs on along the tangent, forward and outward."""
    pts = [(back * math.sin(math.radians(a)), -back * math.cos(math.radians(a))) for a in ARC]
    t0, t1 = math.radians(ARC[0]), math.radians(ARC[-1])
    left = (pts[0][0] - WING * math.cos(t0), pts[0][1] - WING * math.sin(t0))
    right = (pts[-1][0] + WING * math.cos(t1), pts[-1][1] + WING * math.sin(t1))
    return [left, *pts, right]


def _backstop_line(x: float, y: float, heading: float, back: float) -> LineString:
    """The backstop's line on the ground, for a field facing `heading` from home plate (x, y)."""
    t = math.radians(heading)
    fx, fy, rx, ry = math.sin(t), math.cos(t), math.cos(t), -math.sin(t)
    return LineString([(x + u * rx + v * fx, y + u * ry + v * fy) for u, v in _backstop_points(back)])


def _backstop(p, x, y, d, o) -> None:
    """A backstop behind home plate (x, y): a chain-link screen on a curve of posts of radius `size` round the plate, with wings
    that run on forward and outward; drawn as posts and rails like the fences are. `height` tall. About 250 triangles."""
    back = o.size if o.size is not None else 8.0
    height = o.height if o.height is not None else 4.6
    W = _frame(x, y, d)
    iron = colour_bytes(BACKSTOP_IRON)
    pts = _backstop_points(back)
    for u, v in pts:
        cuboid(p, iron, W, u - 0.06, u + 0.06, v - 0.06, v + 0.06, 0.0, height, "tfrlR")
    rails = (0.15, 0.3 * height, 0.55 * height, 0.8 * height, height - 0.1)
    for (u0, v0), (u1, v1) in zip(pts[:-1], pts[1:]):
        for z in rails:
            _panel(p, iron, W, [(u0, v0, z), (u1, v1, z), (u1, v1, z + 0.05), (u0, v0, z + 0.05)])


SHAPES = {"hoop": _hoop, "goal": _goal, "goalpost": _goalpost, "net": _net_posts, "backstop": _backstop}

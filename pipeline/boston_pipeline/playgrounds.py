"""Playgrounds, allotment gardens, bandstands and gazebos, and the small things in parks that nothing drew: barbecue
grills, picnic tables, chess tables and toilet huts.

This is interpretation, not survey. OpenStreetMap says "a playground is here, about this big" (121 of them in the `boston`
area, November 2026) and maps one piece of equipment, a slide. A walker in a playground should find the kind of things a
playground has, so this module puts them there, the way trees.py gives a tree its size from its species:

- **Generic equipment** (`objects`): from a playground's outline, a seeded layout of swings, slides, climbing towers,
  seesaws, spring riders and sandpits, kept 1.5 m inside the outline and 2 m apart, and out of any building, path, road,
  water, planted bed, fence, tree trunk or piece of furniture. The number of pieces follows the area (`equipment_count`).
  Each is a point object whose tags say `generic=yes`: it is a guess at what stands there, never a survey, and a
  `playground=*` point in the data (the one slide on City Hall Plaza) is the real thing, drawn where it is, and stops a
  generic piece of its kind appearing in the same playground. Which pieces, where and how they are turned are chosen by a
  hash of the playground's id, so a build gives the same playground every time.
- **Allotment plots** (`add_plots`): a `landuse=allotments` area, and a `leisure=garden` that is a community garden
  (`garden:type=community`, or "community garden" in its name), is laid out as a grid of plots about 4 x 6 m with
  paths between them, each a bed of soil with rows of planting, and a shed, a bean tepee or a bean tent in some. Drawn in
  the `beds` layer. The plot size and the planting are standard for allotments, not measured.
- **Bandstands and gazebos** (`add_open_structure`): a `leisure=bandstand`, or an `amenity=shelter` with `shelter_type` gazebo
  or pavilion, mapped as a building of 8 to 150 m2 with a roughly round footprint is an open roof on posts over a raised
  floor, instead of a box with the default height. A rectangular one is drawn by pavilions.py as a roof on posts.
- **Small things** (`interpret`, `SHAPES`): `amenity=bbq`, `leisure=picnic_table` or `amenity=picnic_table`, an `amenity=table`
  with `sport=chess`, `amenity=toilets` (a hut, unless a building covers the point), `amenity=shelter` with
  `shelter_type=gazebo` as a point, and `playground=*` points.

Every size is in metres, standard for the kind of thing: a swing frame about 2.4 m high, a slide deck about 1.4 m, a
picnic table 1.8 m long. Nothing is measured from a photograph. Shapes are low-poly and flat-shaded like furniture.py's.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import polylabel, unary_union
from shapely.strtree import STRtree

from .features import Object, Scene, parse_length, parse_number, polygons_of
from .furniture import _frame, _Opts, _Parts, _square_rot, box, cuboid, disc, face, limb, prism, pyramid
from .mesh import colour_bytes
from .pavilions import add_pavilion

# --- Colours -------------------------------------------------------------------
# A playground has one theme, so its pieces match: (platforms and slides, panels and seats, roofs). Bright enough to read
# from the ground and from the air, not neon.
THEMES = (
    ("#c8372d", "#e3b52f", "#2f6cb3"),  # red, yellow, blue
    ("#2f6cb3", "#e3b52f", "#c8372d"),  # blue, yellow, red
    ("#2e9a57", "#e3b52f", "#2f6cb3"),  # green, yellow, blue
    ("#e07a26", "#2f6cb3", "#2e9a57"),  # orange, blue, green
)
STEEL, STEEL_DARK, CHAIN = "#9ba1a6", "#565c61", "#7d8286"
RUBBER, TIMBER, SAND = "#2b2b2d", "#8b6a47", "#d8c595"
STAINLESS = "#c9ced1"
GRILL_IRON, GRILL_GRATE = "#2e3033", "#6e7174"
TABLE_TIMBER, CHESS_DARK, CHESS_LIGHT = "#7a5c3c", "#2e2f31", "#bdb8a8"
HUT_WALL, HUT_ROOF, HUT_DOOR = "#a39a8c", "#4a4d50", "#4a5a52"

# --- Dimensions (metres) -------------------------------------------------------
SWING_BEAM, SWING_SEAT_SPACING, SWING_SEAT_HEIGHT = 2.4, 0.9, 0.46  # beam height; seat to seat; a seat's top
SLIDE_DECK, SLIDE_DECK_HEIGHT, SLIDE_DECK_LIMITS = 1.0, 1.4, (0.7, 3.2)  # a square deck on posts; a typical deck height
SLIDE_RUN_PER_METRE, SLIDE_OUT, SLIDE_WIDTH = 1.7, 0.45, 0.55  # chute length over each metre of drop; the flat run-out; chute width
TOWER_SIDE, TOWER_PLATFORM, TOWER_EAVE, TOWER_APEX = 1.7, 1.2, 2.6, 3.3  # a climbing tower: platform side and height, roof eave and apex
SEESAW_LENGTH, SEESAW_LOW, SEESAW_HIGH = 3.0, 0.28, 0.82  # plank length; the heights of its two ends
SANDPIT_SIDE, SANDPIT_LIMITS = 2.4, (1.2, 6.0)
BBQ_HEIGHT = 0.9  # the grate
TABLE_LENGTH, TABLE_LIMITS = 1.8, (1.0, 3.0)
CHESS_TABLE = (0.9, 0.6)  # length, width, when the data does not say
HUT_SIDE, HUT_HEIGHT = 2.2, 2.6
GAZEBO_DIAMETER = 3.6  # a gazebo mapped as a point


def _swing_span(size) -> float:
    return max(2.4, (size or 2) * SWING_SEAT_SPACING + 1.4)


# Where each kind of generic equipment stands: (u0, u1, v0, v1) in its own frame, the box nothing else is put inside.
EXTENT = {
    "play_swing": lambda size, height: (-_swing_span(size) / 2 - 0.2, _swing_span(size) / 2 + 0.2, -1.3, 1.3),
    "play_slide": lambda size, height: (
        -0.7, 0.7, -(0.5 + (height or SLIDE_DECK_HEIGHT) / 2.7 + 0.3),
        0.5 + (height or SLIDE_DECK_HEIGHT) * SLIDE_RUN_PER_METRE + 0.2 + SLIDE_OUT + 0.1,
    ),
    "play_frame": lambda size, height: (-1.8, 1.8, -1.5, 1.5),
    "play_seesaw": lambda size, height: (-0.4, 0.4, -1.65, 1.65),
    "play_spring": lambda size, height: (-0.4, 0.4, -0.5, 0.55),
    "play_sandpit": lambda size, height: (-(size or SANDPIT_SIDE) / 2 - 0.1, (size or SANDPIT_SIDE) / 2 + 0.1, -(size or SANDPIT_SIDE) / 2 - 0.1, (size or SANDPIT_SIDE) / 2 + 0.1),
}
# What a `playground=*` value is, and the kind it is drawn as. Others (roundabout, balancebeam, ...) are not drawn.
PLAY_KINDS = {
    "swing": "play_swing", "basketswing": "play_swing", "slide": "play_slide", "seesaw": "play_seesaw", "springy": "play_spring",
    "climbingframe": "play_frame", "structure": "play_frame", "playhouse": "play_frame", "sandpit": "play_sandpit",
}
GENERIC_ORDER = ("play_frame", "play_swing", "play_slide", "play_spring", "play_sandpit", "play_seesaw")

# --- Layout of generic equipment -----------------------------------------------
EDGE_MARGIN = 1.5  # a piece stands at least this far inside the playground's outline
SEPARATION = 2.0  # and at least this far from any other piece
# Kept this far from each kind of obstacle (a line or a point is also kept half its width from).
CLEAR = {"building": 0.5, "area": 0.3, "way": 0.3, "barrier": 0.8, "tree": 0.9, "object": 0.8}
MIN_PLAYGROUND = 45.0  # square metres: a smaller outline holds nothing
# Pieces by area: 2 below 200 m2 and 8 at 1,500 m2, one for a very small outline, and never more than the maximum.
EQUIPMENT_COUNT = {"base": 2.0, "per_m2": 6.0 / 1400.0, "from": 100.0, "max": 10, "tiny": 80.0}
CLUSTER = 35.0  # a layout fills a circle of about this many square metres to the piece, round the roomiest spot
TRIES = 500  # positions tried for a piece

# --- Allotments ----------------------------------------------------------------
PLOT_WIDTH, PLOT_LENGTH, PLOT_GAP = 4.0, 6.0, 1.0  # a plot across x along; the path between two plots
PLOT_MAIN_PATH, PLOT_MAIN_EVERY = 0.9, 4  # a wider path (this much extra) after every few plots
PLOT_EDGE = 0.9  # a plot stays this far inside the garden's outline
PLOT_MIN_AREA = 100.0
SOIL_HEIGHT = 0.09
NOT_PLOTS = ("rose", "rain ", "butterfly", "memorial", "sensory", "herb")  # a community garden by name or tag that is a show garden
SOIL, LEAF = "#4a3b2d", ("#5f8f3f", "#4f8a3c", "#6a9a45", "#7aa24c", "#4a7a35")
BLOOM = ("#d9b23a", "#d9792c", "#c9457a", "#e8e3d2", "#b04a42", "#7a5aa8")
SHED_WALLS, SHED_ROOF = ("#7b8a6a", "#8a6a4a", "#6a7f8f", "#b0a58a", "#a04a3a", "#c9c3b0"), "#4a4d50"

# --- Bandstands and gazebos ----------------------------------------------------
OPEN_AREA = (8.0, 150.0)  # a mapped bandstand or gazebo building this size is an open structure; bigger is a stage building
ROUND_LIMIT = 0.82  # 4 pi area / perimeter^2 from here up is a round footprint (a square is 0.785, a regular octagon 0.95)
STRUCTURES = {  # floor height, eave height above the ground, how far the roof rises (a share of its radius), steps, half wall
    "bandstand": dict(floor=0.9, eave=3.7, rise=0.55, steps=4, rail=0.95, colour="#d8d3c6", floor_colour="#a9a59b", roof="#4f6a55"),
    "gazebo": dict(floor=0.3, eave=2.5, rise=0.6, steps=2, rail=0.9, colour="#e9e6dc", floor_colour="#8a7050", roof="#55595c"),
}
ROOF_OVER = 0.55  # the roof reaches this far beyond the posts' ring
FINIAL = 0.45  # the spike on the apex


def _c(hex_colour: str, shade: float = 1.0) -> np.ndarray:
    return colour_bytes(hex_colour, shade)


def _theme(o: _Opts) -> tuple[str, str, str]:
    """The playground's colours: its theme number is in the variant (set by `objects`), else one picked by the piece's own hash."""
    digits = "".join(ch for ch in (o.variant or "") if ch.isdigit())
    return THEMES[(int(digits) if digits else o.seed) % len(THEMES)]


# --- Geometry helpers ----------------------------------------------------------
def _poly(p, colour, W, points, toward) -> None:
    """A flat polygon given by local (u, v, z) points, facing the local direction `toward`."""
    world = [W(*q) for q in points]
    face(p, colour, world, W(*toward) - W(0.0, 0.0, 0.0))


def _slab(p, colour, W, a, b, width, thick, ends: bool = True) -> None:
    """A flat bar from local point `a` to `b`, `width` across and `thick` through, in the frame of `_frame`. The bar's top is
    the face that points up (a slide's chute, a seesaw's plank, a ramp, a sloping roof). 12 triangles, 8 without ends."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    t = b - a
    along = t / np.linalg.norm(t)
    flat = np.array([t[0], t[1], 0.0])
    side = np.cross((0.0, 0.0, 1.0), flat) if np.linalg.norm(flat) > 1e-9 else np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    n = np.cross(along, side)
    if n[2] < 0:
        n = -n
    corner = lambda e, j, k: e + j * side * width / 2 + k * n * thick / 2  # noqa: E731
    quad = lambda *ids: [corner((a, b)[i], j, k) for i, j, k in ids]  # noqa: E731
    _poly(p, colour, W, quad((0, -1, 1), (1, -1, 1), (1, 1, 1), (0, 1, 1)), n)
    _poly(p, colour, W, quad((0, -1, -1), (1, -1, -1), (1, 1, -1), (0, 1, -1)), -n)
    for j in (-1, 1):
        _poly(p, colour, W, quad((0, j, -1), (1, j, -1), (1, j, 1), (0, j, 1)), j * side)
    if ends:
        for i, sign in ((0, -1), (1, 1)):
            _poly(p, colour, W, quad((i, -1, -1), (i, 1, -1), (i, 1, 1), (i, -1, 1)), sign * along)


def _post(p, colour, W, a, b, r=0.05) -> None:
    """A square post from local `a` to `b`, its two ends left open: 8 triangles. A foot on the ground is lifted by its own radius
    so a leaning post's corners do not dip under it."""
    a, b = [(q[0], q[1], r * 1.5 if q[2] < r * 1.5 else q[2]) for q in (a, b)]
    limb(p, colour, W, a, b, r, sides=4, closed=False)


def _rung_ladder(p, rail, rung, W, u, v_top, z_top, v_foot, width=0.44, rungs=4) -> None:
    """A ladder leaning from the deck edge at (u, v_top, z_top) down to the ground at v_foot: two rails and rungs."""
    for side in (-1, 1):
        _post(p, rail, W, (u + side * width / 2, v_top, z_top + 0.1), (u + side * width / 2, v_foot, 0.0), 0.022)
    for k in range(1, rungs + 1):
        f = k / (rungs + 1)
        v, z = v_top + (v_foot - v_top) * f, (z_top + 0.1) * (1 - f)
        cuboid(p, rung, W, u - width / 2, u + width / 2, v - 0.02, v + 0.02, z - 0.02, z + 0.02, "tf")


# --- Playground equipment ------------------------------------------------------
def _swing(p, x, y, d, o: _Opts) -> None:
    """A swing set: a top beam on two A-frames and `size` seats (2 by default) hanging on pairs of chains, the swinging going
    along `d`. About 90 triangles for two seats."""
    seats = int(min(max(round(o.size) if o.size else 2, 1), 6))
    theme = _theme(o)
    steel = o.paint(STEEL if o.seed % 3 else theme[2])
    W = _frame(x, y, d)
    half, top = _swing_span(seats) / 2, SWING_BEAM
    _post(p, steel, W, (-half - 0.06, 0.0, top), (half + 0.06, 0.0, top), 0.055)
    for end in (-half, half):
        for lean in (-1, 1):
            _post(p, steel, W, (end, lean * 0.95, 0.0), (end, 0.0, top + 0.02), 0.05)
    chain, seat = _c(CHAIN), _c(RUBBER if o.seed % 2 else theme[1])
    for k in range(seats):
        u = (k - (seats - 1) / 2) * SWING_SEAT_SPACING
        v = (0.35 if (k + o.seed) % 2 else -0.2) * (0.6 + 0.4 * ((o.seed >> k) & 1))  # swings hang at different angles
        for side in (-0.18, 0.18):
            limb(p, chain, W, (u + side, 0.0, top - 0.03), (u + side, v, SWING_SEAT_HEIGHT), 0.012, sides=3, closed=False)
        cuboid(p, seat, W, u - 0.2, u + 0.2, v - 0.09, v + 0.09, SWING_SEAT_HEIGHT - 0.04, SWING_SEAT_HEIGHT, "tbfrlR")


def _slide(p, x, y, d, o: _Opts) -> None:
    """A slide: a square deck on four posts with side panels and a ladder behind it, and a chute running out ahead along `d`
    to a short flat run-out. The deck is `height` high (1.4 m). `stainless_steel` in the material (the City Hall Plaza slide)
    makes the chute steel. About 110 triangles."""
    theme = _theme(o)
    h = o.tall(SLIDE_DECK_HEIGHT, *SLIDE_DECK_LIMITS)
    W = _frame(x, y, d)
    steel = _c(STEEL)
    chute = _c(STAINLESS) if o.variant == "steel" else _c(theme[1])
    half = SLIDE_DECK / 2
    for u in (-0.47, 0.47):
        for v in (-0.47, 0.47):
            _post(p, steel, W, (u, v, 0.0), (u, v, h), 0.045)
    cuboid(p, _c(theme[0]), W, -half - 0.04, half + 0.04, -half - 0.04, half + 0.04, h - 0.06, h, "tbfrlR")
    for u in (-half, half):  # panels along both sides of the deck
        cuboid(p, _c(theme[2]), W, u - 0.02, u + 0.02, -half, half, h, h + 0.75, "tfrlR")
    _rung_ladder(p, steel, _c(STEEL_DARK), W, 0.0, -half - 0.04, h - 0.06, -half - 0.04 - h / 2.7, rungs=max(3, int(h / 0.35)))
    run = h * SLIDE_RUN_PER_METRE + 0.2
    v0, v1, low = half + 0.04, half + 0.04 + run, 0.26
    _slab(p, chute, W, (0.0, v0, h - 0.03), (0.0, v1, low), SLIDE_WIDTH, 0.04)
    _slab(p, chute, W, (0.0, v1, low), (0.0, v1 + SLIDE_OUT, low - 0.002), SLIDE_WIDTH, 0.04)
    for side in (-1, 1):
        _slab(p, chute, W, (side * (SLIDE_WIDTH / 2 + 0.02), v0, h + 0.05), (side * (SLIDE_WIDTH / 2 + 0.02), v1, low + 0.09), 0.04, 0.18, ends=False)


def _tower(p, x, y, d, o: _Opts) -> None:
    """A climbing tower: a platform on four posts under a pyramid roof, panels on three sides, a ladder up the front (+v)
    and a climbing wall down the right-hand side, and a pole at the back. About 130 triangles."""
    theme = _theme(o)
    W = _frame(x, y, d)
    steel = o.paint(STEEL if o.seed % 3 else theme[2])
    half, plat = TOWER_SIDE / 2, TOWER_PLATFORM
    for u in (-half, half):
        for v in (-half, half):
            _post(p, steel, W, (u, v, 0.0), (u, v, TOWER_EAVE), 0.05)
    cuboid(p, _c(theme[0]), W, -half - 0.05, half + 0.05, -half - 0.05, half + 0.05, plat - 0.07, plat, "tbfrlR")
    panel = _c(theme[1])
    for u in (-half, half):
        cuboid(p, panel, W, u - 0.02, u + 0.02, -half, half, plat, plat + 0.8, "tfrlR")
    cuboid(p, panel, W, -half, half, -half - 0.02, -half + 0.02, plat, plat + 0.8, "tfrlR")
    px, py = W(0.0, 0.0, 0.0)[:2]
    roof_r = (TOWER_SIDE + 0.7) / math.sqrt(2)
    pyramid(p, _c(theme[2]), px, py, TOWER_EAVE, TOWER_APEX, roof_r, 4, _square_rot(d))
    disc(p, _c(theme[2], 0.7), px, py, TOWER_EAVE - 0.02, roof_r * 0.97, 4, _square_rot(d), up=False)
    _rung_ladder(p, steel, _c(STEEL_DARK), W, 0.0, half + 0.05, plat - 0.07, half + 0.05 + plat / 2.7, rungs=4)
    _slab(p, _c(theme[0], 0.85), W, (half + 0.04, 0.0, plat - 0.02), (half + 0.85, 0.0, 0.12), 1.3, 0.06)
    for k, (v, z) in enumerate(((-0.3, 0.9), (0.3, 0.7), (0.0, 0.45))):  # climbing holds on the wall
        u = half + 0.04 + (plat - z) / (plat - 0.12) * 0.81
        cuboid(p, _c(("#e3b52f", "#2e9a57", "#c8372d")[k]), W, u - 0.05, u + 0.12, v - 0.1, v + 0.1, z - 0.05, z + 0.05, "tfR")
    _post(p, _c(STEEL), W, (-half, -half - 0.25, plat + 0.05), (-half, -half - 0.25, 0.0), 0.03)  # a pole to slide down


def _seesaw(p, x, y, d, o: _Opts) -> None:
    """A seesaw: a plank tilted on a pivot with a handle at each end, along `d`. About 45 triangles."""
    theme = _theme(o)
    W = _frame(x, y, d)
    half = SEESAW_LENGTH / 2
    steel = _c(STEEL)
    _slab(p, _c(theme[0]), W, (0.0, -half, SEESAW_LOW), (0.0, half, SEESAW_HIGH), 0.26, 0.06)
    mid = (SEESAW_LOW + SEESAW_HIGH) / 2
    cuboid(p, steel, W, -0.07, 0.07, -0.05, 0.05, 0.0, mid - 0.03, "tfrlR")
    cuboid(p, steel, W, -0.06, 0.06, -0.55, 0.55, 0.0, 0.07, "tfrlR")
    for v, z in ((-1.2, SEESAW_LOW + (SEESAW_HIGH - SEESAW_LOW) * 0.15), (1.2, SEESAW_LOW + (SEESAW_HIGH - SEESAW_LOW) * 0.85)):
        _post(p, _c(STEEL_DARK), W, (0.0, v, z + 0.03), (0.0, v, z + 0.45), 0.02)
        cuboid(p, _c(STEEL_DARK), W, -0.14, 0.14, v - 0.015, v + 0.015, z + 0.43, z + 0.47, "t")


def _spring(p, x, y, d, o: _Opts) -> None:
    """A spring rider: a base plate, a spring and a seat shaped like a horse with a head, handlebars and a back, facing `d`.
    About 80 triangles."""
    theme = _theme(o)
    W = _frame(x, y, d)
    colour = _c(theme[(o.seed >> 4) % 3])
    cuboid(p, _c(STEEL_DARK), W, -0.22, 0.22, -0.32, 0.32, 0.0, 0.04, "tfrlR")
    px, py = W(0.0, 0.0, 0.0)[:2]
    for k, (z0, z1) in enumerate(((0.04, 0.15), (0.16, 0.27), (0.28, 0.38))):
        prism(p, _c(STEEL, 1.0 if k % 2 == 0 else 0.7), px, py, z0, z1, 0.08, sides=5)
    cuboid(p, colour, W, -0.13, 0.13, -0.30, 0.34, 0.38, 0.62, "tbfrlR")
    cuboid(p, colour, W, -0.10, 0.10, 0.26, 0.50, 0.56, 0.82, "tbfrlR")
    cuboid(p, _c(RUBBER), W, -0.11, 0.11, -0.12, 0.12, 0.62, 0.66, "t")  # the saddle
    limb(p, _c(STEEL_DARK), W, (-0.2, 0.27, 0.68), (0.2, 0.27, 0.68), 0.015, sides=3, closed=False)
    for side in (-1, 1):
        cuboid(p, _c(STEEL_DARK), W, side * 0.15 - 0.02, side * 0.15 + 0.02, 0.05, 0.12, 0.36, 0.40, "tf")


def _sandpit(p, x, y, d, o: _Opts) -> None:
    """A sandpit: a square timber border round level sand. `size` is the side (2.4 m). About 45 triangles."""
    side = min(max(o.size if o.size else SANDPIT_SIDE, SANDPIT_LIMITS[0]), SANDPIT_LIMITS[1])
    W = _frame(x, y, d)
    wood, half, rim = o.paint(TIMBER), side / 2, 0.13
    cuboid(p, wood, W, -half, half, half - rim, half, 0.0, 0.24, "tfrlR")
    cuboid(p, wood, W, -half, half, -half, -half + rim, 0.0, 0.24, "tfrlR")
    cuboid(p, wood, W, -half, -half + rim, -half + rim, half - rim, 0.0, 0.24, "tfrlR")
    cuboid(p, wood, W, half - rim, half, -half + rim, half - rim, 0.0, 0.24, "tfrlR")
    inner = half - rim
    _poly(p, _c(SAND), W, [(-inner, -inner, 0.16), (inner, -inner, 0.16), (inner, inner, 0.16), (-inner, inner, 0.16)], (0, 0, 1))


# --- Park furniture the data states --------------------------------------------
def _bbq(p, x, y, d, o: _Opts) -> None:
    """A barbecue grill: a charcoal grill is a bowl on a post, a gas one a steel cabinet, each with a grate at 0.9 m. About 50 triangles."""
    iron = o.paint(GRILL_IRON)
    if o.variant == "gas":
        box(p, iron, x, y, 0.0, 0.78, 0.9, 0.5, d)
        box(p, _c(GRILL_GRATE), x, y, 0.78, 0.92, 0.94, 0.54, d)
        return
    prism(p, iron, x, y, 0.0, 0.62, 0.05, 0.05, 6)
    prism(p, iron, x, y, 0.62, BBQ_HEIGHT - 0.04, 0.2, 0.3, 8)
    disc(p, _c(GRILL_GRATE), x, y, BBQ_HEIGHT, 0.3, 8)


def _picnic_table(p, x, y, d, o: _Opts) -> None:
    """A picnic table with a bench along each side, `size` long (1.8 m), its length along `d`. About 70 triangles."""
    length = min(max(o.size if o.size else TABLE_LENGTH, TABLE_LIMITS[0]), TABLE_LIMITS[1])
    wood = o.paint(TABLE_TIMBER)
    W = _frame(x, y, d)
    half = length / 2
    cuboid(p, wood, W, -0.38, 0.38, -half, half, 0.72, 0.77, "tfrlR")
    for side in (-1, 1):
        cuboid(p, wood, W, side * 0.62 - 0.14, side * 0.62 + 0.14, -half, half, 0.43, 0.46, "tfrlR")
    legs = _c(TABLE_TIMBER, 0.8)
    for v in (-half * 0.7, half * 0.7):
        for side in (-1, 1):
            _post(p, legs, W, (side * 0.62, v, 0.0), (side * 0.3, v, 0.72), 0.035)
        cuboid(p, legs, W, -0.62, 0.62, v - 0.02, v + 0.02, 0.38, 0.42, "t")


def _chess_table(p, x, y, d, o: _Opts) -> None:
    """A chess table: a top with an 8 x 8 board (0.5 m) on a pedestal, and a stool at each end. `size` is the table's length
    (0.9 m), its width 0.6 m; the table is the stated `colour` (black) or grey metal, the board's light squares cream. About 140 triangles."""
    length = min(max(o.size if o.size else CHESS_TABLE[0], 0.6), 2.0)
    width = CHESS_TABLE[1]
    W = _frame(x, y, d)
    cuboid(p, o.paint(CHESS_DARK), W, -width / 2, width / 2, -length / 2, length / 2, 0.70, 0.75, "tfrlR")
    board, cell = 0.5, 0.5 / 8
    light = _c(CHESS_LIGHT)
    z = 0.752
    for i in range(8):
        for j in range(8):
            if (i + j) % 2:
                u0, v0 = -board / 2 + i * cell, -board / 2 + j * cell
                _poly(p, light, W, [(u0, v0, z), (u0 + cell, v0, z), (u0 + cell, v0 + cell, z), (u0, v0 + cell, z)], (0, 0, 1))
    px, py = W(0.0, 0.0, 0.0)[:2]
    prism(p, _c(CHESS_DARK), px, py, 0.0, 0.70, 0.07, sides=6)
    for side in (-1, 1):
        sx, sy = W(0.0, side * (length / 2 + 0.28), 0.0)[:2]
        prism(p, _c(CHESS_DARK), sx, sy, 0.0, 0.44, 0.17, sides=6, top=True)


def _toilets(p, x, y, d, o: _Opts) -> None:
    """A toilet hut with its door facing `d`, nothing at all when a building covers the point (variant "indoor"). A steel one
    (`The Portland Loo`: variant "steel") is narrower and taller. About 40 triangles."""
    if o.variant == "indoor":
        return
    steel = o.variant == "steel"
    side, depth, tall = (1.6, 2.2, 2.7) if steel else (HUT_SIDE, HUT_SIDE, HUT_HEIGHT)
    box(p, o.paint("#8e9496" if steel else HUT_WALL), x, y, 0.0, tall, side, depth, d)
    W = _frame(x, y, d)
    box(p, _c(HUT_ROOF), x, y, tall, tall + 0.12, side + 0.3, depth + 0.3, d)
    front = depth / 2 + 0.006
    _poly(p, _c("#52606a" if steel else HUT_DOOR), W, [(-0.4, front, 0.0), (0.4, front, 0.0), (0.4, front, 2.0), (-0.4, front, 2.0)], (0, 1, 0))
    for u in (-side / 2 + 0.25, side / 2 - 0.25):  # a vent each side of the door
        _poly(p, _c(HUT_ROOF), W, [(u - 0.15, front, tall - 0.5), (u + 0.15, front, tall - 0.5), (u + 0.15, front, tall - 0.25), (u - 0.15, front, tall - 0.25)], (0, 1, 0))


def _gazebo(p, x, y, d, o: _Opts) -> None:
    """A gazebo mapped as a point: an eight-sided open roof on posts over a low floor, 3.6 m across (or `size`)."""
    radius = min(max((o.size if o.size else GAZEBO_DIAMETER) / 2, 1.2), 6.0)
    _pavilion(p, p, x, y, 8, radius, d, STRUCTURES["gazebo"], steps_toward=d)


# --- Open structures: bandstands and gazebos -----------------------------------
def _pavilion(body, roof, x, y, sides: int, radius: float, rot: float, style: dict, steps_toward: float | None = None, tint: dict | None = None) -> None:
    """A ring of posts round a raised floor with a half wall between them (except at the steps) and a conical roof. `(x, y)`
    is the centre in tile metres; `radius` the circumradius of the floor's polygon; `rot` the angle in degrees (counter-clockwise
    from east) of its first corner; `steps_toward` the compass heading the steps face. `body` collects floor, posts and walls and
    `roof` the roof (the same collector, for a point object). About 150 triangles."""
    floor, eave, rail = style["floor"], style["eave"], style["rail"]
    names = ("colour", "floor_colour", "roof")
    colour, floor_colour, roof_colour = (_c((tint or {}).get(k, style[k])) for k in names)
    angles = [math.radians(rot) + 2 * math.pi * k / sides for k in range(sides)]
    prism(body, floor_colour, x, y, 0.0, floor, radius + 0.2, sides=sides, rot=rot, top=True)
    ring = radius - 0.15
    corners = [(x + ring * math.cos(a), y + ring * math.sin(a)) for a in angles]
    entry = -1
    if steps_toward is not None:  # steps on the side facing the path: the bay whose middle points that way, and its steps
        want = math.atan2(math.cos(math.radians(steps_toward)), math.sin(math.radians(steps_toward)))  # the heading as an angle from east
        mids = [a + math.pi / sides for a in angles]
        entry = min(range(sides), key=lambda k: abs((mids[k] - want + math.pi) % (2 * math.pi) - math.pi))
        mx, my = math.cos(mids[entry]), math.sin(mids[entry])
        yaw, tread, steps = math.degrees(math.atan2(mx, my)), 0.3, max(1, int(style["steps"]))
        edge = (radius + 0.2) * math.cos(math.pi / sides)
        for k in range(steps):
            reach = edge + tread * (steps - k - 0.5)
            box(body, floor_colour, x + mx * reach, y + my * reach, 0.0, floor * (k + 1) / (steps + 1), min(1.6 + radius * 0.1, 2.4), tread, yaw)
    for k in range(sides):
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % sides]
        prism(body, colour, ax, ay, floor, eave, 0.09, sides=4, rot=45.0)
        if k == entry:
            continue
        nx, ny = by - ay, -(bx - ax)  # the way the bay faces out (the ring runs counter-clockwise)
        norm = math.hypot(nx, ny) or 1.0
        nx, ny = nx / norm * 0.05, ny / norm * 0.05
        low, high = floor, floor + rail  # the half wall: both faces and its top
        for sign in (1, -1):
            face(body, colour, [(ax + sign * nx, ay + sign * ny, low), (bx + sign * nx, by + sign * ny, low), (bx + sign * nx, by + sign * ny, high), (ax + sign * nx, ay + sign * ny, high)], (sign * nx, sign * ny, 0.0))
        face(body, colour, [(ax + nx, ay + ny, high), (bx + nx, by + ny, high), (bx - nx, by - ny, high), (ax - nx, ay - ny, high)], (0, 0, 1))
    span = radius + ROOF_OVER
    rise = max(0.9, style["rise"] * span)
    fascia = [(x + span * math.cos(a), y + span * math.sin(a)) for a in angles]
    for k in range(sides):
        (ax, ay), (bx, by) = fascia[k], fascia[(k + 1) % sides]
        face(roof, _c((tint or {}).get("roof", style["roof"]), 0.75), [(ax, ay, eave - 0.2), (bx, by, eave - 0.2), (bx, by, eave), (ax, ay, eave)], (by - ay, -(bx - ax), 0.0))
    pyramid(roof, roof_colour, x, y, eave, eave + rise, span, sides, rot)
    disc(roof, _c(style["colour"], 0.9), x, y, eave - 0.02, span, sides, rot, up=False)
    prism(roof, colour, x, y, eave + rise, eave + rise + FINIAL, 0.07, 0.03, sides=4, rot=45.0)


def _isoperimetric(polygon: Polygon) -> float:
    return 4 * math.pi * polygon.area / max(polygon.length**2, 1e-9)


def open_structure(building) -> str | None:
    """"bandstand" or "gazebo" when this mapped building is an open bandstand or gazebo (see OPEN_AREA), else None."""
    tags, area = building.tags, building.polygon.area
    if building.part or not (OPEN_AREA[0] <= area <= OPEN_AREA[1]):
        return None
    if tags.get("leisure") == "bandstand":
        return "bandstand"
    if tags.get("amenity") == "shelter" and str(tags.get("shelter_type", "")).strip().lower() in ("gazebo", "pavilion"):
        return "gazebo"
    return None


def _toward_way(ctx, point) -> float | None:
    """The heading from `point` toward the nearest footway, path or road within 40 m, if there is one (where a bandstand's steps go)."""
    scene, best = ctx.scene, None
    for i in ctx.ways_tree.query(point, predicate="dwithin", distance=40.0):
        way = scene.ways[i]
        if way.surface == "rail":
            continue
        distance = way.line.distance(point)
        if best is None or distance < best[0]:
            best = (distance, way.line)
    if best is None:
        return None
    foot = best[1].interpolate(best[1].project(point))
    return math.degrees(math.atan2(foot.x - point.x, foot.y - point.y)) % 360 if foot.distance(point) > 0.5 else None


def add_open_structure(ctx, walls, roofs, building, origin, feature: int) -> bool:
    """Draw a bandstand or gazebo building as an open structure (see the module docstring) into the tile's `walls` and `roofs`;
    False if it is not one (the caller then draws it as a building). A round footprint gets posts, a half wall and a conical roof;
    any other pavilions.py's rectangle of posts and hipped roof. A `height` or `building:levels` in the data is kept as the total
    height to the apex; otherwise the standard one."""
    kind = open_structure(building)
    if kind is None:
        return False
    if _isoperimetric(building.polygon) < ROUND_LIMIT:  # a rectangular roof on posts, as pavilions.py draws a pavilion
        add_pavilion(walls, roofs, building, origin, feature)
        return True
    style = dict(STRUCTURES[kind])
    centre = building.polygon.centroid
    coords = np.asarray(building.polygon.exterior.coords)[:-1]
    sides = len(coords) if len(coords) in (6, 8) else 8
    radius = min(max(math.sqrt(2 * building.polygon.area / (sides * math.sin(2 * math.pi / sides))), 1.0), 12.0)
    rot = math.degrees(math.atan2(coords[0][1] - centre.y, coords[0][0] - centre.x))
    if building.height_source in ("height", "levels", "override"):  # a stated total height sets the eaves, the roof and finial taking their share
        rise = max(0.9, style["rise"] * (radius + ROOF_OVER))
        style["eave"] = max(building.height - rise - FINIAL, style["floor"] + 2.0)
    toward = _toward_way(ctx, Point(centre.x, centre.y))
    tint = {}
    colour = building.tags.get("colour") or building.tags.get("building:colour")
    if colour and str(colour).startswith("#"):
        tint["colour"] = colour
    body, roof = _Parts(), _Parts()
    _pavilion(body, roof, centre.x - origin[0], centre.y - origin[1], sides, radius, rot, style, steps_toward=toward if toward is not None else 180.0, tint=tint)
    body.flush(walls, feature)
    roof.flush(roofs, feature)
    return True


# --- Placing generic equipment -------------------------------------------------
class _Index:
    """What a playground or a garden must keep clear of, by place, built once per scene."""

    def __init__(self, scene: Scene) -> None:
        geoms, pad = [], []

        def add(geom, how: str, extra: float = 0.0) -> None:
            if geom is not None and not geom.is_empty:
                geoms.append(geom)
                pad.append(CLEAR[how] + extra)

        for building in scene.buildings:
            add(building.polygon, "building")
        for kind in ("road", "path", "water", "bed"):
            for area in scene.areas[kind]:
                add(area.polygon, "area")
        for way in scene.ways:
            if way.surface in ("path", "road"):
                add(way.line, "way", way.width / 2)
        for barrier in scene.barriers:
            add(barrier.line, "barrier")
        for tree in scene.trees:
            add(Point(tree.x, tree.y), "tree")
        for thing in scene.objects:
            add(Point(thing.x, thing.y), "object")
        self.geoms, self.pad = geoms, np.array(pad)
        self.tree = STRtree(geoms) if geoms else None
        # Other ground that is not the playground: where a smaller pitch, garden, wood or lawn lies on it, that is what is drawn there
        # (the smaller area shows), so the equipment is kept to the play surface that is left.
        self.other = [a for a in scene.areas["green"] if a.tags.get("leisure") != "playground"]
        self.other_tree = STRtree([a.polygon for a in self.other]) if self.other else None
        self.plots: dict[str, list] | None = None

    def obstacles(self, region: Polygon):
        """The union of everything near `region`, each thing grown by its clearance."""
        if self.tree is None:
            return Polygon()
        hits = self.tree.query(region.buffer(12.0))
        return unary_union([self.geoms[i].buffer(self.pad[i], quad_segs=2) for i in hits]) if len(hits) else Polygon()

    def other_ground(self, playground: Polygon):
        """Pitches, gardens, woods and lawns smaller than `playground` lying on it."""
        if self.other_tree is None:
            return Polygon()
        hits = [self.other[i].polygon for i in self.other_tree.query(playground, predicate="intersects") if self.other[i].polygon.area < playground.area]
        return unary_union(hits) if hits else Polygon()


_cache: dict = {}


def _index(scene: Scene) -> _Index:
    if _cache.get("scene") is not scene:
        _cache.clear()
        _cache["scene"], _cache["index"] = scene, _Index(scene)
    return _cache["index"]


def equipment_count(area: float) -> int:
    """How many pieces a playground of `area` square metres gets: 2 under 200, 8 at 1,500, at most EQUIPMENT_COUNT["max"]."""
    k = EQUIPMENT_COUNT
    if area < MIN_PLAYGROUND:
        return 0
    if area < k["tiny"]:
        return 1
    return int(min(max(round(k["base"] + k["per_m2"] * (area - k["from"])), 2), k["max"]))


def _axis(polygon: Polygon) -> float:
    """The heading, 0 to 180 degrees clockwise from north, of the long side of the polygon's smallest enclosing rectangle."""
    corners = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)
    edges = np.diff(corners, axis=0)[:2]
    long_edge = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    return math.degrees(math.atan2(long_edge[0], long_edge[1])) % 180


def _footprint(kind: str, x: float, y: float, direction: float, size, height) -> Polygon:
    """The ground a piece takes, as a polygon in world metres."""
    u0, u1, v0, v1 = EXTENT[kind](size, height)
    t = math.radians(direction)
    return Polygon([(x + u * math.cos(t) + v * math.sin(t), y - u * math.sin(t) + v * math.cos(t)) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])


def _piece_size(kind: str, rng) -> tuple[float | None, float | None]:
    """(size, height) for a generic piece: a swing's seats, a slide's deck height, a sandpit's side."""
    if kind == "play_swing":
        return float(rng.choice([2, 2, 2, 4])), None
    if kind == "play_slide":
        return None, round(float(rng.uniform(1.1, 1.7)), 2)
    if kind == "play_sandpit":
        return round(float(rng.uniform(2.0, 3.0)), 1), None
    return None, None


def _equipment_for(area, index: _Index, surveyed: dict[str, list]) -> list[Object]:
    polygon: Polygon = area.polygon
    if polygon.area < MIN_PLAYGROUND or area.tags.get("indoor") == "yes":
        return []
    rng = np.random.default_rng(zlib.crc32(area.id.encode()) ^ (int(polygon.area * 10) & 0xFFFF))
    count = equipment_count(polygon.area)
    usable = polygon.buffer(-EDGE_MARGIN)
    if usable.is_empty:
        return []
    usable = usable.difference(index.obstacles(polygon)).difference(index.other_ground(polygon).buffer(0.3))
    pieces = [g for g in polygons_of(usable) if g.area >= 6.0]
    if not pieces:
        return []
    usable = unary_union(pieces)
    shapely.prepare(usable)
    heading = _axis(polygon)
    # The pieces, in the order they are tried: a big structure and a swing first, and none of a kind the data already has here.
    near = polygon.buffer(4.0)
    have = {kind for kind, points in surveyed.items() if any(near.contains(point) for point in points)}
    order = [k for k in GENERIC_ORDER if k not in have] or list(GENERIC_ORDER)
    first = ["play_frame", "play_slide"][int(rng.integers(0, 2))] if "play_frame" in order and "play_slide" in order else order[0]
    rest = [k for k in order if k not in (first, "play_swing")]
    rng.shuffle(rest)
    sequence = [first, *(["play_swing"] if "play_swing" in order and first != "play_swing" else []), *rest]
    sequence = (sequence * 3)[:count]
    theme = int(rng.integers(0, len(THEMES)))
    # A large outline is a park with a playground in it somewhere: the pieces cluster round the roomiest spot.
    centre = polylabel(max(pieces, key=lambda g: g.area), tolerance=0.5)
    radius = math.sqrt(count * CLUSTER) + 6.0
    placed: list[tuple[Polygon, Object]] = []
    for number, kind in enumerate(sequence):
        size, height = _piece_size(kind, rng)
        for x, y in _candidate_spots(rng, usable, centre, radius):
            direction = (heading + 90 * int(rng.integers(0, 4)) + float(rng.uniform(-6, 6))) % 360
            footprint = _footprint(kind, x, y, direction, size, height)
            if not usable.contains(footprint) or any(footprint.distance(other) < SEPARATION for other, _ in placed):
                continue
            tags = {
                "playground": kind.removeprefix("play_").replace("spring", "springy").replace("frame", "structure"),
                "generic": "yes", "source": "layout from the playground's outline (playgrounds.py)",
            }
            placed.append((footprint, Object(f"pg:{area.id}:{number}", float(x), float(y), kind, None, direction, height, size, None, None, str(theme), tags)))
            break
    return [thing for _, thing in placed]


def _candidate_spots(rng, region, centre, radius: float) -> list[tuple[float, float]]:
    """Positions to try for a piece: random points in `region`, the ones within `radius` of `centre` first."""
    x0, y0, x1, y1 = region.bounds
    xs, ys = rng.uniform(x0, x1, TRIES), rng.uniform(y0, y1, TRIES)
    inside = shapely.contains_xy(region, xs, ys)
    xs, ys = xs[inside], ys[inside]
    rank = np.hypot(xs - centre.x, ys - centre.y) + rng.uniform(0.0, 3.0, len(xs))
    order = np.argsort(rank)
    return [(float(xs[i]), float(ys[i])) for i in order if rank[i] <= radius] + [(float(xs[i]), float(ys[i])) for i in order if rank[i] > radius]


def _tidy_toilets(scene: Scene) -> None:
    """A toilet node inside a building stands for the room, which the building already is: it keeps its place in the data but
    draws nothing (variant "indoor"). One outside faces the nearest path or road within 25 m, if the data gives no direction.
    This changes the objects in place; the scene's own list and the one tiles.prepare builds share them."""
    huts = [t for t in scene.objects if t.kind == "toilets"]
    if not huts:
        return
    footprints = STRtree([b.polygon for b in scene.buildings]) if scene.buildings else None
    lines = [w.line for w in scene.ways if w.surface in ("path", "road")]
    ways = STRtree(lines) if lines else None
    for hut in huts:
        point = Point(hut.x, hut.y)
        if footprints is not None and len(footprints.query(point, predicate="within")):
            hut.variant = "indoor"
        elif hut.direction is None and ways is not None:
            near = ways.query(point, predicate="dwithin", distance=25.0)
            if len(near):
                line = lines[min(near, key=lambda i: lines[i].distance(point))]
                foot = line.interpolate(line.project(point))
                if foot.distance(point) > 0.3:
                    hut.direction = math.degrees(math.atan2(foot.x - point.x, foot.y - point.y)) % 360


def objects(scene: Scene) -> list[Object]:
    """The generic equipment of every playground in the scene (see the module docstring); also settles the toilets (see `_tidy_toilets`)."""
    if not isinstance(scene, Scene):  # extras.objects is also asked with stand-ins in tests/test_extras.py
        return []
    index = _index(scene)
    surveyed: dict[str, list] = {}
    for thing in scene.objects:
        if thing.kind.startswith("play_"):
            surveyed.setdefault(thing.kind, []).append(Point(thing.x, thing.y))
    made: list[Object] = []
    for area in scene.areas["green"]:
        if area.tags.get("leisure") == "playground":
            made.extend(_equipment_for(area, index, surveyed))
    _tidy_toilets(scene)
    return made


# --- Allotment plots -----------------------------------------------------------
@dataclass
class Plot:
    x: float
    y: float
    heading: float  # of the plot's long side, degrees clockwise from north
    width: float
    length: float
    seed: int


def is_allotment(tags: dict) -> bool:
    """Whether a garden is a field of plots: `landuse=allotments`, or a `leisure=garden` that is a community garden."""
    if tags.get("landuse") == "allotments":
        return True
    if tags.get("leisure") != "garden":
        return False
    name = str(tags.get("name", "")).lower()
    if any(word in name for word in NOT_PLOTS) or tags.get("landuse") == "grass":
        return False
    kind = tags.get("garden:type")
    if kind:
        return kind in ("community", "allotment")
    return "community garden" in name


def _plots_for(area, index: _Index) -> list[Plot]:
    polygon: Polygon = area.polygon
    rng = np.random.default_rng(zlib.crc32(area.id.encode()) ^ (int(polygon.area * 10) & 0xFFFF))
    free = polygon.buffer(-PLOT_EDGE).difference(index.obstacles(polygon))
    if free.is_empty:
        return []
    shapely.prepare(free)
    corners = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
    lengths = [float(np.hypot(*e)) for e in edges]
    first = 0 if lengths[0] >= lengths[1] else 1
    along = edges[first] / lengths[first]  # the long side: plots' long sides run along it
    across = np.array([-along[1], along[0]])
    extent = (lengths[first], lengths[1 - first])
    origin = corners.mean(axis=0) - along * extent[0] / 2 - across * extent[1] / 2
    width = PLOT_WIDTH * float(rng.uniform(0.9, 1.1))
    length = PLOT_LENGTH * float(rng.uniform(0.9, 1.1))
    heading = math.degrees(math.atan2(along[0], along[1])) % 360
    found: list[Plot] = []
    j, v = 0, 0.0
    while v + width <= extent[1] + 1e-6:
        i, u = 0, 0.0
        while u + length <= extent[0] + 1e-6:
            centre = origin + along * (u + length / 2) + across * (v + width / 2)
            rect = Polygon([centre + along * sx * length / 2 + across * sy * width / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if free.contains(rect):
                found.append(Plot(float(centre[0]), float(centre[1]), heading, width, length, zlib.crc32(f"{area.id}:{i}:{j}".encode())))
            i += 1
            u += length + PLOT_GAP + (PLOT_MAIN_PATH if i % PLOT_MAIN_EVERY == 0 else 0.0)
        j += 1
        v += width + PLOT_GAP + (PLOT_MAIN_PATH if j % PLOT_MAIN_EVERY == 0 else 0.0)
    return found


def plots(scene: Scene) -> list[Plot]:
    """Every allotment plot in the scene, laid out once."""
    index = _index(scene)
    if index.plots is None:
        index.plots = {}
        for area in scene.areas["green"]:
            if is_allotment(area.tags) and area.polygon.area >= PLOT_MIN_AREA:
                index.plots.setdefault(area.id, []).extend(_plots_for(area, index))
    return [plot for found in index.plots.values() for plot in found]


def _draw_plot(p, plot: Plot, x0: float, y0: float) -> None:
    """One plot at (x0, y0) tile metres: a low bed of soil, rows of planting along its length, sometimes a shed, a bean tepee or
    a bean tent. About 50 triangles."""
    rng = np.random.default_rng(plot.seed)
    W = _frame(plot.x - x0, plot.y - y0, plot.heading)
    w, l = plot.width, plot.length
    shade = float(rng.uniform(0.85, 1.2))
    cuboid(p, _c(SOIL, shade), W, -w / 2, w / 2, -l / 2, l / 2, 0.0, SOIL_HEIGHT, "t")
    cuboid(p, _c(SOIL, shade * 0.7), W, -w / 2, w / 2, -l / 2, l / 2, 0.0, SOIL_HEIGHT, "frlR")
    crop = int(rng.integers(0, 8))  # 0-2 leafy rows, 3-4 tall beans and tomatoes, 5 flowers, 6 mixed, 7 bare
    leaf = LEAF[int(rng.integers(0, len(LEAF)))]
    bloom = BLOOM[int(rng.integers(0, len(BLOOM)))]
    rows = 0 if crop == 7 else max(2, int((w - 0.4) / 0.78))
    spacing = (w - 0.4) / max(rows, 1)
    for r in range(rows):
        u = -w / 2 + 0.2 + spacing * (r + 0.5)
        kind = crop if crop != 6 else int(rng.choice([1, 2, 3, 5]))
        tall = 0.18 if kind <= 2 else 0.95 if kind in (3, 4) else 0.5
        wide = 0.5 if kind <= 2 else 0.38 if kind in (3, 4) else 0.45
        colour = _c(leaf, float(rng.uniform(0.85, 1.1)))
        side = _c(leaf, float(rng.uniform(0.7, 0.9)))
        crest = _c(bloom) if kind == 5 else colour  # a flower row is green with a crest of bloom
        v0, v1 = -l / 2 + 0.35, l / 2 - 0.35
        runs = int(rng.integers(1, 4)) if tall > 0.3 else 1  # a tall row (beans, tomatoes, flowers) is a few clumps, not a wall
        run = (v1 - v0 - 0.4 * (runs - 1)) / runs
        for k in range(runs):
            a = v0 + k * (run + 0.4)
            _row(p, W, colour, side, crest, u, a, a + run, wide, SOIL_HEIGHT, SOIL_HEIGHT + tall * (float(rng.uniform(0.75, 1.25)) if runs > 1 else 1.0), tall > 0.3)
    _structures(p, W, rng, plot.heading, w, l, bare=rows == 0)


def _row(p, W, colour, side, crest, u, v0, v1, wide, z0, top, flat_top: bool) -> None:
    """A row of planting along v from v0 to v1 centred on u: two sloping sides and two ends; a tall row has a flat top (`crest`)."""
    inset = wide * 0.22 if flat_top else 0.0
    lo, hi = u - wide / 2, u + wide / 2
    _poly(p, side, W, [(lo, v0, z0), (lo, v1, z0), (u - inset, v1, top), (u - inset, v0, top)], (-1, 0, 0.6))
    _poly(p, colour, W, [(hi, v0, z0), (hi, v1, z0), (u + inset, v1, top), (u + inset, v0, top)], (1, 0, 0.6))
    if flat_top:
        _poly(p, crest, W, [(u - inset, v0, top), (u + inset, v0, top), (u + inset, v1, top), (u - inset, v1, top)], (0, 0, 1))
    for v, toward in ((v0, (0, -1, 0)), (v1, (0, 1, 0))):
        pts = [(lo, v, z0), (hi, v, z0), (u + inset, v, top), (u - inset, v, top)] if flat_top else [(lo, v, z0), (hi, v, z0), (u, v, top)]
        _poly(p, colour, W, pts, toward)


def _structures(p, W, rng, heading: float, w: float, l: float, bare: bool) -> None:
    """The odd shed, bean tepee or bean tent in a plot."""
    roll = float(rng.random())
    if roll < 0.12:  # a shed at the back corner
        u = (w / 2 - 0.9) * (1 if rng.random() < 0.5 else -1)
        v = l / 2 - 0.85
        cuboid(p, _c(SHED_WALLS[int(rng.integers(0, len(SHED_WALLS)))]), W, u - 0.7, u + 0.7, v - 0.6, v + 0.6, SOIL_HEIGHT, 1.85, "tfrlR")
        _slab(p, _c(SHED_ROOF), W, (u, v - 0.72, 2.0), (u, v + 0.72, 1.82), 1.7, 0.06)
        _poly(p, _c("#3a3d3f"), W, [(u - 0.2, v - 0.606, SOIL_HEIGHT), (u + 0.2, v - 0.606, SOIL_HEIGHT), (u + 0.2, v - 0.606, 1.6), (u - 0.2, v - 0.606, 1.6)], (0, -1, 0))
    elif roll < 0.30 and not bare:  # bean poles in a tepee
        x, y = W((rng.random() - 0.5) * (w - 1.4), (rng.random() - 0.5) * (l - 2.0), 0.0)[:2]
        pyramid(p, _c(LEAF[int(rng.integers(0, len(LEAF)))], 0.9), x, y, SOIL_HEIGHT, 1.9, 0.6, 4, _square_rot(heading))
    elif roll < 0.42 and not bare:  # a tent of canes along a row
        u, v = (rng.random() - 0.5) * (w - 1.4), (rng.random() - 0.5) * (l - 3.0)
        leaf = _c(LEAF[int(rng.integers(0, len(LEAF)))], 0.9)
        _slab(p, leaf, W, (u - 0.5, v, SOIL_HEIGHT), (u, v, 1.5), 1.5, 0.05, ends=False)
        _slab(p, leaf, W, (u + 0.5, v, SOIL_HEIGHT), (u, v, 1.5), 1.5, 0.05, ends=False)


def add_plots(scene: Scene, tile_box, origin, builder) -> None:
    """Draw the allotment plots whose middle lies in the tile into its `beds` layer (each plot once, in the tile it is centred in)."""
    x0, y0, x1, y1 = tile_box.bounds
    found = [plot for plot in plots(scene) if x0 <= plot.x < x1 and y0 <= plot.y < y1]
    if not found:
        return
    parts = _Parts()
    for plot in found:
        _draw_plot(parts, plot, float(origin[0]), float(origin[1]))
    parts.flush(builder, 0)


# --- Interpretation of points --------------------------------------------------
def interpret(tags: dict):
    """(kind, variant, size) for a point that is one of this module's things, else None (see extras.py)."""
    play = tags.get("playground")
    if play:
        kind = PLAY_KINDS.get(str(play).split(";")[0].strip().lower())
        if kind:
            seats = parse_number(tags.get("seats"))
            return kind, "steel" if "stainless" in str(tags.get("material", "")).lower() else None, (seats if kind == "play_swing" and seats and seats > 0 else None)
    amenity = tags.get("amenity")
    if amenity == "bbq":
        return "bbq", "gas" if str(tags.get("fuel", "")).lower() in ("gas", "propane", "lpg", "natural_gas") else None, None
    if tags.get("leisure") == "picnic_table" or amenity == "picnic_table":
        return "picnic_table", None, parse_length(tags.get("length"))
    if tags.get("sport") == "chess" and amenity == "table":
        return "chess_table", None, parse_length(tags.get("length"))
    if amenity == "toilets":
        return "toilets", "steel" if "loo" in str(tags.get("name", "")).lower() else None, None
    if amenity == "shelter" and str(tags.get("shelter_type", "")).strip().lower() == "gazebo":
        return "open_gazebo", None, parse_length(tags.get("diameter") or tags.get("width"))
    return None


SHAPES = {
    "play_swing": _swing,
    "play_slide": _slide,
    "play_frame": _tower,
    "play_seesaw": _seesaw,
    "play_spring": _spring,
    "play_sandpit": _sandpit,
    "bbq": _bbq,
    "picnic_table": _picnic_table,
    "chess_table": _chess_table,
    "toilets": _toilets,
    "open_gazebo": _gazebo,
}

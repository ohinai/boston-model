"""The park's landmarks in more detail: people, animals, pedestals, fountain centrepieces, a stone lantern, a flag.

`furniture.py` draws every statue, fountain and memorial and calls into this module for the shapes that need more than
a few boxes. Nothing here is a likeness or a measurement: each shape is a placeholder of the right kind and size. What
a thing is comes from its tags, and the tags are the OpenStreetMap ones where there are any, else the project's own,
set for one place in `fixes/` (documented in AGENTS.md):

- `animal=duck|goose|dog|lion|panther|turtle|...` on a statue or fountain: which animal (`animal_family`). `length`
  is its nose-to-tail size; with none the caller's default is used.
- `figure=child` on a statue or fountain: a child's proportions (a big head, short legs) instead of an adult's.
- `figures=2|3` on a standing statue or a sculpted fountain: how many figures (a group on one pedestal).
- `pedestal=bell|shaft` on a standing statue: a round bell-shaped pedestal, or a tall tapering granite shaft.
- a `man_made=flagpole` memorial that flies the United States flag (`flag:wikidata=Q42537` or `country=US`).

Everything is z-up metres, flat-shaded, one normal per face, in the style of furniture.py (this module imports its
helpers: furniture.py imports this one once they exist, so either can be imported first).
"""

from __future__ import annotations

import math

import numpy as np

from .features import flies_us_flag  # noqa: F401  (furniture.py asks for it here)
from .furniture import PLINTH_STONE, _frame, _square_rot, annulus, box, disc, limb, octahedron, prism, pyramid
from .mesh import colour_bytes

# --- Animals -------------------------------------------------------------------------------------------------------
BIRDS = frozenset({"duck", "ducks", "duckling", "mallard", "goose", "geese", "swan", "bird"})
FOUR_LEGGED = frozenset({"dog", "lion", "cat", "panther", "leopard", "tiger", "bear", "wolf", "fox", "deer", "hare", "rabbit"})
TURTLES = frozenset({"turtle", "sea_turtle", "tortoise", "terrapin"})
# The shapes below are drawn in a unit `k`; these are what that unit shape measures, so that `animal` can be asked for
# a nose-to-tail length in metres: its nose-to-tail length in units of k, and its height as a fraction of its length
# (an adult bird; a duckling, drawn when the bird is under 0.6 m, is a little lower).
# tests/test_garden_monuments.py measures them again.
ANIMAL_UNIT_LENGTH = {"bird": 0.854, "four_legged": 0.79, "turtle": 0.73}
ANIMAL_HEIGHT = {"bird": 0.795, "four_legged": 0.725, "turtle": 0.30}

DETAIL_MIN_HEIGHT = 1.6  # a figure at least this tall (a statue a walker looks up at) is drawn in full: about 330 triangles
SPRAY = "#d4e4ec"
LANTERN_STONE, LANTERN_OPENING, LANTERN_HEIGHT = "#8a857b", "#1d1d1f", 2.4
FLAG_RED, FLAG_WHITE, FLAG_BLUE = "#b22234", "#f0f0f0", "#3c3b6e"
FLAG_STRIPES, FLAG_CELLS = 7, 5  # the flag is thirteen stripes in fact; seven read the same at any distance a walker has


def _first(value) -> str:
    return str(value or "").split(";")[0].strip().lower()


def species_family(tags) -> str | None:
    """Which animal shape a statue's `animal` tag asks for: "bird", "four_legged" or "turtle"; None for anything else
    (a frog, a hare and tortoise pair), which keeps furniture.py's generic low animal."""
    animal = _first((tags or {}).get("animal"))
    if animal in BIRDS:
        return "bird"
    if animal in FOUR_LEGGED:
        return "four_legged"
    if animal in TURTLES:
        return "turtle"
    return None


def animal_extent(family: str, size, height, default: float) -> float:
    """Nose-to-tail length: the stated `length` (size), else what the stated height implies, else `default`."""
    if size:
        extent = size
    elif height:
        extent = height / ANIMAL_HEIGHT[family]
    else:
        extent = default
    return min(max(extent, 0.15), 10.0)


def animal(p, x, y, d, family: str, extent: float, colour, z0: float = 0.0) -> None:
    """A bronze animal `extent` metres from nose to tail standing at (x, y) on height z0, nose toward `d`."""
    to_world = _frame(x, y, d, z0)
    k = extent / ANIMAL_UNIT_LENGTH[family]
    bar = lambda a, b, r0, r1=None, **opt: limb(p, colour, to_world, np.asarray(a) * k, np.asarray(b) * k, r0 * k, (r0 if r1 is None else r1) * k, **opt)  # noqa: E731
    at = lambda u, v: to_world(u * k, v * k, 0.0)[:2]  # noqa: E731
    if family == "bird":
        young = extent < 0.6  # ducklings: a short neck and a big head
        head_z = 0.52 if young else 0.62
        bar((0, -0.26, 0.24), (0, 0.04, 0.25), 0.12, 0.17, sides=6, flat=0.9)  # the body, from the rump to the breast
        bar((0, 0.04, 0.25), (0, 0.20, 0.30), 0.17, 0.12, sides=6, flat=0.9)
        bar((0, -0.26, 0.26), (0, -0.44, 0.34), 0.08, 0.015, sides=4, flat=0.4)  # the tail, a little cocked
        bar((0, 0.17, 0.32), (0, 0.22, head_z - 0.03), 0.085, 0.055 if not young else 0.07, sides=5)  # neck
        bar((0, 0.21, head_z - 0.03), (0, 0.29, head_z + 0.01), 0.07 if not young else 0.085, 0.055, sides=6)  # head
        bar((0, 0.29, head_z - 0.01), (0, 0.41, head_z - 0.04), 0.04, 0.028, sides=4, flat=0.4)  # the bill
        for side in (-1, 1):
            bar((side * 0.06, 0.02, 0.15), (side * 0.06, 0.02, 0.0), 0.028, 0.02, sides=4)  # legs (upright: they end on the ground)
            fx, fy = at(side * 0.06, 0.05)
            box(p, colour, fx, fy, z0, z0 + 0.012 * k, 0.07 * k, 0.12 * k, d)  # webbed feet
    elif family == "four_legged":
        bar((0, -0.24, 0.31), (0, 0.17, 0.33), 0.12, 0.14, sides=6)  # the body
        for side in (-1, 1):
            bar((side * 0.07, -0.19, 0.28), (side * 0.07, -0.19, 0.0), 0.055, 0.035, sides=4)  # hind legs
            bar((side * 0.07, 0.14, 0.28), (side * 0.07, 0.14, 0.0), 0.05, 0.032, sides=4)  # forelegs
            bar((side * 0.045, 0.30, 0.50), (side * 0.05, 0.28, 0.57), 0.03, 0.01, sides=3)  # ears
        bar((0, 0.15, 0.36), (0, 0.27, 0.51), 0.10, 0.07, sides=5)  # neck
        bar((0, 0.26, 0.49), (0, 0.37, 0.45), 0.075, 0.045, sides=5)  # head and muzzle
        bar((0, -0.24, 0.33), (0, -0.40, 0.20), 0.03, 0.012, sides=4)  # tail
    elif family == "turtle":
        prism(p, colour, x, y, z0 + 0.06 * k, z0 + 0.14 * k, 0.30 * k, 0.22 * k, 8, rot=-d)  # the shell
        prism(p, colour, x, y, z0 + 0.14 * k, z0 + 0.22 * k, 0.22 * k, 0.07 * k, 8, rot=-d, top=True)
        bar((0, 0.18, 0.10), (0, 0.42, 0.15), 0.07, 0.05, sides=5)  # neck and head
        for sx in (-1, 1):
            for sv in (-1, 1):
                bar((sx * 0.20, sv * 0.14, 0.10), (sx * 0.30, sv * 0.20, 0.05), 0.05, 0.04, sides=4, flat=0.5)  # flippers
        box(p, colour, x, y, z0, z0 + 0.06 * k, 0.40 * k, 0.40 * k, d)  # the plastron under the shell
    else:  # pragma: no cover - callers only pass the families above
        raise ValueError(f"unknown animal family {family!r}")


# --- People --------------------------------------------------------------------------------------------------------
# Where the parts of a standing person are, as fractions of the figure's height (from the soles of the feet).
_BODY = {
    False: dict(hip=0.52, hem=0.32, shoulder=0.80, neck=0.87, head=0.075, spread=0.155, chest=(0.115, 0.145), skirt=(0.165, 0.12)),
    True: dict(hip=0.44, hem=0.30, shoulder=0.68, neck=0.74, head=0.12, spread=0.13, chest=(0.13, 0.15), skirt=(0.16, 0.13)),
}


def person(p, x, y, d, z0, fh: float, colour, *, pose: int = 0, child: bool = False) -> None:
    """A standing figure in a long coat, `fh` metres tall from the soles to the crown, facing `d`: boots, legs, a
    coat with a hem and a chest, square shoulders, arms with hands, a neck, a head with a nose and hair. `pose` (0 to
    3) chooses what the arms do: hang, hold a scroll, hold a cane, one hand on the hip. A child has a bigger head and
    shorter legs. About 330 triangles."""
    q = _BODY[child]
    to_world = _frame(x, y, d, z0)
    bar = lambda a, b, r0, r1=None, **opt: limb(p, colour, to_world, np.asarray(a) * fh, np.asarray(b) * fh, r0 * fh, (r0 if r1 is None else r1) * fh, **opt)  # noqa: E731
    prism(p, colour, x, y, z0, z0 + 0.035 * fh, 0.17 * fh, sides=6, top=True)  # the bronze's own base
    hip, hem, sh, ao = q["hip"], q["hem"], q["shoulder"], q["spread"]
    for side in (-1, 1):
        bar((side * 0.05, 0, hip), (side * 0.055, 0.005, 0.085), 0.062, 0.04, sides=5)  # leg
        bar((side * 0.055, 0.0, 0.09), (side * 0.058, 0.075, 0.045), 0.042, 0.035, sides=5, flat=0.8)  # boot
    bar((0, 0, hem), (0, 0, hip + 0.04), q["skirt"][0], q["skirt"][1], sides=8, flat=0.7)  # the coat's skirt
    bar((0, 0, hip), (0, 0, sh), q["chest"][0], q["chest"][1], sides=8, flat=0.6)  # chest
    bar((-ao, 0, sh - 0.015), (ao, 0, sh - 0.015), 0.052, sides=6)  # shoulders
    for side in (-1, 1):
        elbow = (side * (ao + 0.015), 0.005, sh - 0.22)
        hand = (side * (ao + 0.005), 0.03, sh - 0.40)
        if side == 1 and pose == 1:  # the right forearm forward, holding a scroll
            elbow, hand = (0.17, 0.03, sh - 0.20), (0.12, 0.115, sh - 0.20)
            bar((0.03, 0.115, sh - 0.19), (0.17, 0.115, sh - 0.21), 0.022, sides=5)
        elif side == 1 and pose == 2:  # the right hand forward on a cane
            elbow, hand = (0.17, 0.02, sh - 0.22), (0.15, 0.085, sh - 0.38)
            bar((0.145, 0.087, sh - 0.37), (0.12, 0.095, 0.036), 0.012, sides=4)  # the cane reaches the bronze's base
        elif side == -1 and pose == 3:  # the left hand on the hip
            elbow, hand = (-0.20, -0.01, sh - 0.20), (-0.115, 0.02, sh - 0.30)
        bar((side * ao, 0, sh - 0.015), elbow, 0.05, 0.042, sides=5)  # upper arm
        bar(elbow, hand, 0.042, 0.034, sides=5)  # forearm
        bar(hand, (hand[0], hand[1] + 0.012, hand[2] - 0.055), 0.032, 0.028, sides=5)  # the hand
    top, head = 1.0, q["head"]
    bar((0, 0, sh), (0, 0, q["neck"] + 0.01), 0.042, sides=5)  # neck
    bar((0, 0, top - 2 * head - 0.005), (0, 0, top), head * 0.78, head * 0.68, sides=6, flat=1.05)  # head
    bar((0, 0.5 * head, top - 1.4 * head), (0, 1.0 * head, top - 1.5 * head), 0.2 * head, 0.1 * head, sides=4)  # nose
    bar((0, -0.15 * head, top - 0.55 * head), (0, -0.15 * head, top), 0.9 * head, 0.65 * head, sides=6)  # hair


def lying_figure(p, x, y, d, z0, fh: float, colour) -> None:
    """A figure lying propped on the pedestal in front of a standing one (the second figure of a group): a body, legs,
    a head and an arm reaching up. `fh` is the standing figure's height. About 100 triangles."""
    to_world = _frame(x, y, d, z0)
    bar = lambda a, b, r0, r1=None, **opt: limb(p, colour, to_world, np.asarray(a) * fh, np.asarray(b) * fh, r0 * fh, (r0 if r1 is None else r1) * fh, **opt)  # noqa: E731
    bar((-0.26, 0.26, 0.09), (0.10, 0.26, 0.13), 0.10, 0.11, sides=6, flat=0.7)  # the body
    bar((-0.31, 0.26, 0.11), (-0.39, 0.26, 0.14), 0.06, 0.055, sides=6)  # the head
    bar((0.10, 0.24, 0.12), (0.42, 0.29, 0.07), 0.06, 0.045, sides=5)  # legs
    bar((0.10, 0.29, 0.12), (0.38, 0.36, 0.09), 0.06, 0.045, sides=5)
    bar((-0.20, 0.25, 0.15), (-0.10, 0.16, 0.30), 0.04, 0.032, sides=5)  # an arm reaching up to the standing one
    bar((-0.18, 0.31, 0.13), (-0.04, 0.38, 0.07), 0.04, 0.032, sides=5)


# --- Pedestals -----------------------------------------------------------------------------------------------------
# A bell: (radius, height) as fractions of the footing's radius and of the pedestal's height, from the ground up.
BELL = (
    (1.00, 0.00), (1.00, 0.07), (0.90, 0.07), (0.90, 0.12), (0.86, 0.16), (0.75, 0.26), (0.66, 0.38),
    (0.61, 0.52), (0.59, 0.68), (0.59, 0.78), (0.70, 0.83), (0.72, 0.88), (0.72, 1.00),
)


def bell_pedestal(p, x, y, height: float, radius: float, shade: float = 1.0, sides: int = 14) -> float:
    """A round bell-shaped pedestal in warm grey granite: a footing, a ledge, a flaring bell, a neck and a rounded
    cap. `radius` is the footing's. Returns the height it ends at. About 340 triangles."""
    for i, ((r0, z0), (r1, z1)) in enumerate(zip(BELL, BELL[1:])):
        stone = colour_bytes(PLINTH_STONE, shade * (0.92 + 0.16 * (z0 + z1) / 2))  # darker at the foot, paler at the cap
        if z1 - z0 < 1e-9:
            annulus(p, stone, x, y, z0 * height, r0 * radius, r1 * radius, sides)
        else:
            prism(p, stone, x, y, z0 * height, z1 * height, r0 * radius, r1 * radius, sides)
    disc(p, colour_bytes(PLINTH_STONE, shade * 1.08), x, y, height, BELL[-1][0] * radius, sides)
    return height


# A shaft's courses: (from, to as fractions of its height, footprint relative to its width, shade).
SHAFT_COURSES = (
    (0.000, 0.030, 1.34, 0.90), (0.030, 0.060, 1.18, 0.94), (0.060, 0.190, 1.00, 0.98),  # two steps and a base
    (0.500, 0.525, 0.86, 1.04),  # a band round the shaft
    (0.860, 0.900, 0.90, 1.05), (0.900, 0.950, 0.82, 1.08), (0.950, 1.000, 0.94, 1.10),  # cornice and a cap
)


def shaft_pedestal(p, x, y, d, height: float, width: float, shade: float = 1.0) -> float:
    """A tall granite pedestal for a group: steps and a base, a long square shaft that tapers a little, a band, a
    cornice and a cap slab. `width` is the base's. Returns the height it ends at. About 80 triangles."""
    for lo, hi, scale, tone in SHAFT_COURSES:
        box(p, colour_bytes(PLINTH_STONE, shade * tone), x, y, lo * height, hi * height, width * scale, width * scale, d)
    root = math.sqrt(2) / 2
    prism(p, colour_bytes(PLINTH_STONE, shade), x, y, 0.19 * height, 0.86 * height, 0.80 * width * root, 0.68 * width * root, 4, _square_rot(d))
    return height


# --- A stone lantern -----------------------------------------------------------------------------------------------
def lantern(p, x, y, d, o) -> None:
    """A Japanese stone lantern: a wide base, a slender shaft, a flared seat, a fire box with four openings, a broad
    roof and a finial. Stone grey unless the data gives a colour. About 130 triangles."""
    height = o.tall(LANTERN_HEIGHT, 0.8, 8.0)
    stone = lambda tone=1.0: o.paint(LANTERN_STONE) if o.colour else colour_bytes(LANTERN_STONE, o.shade * tone)  # noqa: E731
    rot = 67.5 - d  # an octagon with a flat face toward `d`
    prism(p, stone(0.9), x, y, 0.0, 0.05 * height, 0.22 * height, 0.19 * height, 8, rot, top=True)  # the base
    prism(p, stone(1.0), x, y, 0.05 * height, 0.35 * height, 0.060 * height, 0.050 * height, 8, rot)  # the shaft
    prism(p, stone(1.04), x, y, 0.35 * height, 0.41 * height, 0.050 * height, 0.175 * height, 8, rot, top=True)  # the seat
    prism(p, stone(1.0), x, y, 0.41 * height, 0.61 * height, 0.118 * height, sides=8, rot=rot)  # the fire box
    apothem = 0.118 * height * math.cos(math.pi / 8)
    for k in range(4):  # the openings
        heading = d + 90 * k
        ox, oy = x + (apothem + 0.003 * height) * math.sin(math.radians(heading)), y + (apothem + 0.003 * height) * math.cos(math.radians(heading))
        box(p, colour_bytes(LANTERN_OPENING), ox, oy, 0.46 * height, 0.57 * height, 0.07 * height, 0.012 * height, heading)
    prism(p, stone(1.08), x, y, 0.61 * height, 0.66 * height, 0.31 * height, 0.27 * height, 8, rot)  # the roof's edge
    pyramid(p, stone(1.1), x, y, 0.66 * height, 0.78 * height, 0.27 * height, 8, rot)
    prism(p, stone(1.0), x, y, 0.78 * height, 0.955 * height, 0.028 * height, 0.02 * height, 6)  # the finial's stem
    octahedron(p, stone(1.1), x, y, 0.955 * height, 0.045 * height)


# --- Water ---------------------------------------------------------------------------------------------------------
def spray(p, x, y, z: float, height: float, spread: float) -> None:
    """A fountain's jets from (x, y, z): one straight up and a ring leaning outward, in pale water colour. About 70
    triangles."""
    water = colour_bytes(SPRAY)
    prism(p, water, x, y, z, z + height, 0.03 * height, 0.012 * height, 6)
    to_world = _frame(x, y, 0.0, z)
    for k in range(5):
        angle = 2 * math.pi * k / 5
        a, b = (0.0, 0.0, 0.0), (spread * math.sin(angle), spread * math.cos(angle), 0.8 * height)
        limb(p, water, to_world, np.array(a), np.array(b), 0.025 * height, 0.012 * height, sides=4)  # (u east, v north, z up)


def fountain_figures(p, x, y, d, o, top: float, radius: float, stone, colour) -> None:
    """The middle of a fountain that has a bronze figure: a stone pier with the figure (or two or three round it, or
    an animal) on top, `top` metres in all. Only drawn when the data says what the figure is (`figure` or `animal`)."""
    tags = o.tags or {}
    pier = max(0.35, 0.50 * top)
    family = species_family(tags)
    pier_radius = max(0.18, 0.12 * radius) * (1.6 if family else 1.2)
    prism(p, stone, x, y, 0.0, pier, pier_radius * 1.15, pier_radius * 0.85, 8, top=True)
    figure_height = top - pier
    if family:
        animal(p, x, y, d, family, figure_height / ANIMAL_HEIGHT[family], colour, z0=pier)
        return
    count = int(min(max(number(tags.get("figures"), 1), 1), 3))
    child = _first(tags.get("figure")) == "child"
    if count == 1:
        person(p, x, y, d, pier, figure_height, colour, pose=o.seed % 4, child=child)
        return
    for k in range(count):  # a ring of figures, each looking outward
        angle = d + 360.0 * k / count
        fx, fy = x + 0.55 * pier_radius * math.sin(math.radians(angle)), y + 0.55 * pier_radius * math.cos(math.radians(angle))
        person(p, fx, fy, angle, pier, figure_height, colour, pose=(o.seed + k) % 4, child=child)


def number(value, default: float) -> float:
    """A tag's first number ("2", "2;3"), else `default`."""
    try:
        return float(str(value).split(";")[0])
    except (TypeError, ValueError):
        return default


def is_sculpted_fountain(tags) -> bool:
    """A fountain whose data says it has a figure: a person (`figure`) or an animal (`animal`)."""
    tags = tags or {}
    return bool(tags.get("figure") or species_family(tags))


# --- A memorial that flies a flag ----------------------------------------------------------------------------------
def flag(p, x, y, d, top: float) -> None:
    """The United States flag hoisted at the top of a pole at (x, y) of height `top`, streaming toward `d`, with a
    little wave in it, drawn on both faces. About 140 triangles."""
    length = min(max(0.2 * top, 0.7), 3.2)
    width = length / 1.9
    stripe = width / FLAG_STRIPES
    canton_rows, canton_cells = 4, 2
    top_edge = top - 0.35 * width
    to_world = _frame(x, y, d, 0.0)

    def point(s, z):  # s along the flag from the pole (0 to 1), z the height
        wave = 0.10 * length * s * math.sin(2.2 * math.pi * s)
        return to_world(wave, 0.06 + s * length, z)

    for i in range(FLAG_CELLS):
        s0, s1 = i / FLAG_CELLS, (i + 1) / FLAG_CELLS
        for j in range(FLAG_STRIPES):
            z1, z0 = top_edge - j * stripe, top_edge - (j + 1) * stripe
            if i < canton_cells and j < canton_rows:
                colour = FLAG_BLUE
            else:
                colour = FLAG_RED if j % 2 == 0 else FLAG_WHITE
            corners = np.array([point(s0, z0), point(s1, z0), point(s1, z1), point(s0, z1)])
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            normal /= np.linalg.norm(normal)
            paint = colour_bytes(colour)
            p.add(corners, normal, [0, 1, 2, 0, 2, 3], paint)
            p.add(corners[::-1], -normal, [0, 1, 2, 0, 2, 3], paint)  # the back, wound the other way


def flag_memorial(p, x, y, d, o, top: float) -> None:
    """A granite memorial that is a flagpole: two steps, a short round column with a square capital, then the pole,
    a gilt ball and the flag. `top` is the whole height."""
    stone = lambda tone: colour_bytes(PLINTH_STONE, o.shade * tone)  # noqa: E731
    box(p, stone(0.92), x, y, 0.0, 0.15, 1.7, 1.7, d)
    box(p, stone(0.96), x, y, 0.15, 0.30, 1.35, 1.35, d)
    prism(p, stone(1.0), x, y, 0.30, 2.0, 0.34, 0.30, 10, top=False)
    box(p, stone(1.08), x, y, 2.0, 2.2, 0.8, 0.8, d)
    ball = 0.1
    prism(p, colour_bytes("#d8d8d4", o.shade), x, y, 2.2, top - ball, 0.07, 0.035, 8)
    octahedron(p, colour_bytes("#c9a24a"), x, y, top - ball, ball)
    flag(p, x, y, d, top)

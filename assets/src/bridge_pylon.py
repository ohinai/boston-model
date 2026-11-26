"""One granite pylon with its globe lamp, as stands at each end of the Public Garden's suspension footbridge.

z = 0 is the top surface of the deck; the dark pier below it stays procedural (pipeline/boston_pipeline/bridges.py).
The pylon is square in plan and the same on all four faces, so it can be turned to the bridge's direction. Stepped
from the bottom: a broad base with a moulding, a plain die with a recessed panel on each face, a projecting cornice,
a hipped capstone, then an iron lamp post that carries a white globe whose top is at 4.9 m (the lamp's height in the
data). The die is 1.1 m wide and the cables run into it 2.1 m up, where the pipeline strings them; the base is
1.5 m wide, as wide as the procedural plinth it replaces.

    python assets/build.py bridge-pylon
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c

c.reset()

HEIGHT = 4.9
ROOT2 = math.sqrt(2.0)

# Granite in a few tones so the stepped form reads: the base darkest, the die lightest, the capstone weathered.
BASE = "#7d7a74"
MOULDING = "#8d8a83"
DIE = "#b4b0a6"
PANEL = "#a09c92"
CORNICE = "#aaa69c"
CAP = "#928f87"
IRON = "#9cc8c3"  # the same pale blue-green paint as the bridge's railings (IRONWORK in bridges.py)
GLOBE = "#f3f1ea"

DIE_HALF = 0.55  # a 1.1 m shaft
DIE_FOOT, DIE_HEAD = 0.55, 1.92
PANEL_HALF, PANEL_FOOT, PANEL_HEAD, PANEL_DEPTH = 0.38, 0.85, 1.74, 0.05
GLOBE_RADIUS = 0.27
GLOBE_CENTRE = HEIGHT - GLOBE_RADIUS


def tower(name, colour, profile):
    """A stepped square tower from a profile of (half side, z) rings, bottom up; a half side of 0 is a point."""
    rings = [(half * ROOT2, z) for half, z in profile]
    return c.lathe(name, colour, rings, sides=4, rotate=45)


def welded(name, colour, polygons):
    """A mesh from polygons given as lists of (x, y, z) corners, sharing every corner that has the same position."""
    index, verts, faces = {}, [], []
    for polygon in polygons:
        face = []
        for corner in polygon:
            key = tuple(round(v, 6) for v in corner)
            if key not in index:
                index[key] = len(verts)
                verts.append(key)
            face.append(index[key])
        faces.append(tuple(face))
    return c.part(name, verts, faces, colour)


def turned(k, u, z, n):
    """Face-local (u along the face, z up, n out of the face) turned a quarter turn k times about the vertical."""
    angle = math.radians(90 * k)
    return (u * math.cos(angle) - n * math.sin(angle), u * math.sin(angle) + n * math.cos(angle), z)


def die():
    """The shaft: a square block with a recessed panel on each face, closed all round."""
    h, z0, z1 = DIE_HALF, DIE_FOOT, DIE_HEAD
    p, za, zb, back = PANEL_HALF, PANEL_FOOT, PANEL_HEAD, DIE_HALF - PANEL_DEPTH
    polygons = []
    for k in range(4):
        def at(u, z, n=h):
            return turned(k, u, z, n)

        polygons += [
            [at(-h, z0), at(h, z0), at(h, za), at(p, za), at(-p, za), at(-h, za)],  # the frame, below the panel
            [at(-h, zb), at(-p, zb), at(p, zb), at(h, zb), at(h, z1), at(-h, z1)],  # above it
            [at(-h, za), at(-p, za), at(-p, zb), at(-h, zb)],  # to its left
            [at(p, za), at(h, za), at(h, zb), at(p, zb)],  # to its right
            [at(-p, za), at(p, za), at(p, za, back), at(-p, za, back)],  # the reveal: sill
            [at(-p, zb), at(-p, zb, back), at(p, zb, back), at(p, zb)],  # lintel
            [at(-p, za), at(-p, za, back), at(-p, zb, back), at(-p, zb)],  # left jamb
            [at(p, za), at(p, zb), at(p, zb, back), at(p, za, back)],  # right jamb
        ]  # no floor to the panels: panels() fills each recess, so nothing is drawn twice there
    polygons += [
        [(-h, -h, z1), (h, -h, z1), (h, h, z1), (-h, h, z1)],
        [(-h, -h, z0), (-h, h, z0), (h, h, z0), (h, -h, z0)],
    ]
    return welded("die", DIE, polygons)


def panels():
    """The floor of each sunk panel, in a darker tone so the recess reads from far off. A slab whose front face is the
    recess's floor and whose edges reach 1 cm into the stone around it, so there is no crack and no second surface to
    fight with."""
    p, za, zb = PANEL_HALF + 0.01, PANEL_FOOT - 0.01, PANEL_HEAD + 0.01
    front = DIE_HALF - PANEL_DEPTH
    back = front - 0.04
    c.box("panel_n", PANEL, -p, p, back, front, za, zb)
    c.box("panel_s", PANEL, -p, p, -front, -back, za, zb)
    c.box("panel_e", PANEL, back, front, -p, p, za, zb)
    c.box("panel_w", PANEL, -front, -back, -p, p, za, zb)


# --- Base: a broad socle, then a moulding that steps in to the die.
tower("base", BASE, [(0.75, 0.0), (0.75, 0.30), (0.69, 0.30), (0.69, 0.38), (0.61, 0.50), (0.61, DIE_FOOT)])
die()
panels()

# --- Cornice: a neck, a cove that flares out, a plain fascia, then steps back to the capstone.
tower("cornice", CORNICE, [
    (0.60, DIE_HEAD), (0.60, 1.98), (0.62, 2.04), (0.67, 2.10), (0.72, 2.15), (0.72, 2.30), (0.66, 2.30), (0.66, 2.37),
])

# --- Capstone: a hipped roof, steep at the top, finishing in a small flat for the lamp.
tower("cap", CAP, [(0.64, 2.37), (0.57, 2.45), (0.47, 2.57), (0.37, 2.69), (0.28, 2.80), (0.26, 2.84), (0.20, 2.90)])

# --- Lamp post: a flared foot, collars, then a cup that holds the globe.
c.lathe("lamp", IRON, [
    (0.15, 2.88), (0.15, 2.94), (0.09, 3.02), (0.05, 3.14),
    (0.05, 3.50), (0.08, 3.50), (0.08, 3.56), (0.05, 3.56),
    (0.05, 4.08), (0.08, 4.08), (0.08, 4.14), (0.05, 4.14),
    (0.05, 4.20), (0.09, 4.27), (0.13, 4.41),
], sides=8)

# --- The globe: a sphere, lower pole hidden in the cup. Its top is the top of the lamp.
rings = []
for degrees in (-60.0, -37.5, -15.0, 7.5, 30.0, 52.5, 75.0):
    angle = math.radians(degrees)
    rings.append((GLOBE_RADIUS * math.cos(angle), GLOBE_CENTRE + GLOBE_RADIUS * math.sin(angle)))
rings.append((0.0, HEIGHT))
c.lathe("globe", GLOBE, rings, sides=16)

c.finish(HEIGHT)

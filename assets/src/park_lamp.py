"""The park street lamp: a slim black cast-iron post on a flared foot, a collar, a round opal globe and a small cap.

Modelled for `highway=street_lamp` points that are not `lamp_mount=arm`: the lamps along the walks of the Public Garden,
the Common and the Commonwealth Avenue Mall. The pipeline draws about nine hundred of these, so the budget is tight
(110 triangles; this uses 94). The viewer shades everything but trees flat, whatever normals a file carries, so a curve
has to be made of facets: the shaft and collar are hexagons, the foot and globe octagons, and the globe, which is what
you see from forty metres, gets three bands spread over the part of the ball that shows (the collar hides its bottom,
the cap its top).

    python assets/build.py park-lamp

Modelled at the typical 4.9 m (the data's `height` for most of them), z up, origin on the ground at the centre of the
foot. The lamp is the same from every side, so which way it faces does not matter. There are two colours: the black of
the iron, and the opal of the globe (the pipeline repaints the iron where the data gives the lamp a colour).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

c.reset()

HEIGHT = 4.9
POST, GLASS = "#1c1e20", "#efece0"

# Profiles are (radius, z) from the bottom up, in metres.
FOOT_TOP = 0.95  # where the flared foot meets the shaft
# The foot is one straight band. A second ring at (0.150, 0.20) makes it a bell for 16 more triangles, which on the
# ground at walking distance cannot be told from the cone (compared in the viewer at 5 to 15 m, 94 against 110).
FOOT = [(0.205, 0.0), (0.078, FOOT_TOP)]
SHAFT = [(0.072, FOOT_TOP - 0.02), (0.052, 4.32)]
COLLAR = [(0.058, 4.30), (0.140, 4.465)]  # the cup the globe sits in; its rim clears the globe, so nothing overlaps

# The globe is an egg of revolution (0.2 m wide from the axis, 0.22 m above and below its middle). Only the part from
# the collar to the cap shows, so its four rings are spread over that, about 34 degrees apart.
GLOBE_MID, GLOBE_RX, GLOBE_RZ = 4.64, 0.20, 0.22
GLOBE_LATITUDES = (-57.0, -23.0, 11.0, 45.0)
GLOBE = [
    (GLOBE_RX * math.cos(math.radians(a)), GLOBE_MID + GLOBE_RZ * math.sin(math.radians(a))) for a in GLOBE_LATITUDES
]
CAP = [(GLOBE[-1][0] + 0.035, GLOBE[-1][1] - 0.004), (0.0, HEIGHT)]  # a small pointed hat; its brim clears the globe


def lathe(name, colour, profile, sides, rotate=0.0):
    """A solid of revolution with open ends: the foot stands on the ground, and the collar, globe and cap hide each
    other's ends, so a flat cap would be 4 to 6 triangles nobody sees. Flat shaded, as the viewer draws it."""
    verts, faces, rings = [], [], []
    for radius, z in profile:
        if radius <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
            continue
        ring = []
        for k in range(sides):
            angle = math.radians(rotate) + 2 * math.pi * k / sides
            ring.append(len(verts))
            verts.append((radius * math.cos(angle), radius * math.sin(angle), z))
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for k in range(sides):
            k2 = (k + 1) % sides
            if len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
    return c.part(name, verts, faces, colour)


lathe("foot", POST, FOOT, 8, rotate=22.5)
lathe("shaft", POST, SHAFT, 6)
lathe("collar", POST, COLLAR, 6)
lathe("globe", GLASS, GLOBE, 8, rotate=22.5)
lathe("cap", POST, CAP, 6)


class _KeepWinding:
    """`c.finish` recalculates every face's normal, which is right for a closed solid and a coin toss for the open
    surfaces made above (it turned the collar inside out, so the viewer, which draws only front faces, showed its far
    wall). `lathe` already winds every face outward, so here the recalculation is switched off."""

    def __init__(self, real):
        self._real = real
        self.ops = type("ops", (), {"recalc_face_normals": staticmethod(lambda bm, faces: None)})

    def __getattr__(self, name):
        return getattr(self._real, name)


c.bmesh = _KeepWinding(c.bmesh)
c.finish(HEIGHT)

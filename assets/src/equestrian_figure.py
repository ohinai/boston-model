"""A bronze horse and rider, without the pedestal: the figure of `statue=equestrian` (Washington, Hooker, Revere).

Modelled to look like a heroic equestrian bronze: a horse standing on three legs with one foreleg raised, an arched
neck and a flowing tail, and a rider sitting upright in a cocked hat, one arm held out, a sword at his hip. The
proportions are a life-size horse (withers 1.6 m) and rider enlarged to the statue's height; they are not any
particular statue's, and nothing here is traced from a photograph.

How it is made (all in this script): a "tube" is a smooth solid swept along a spline, with an elliptical section that
changes along the way. Limbs, the neck, the body, the tail and the rider's arms and legs are tubes; the head, the
hat's crown and the saddle cloth are blobs (ellipsoids); hooves are lathes. Each part is one flat colour, then every
vertex is darkened or lightened a little by the way it faces (up is lighter) so the form reads in the viewer's
single flat-coloured material.

    python assets/build.py equestrian-figure
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

c.reset()

HEIGHT = 3.5  # metres, hat to hoof: what the sidecar says
DENSITY = 1.0  # raise or lower to trade triangles for smoothness (rings per metre scale with it)

# --- Colours (sRGB): dark bronze with a few lighter and darker tones so the form reads -----------------------------
BRONZE = "#3a3228"  # the horse's coat
BRONZE_DARK = "#1f1d18"  # mane, tail, hooves
COAT = "#43392b"  # the rider's coat
BREECHES = "#4e4532"
BOOT = "#1d1b17"
SKIN = "#6d5f47"  # face and hands: the polished parts
HAT = "#322e25"
CLOTH = "#54492f"  # saddle cloth
CLOAK = "#2f2c23"
STEEL = "#6c674f"  # sword, epaulettes

STATS: list[tuple[str, int]] = []


# --- Shapes --------------------------------------------------------------------------------------------------------
def spline(points: list, samples: int) -> list:
    """A Catmull-Rom curve through `points` (tuples of floats, any length), sampled at `samples` places."""
    n = len(points)
    ext = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(samples):
        t = i / (samples - 1) * (n - 1)
        k = min(int(t), n - 2)
        u = t - k
        p0, p1, p2, p3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        out.append(tuple(
            0.5 * (2 * b + (-a + c2) * u + (2 * a - 5 * b + 4 * c2 - d) * u * u + (-a + 3 * b - 3 * c2 + d) * u ** 3)
            for a, b, c2, d in zip(p0, p1, p2, p3)
        ))
    return out


def tube(name, colour, ctrl, sides=10, spacing=0.08, hint=(1, 0, 0), caps=(1, 1), shade=1.0):
    """A solid swept along a spline. `ctrl` is a list of (x, y, z, a, b): a point on the axis and the half-width
    of the section there along `hint` (a) and across it (b). `caps` is the number of rings rounding each end."""
    spacing /= DENSITY
    length = sum(math.dist(p[:3], q[:3]) for p, q in zip(ctrl, ctrl[1:]))
    samples = max(len(ctrl), round(length / spacing) + 1)
    path = spline([tuple(p) for p in ctrl], samples)
    centres = [Vector(q[:3]) for q in path]
    radii = [(max(q[3], 1e-3), max(q[4], 1e-3)) for q in path]
    hint = Vector(hint).normalized()
    angles = [2 * math.pi * k / sides for k in range(sides)]
    verts, faces = [], []

    def ring(centre, e1, e2, a, b):
        start = len(verts)
        for th in angles:
            verts.append(centre + e1 * (a * math.cos(th)) + e2 * (b * math.sin(th)))
        return list(range(start, start + sides))

    frames = []
    for i, centre in enumerate(centres):
        tangent = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)]).normalized()
        e1 = hint - tangent * hint.dot(tangent)
        if e1.length < 1e-3:
            e1 = Vector((0, 1, 0)) - tangent * tangent.y
        e1.normalize()
        frames.append((tangent, e1, tangent.cross(e1)))

    rings = []
    first_tangent, e1, e2 = frames[0]
    a0, b0 = radii[0]
    r0 = (a0 + b0) / 2
    pole_start = None
    if caps[0]:
        pole_start = len(verts)
        verts.append(centres[0] - first_tangent * r0)
        for j in range(caps[0], 0, -1):
            phi = j / (caps[0] + 1) * math.pi / 2
            rings.append(ring(centres[0] - first_tangent * (r0 * math.sin(phi)), e1, e2, a0 * math.cos(phi), b0 * math.cos(phi)))
    for (tangent, e1, e2), centre, (a, b) in zip(frames, centres, radii):
        rings.append(ring(centre, e1, e2, a, b))
    last_tangent, e1, e2 = frames[-1]
    a1, b1 = radii[-1]
    r1 = (a1 + b1) / 2
    if caps[1]:
        for j in range(1, caps[1] + 1):
            phi = j / (caps[1] + 1) * math.pi / 2
            rings.append(ring(centres[-1] + last_tangent * (r1 * math.sin(phi)), e1, e2, a1 * math.cos(phi), b1 * math.cos(phi)))
    for lo, hi in zip(rings, rings[1:]):
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((lo[k], lo[k2], hi[k2], hi[k]))
    if pole_start is not None:
        for k in range(sides):
            faces.append((pole_start, rings[0][(k + 1) % sides], rings[0][k]))
    else:
        faces.append(tuple(reversed(rings[0])))
    if caps[1]:
        pole = len(verts)
        verts.append(centres[-1] + last_tangent * r1)
        for k in range(sides):
            faces.append((rings[-1][k], rings[-1][(k + 1) % sides], pole))
    else:
        faces.append(tuple(rings[-1]))
    return _solid(name, verts, faces, colour, shade)


def blob(name, colour, centre, radii, sides=10, rings=6, yaw=0.0, pitch=0.0, shade=1.0):
    """An ellipsoid with half-sizes `radii` (x, y, z), pitched (about x, degrees, nose up positive) then turned
    clockwise seen from above by `yaw`."""
    rotation = Matrix.Rotation(math.radians(-yaw), 3, "Z") @ Matrix.Rotation(math.radians(pitch), 3, "X")
    verts, faces = [], []
    bottom = len(verts)
    verts.append(Vector(centre) + rotation @ Vector((0, 0, -radii[2])))
    ring_ids = []
    for j in range(1, rings):
        theta = math.pi * j / rings
        ids = []
        for k in range(sides):
            phi = 2 * math.pi * k / sides
            local = Vector((radii[0] * math.sin(theta) * math.cos(phi), radii[1] * math.sin(theta) * math.sin(phi), -radii[2] * math.cos(theta)))
            ids.append(len(verts))
            verts.append(Vector(centre) + rotation @ local)
        ring_ids.append(ids)
    top = len(verts)
    verts.append(Vector(centre) + rotation @ Vector((0, 0, radii[2])))
    for k in range(sides):
        faces.append((bottom, ring_ids[0][(k + 1) % sides], ring_ids[0][k]))
        faces.append((ring_ids[-1][k], ring_ids[-1][(k + 1) % sides], top))
    for lo, hi in zip(ring_ids, ring_ids[1:]):
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((lo[k], lo[k2], hi[k2], hi[k]))
    return _solid(name, verts, faces, colour, shade)


def _solid(name, verts, faces, colour, shade):
    obj = c.part(name, [tuple(v) for v in verts], faces, colour, shade)
    STATS.append((name, sum(len(f) - 2 for f in faces)))
    return obj


def leg(name, colour, ctrl, hoof_at, hoof_lift=0.0, sides=8, spacing=0.14, hoof_radius=0.075, hoof_height=0.09, squash=1.15):
    """A leg as a tube from the top down, and a hoof under it."""
    tube(name, colour, ctrl, sides=sides, spacing=spacing, caps=(1, 0))
    r = hoof_radius
    profile = [(r, hoof_lift), (r * 0.82, hoof_lift + hoof_height * 0.8), (r * 0.74, hoof_lift + hoof_height)]
    c.lathe(name + "_hoof", BRONZE_DARK, profile, sides=6, centre=hoof_at, squash=squash)
    STATS.append((name + "_hoof", 6 * 2 * 2 + 8))


def mirror(ctrl, dy=0.0):
    return [(-x, y + dy, z, a, b) for x, y, z, a, b in ctrl]


# --- The horse (facing +Y, its right side is +X; metres, life size) ----------------------------------------------------
body = [
    (0, -0.62, 1.18, 0.21, 0.32),
    (0, -0.42, 1.22, 0.285, 0.37),
    (0, -0.15, 1.16, 0.27, 0.40),
    (0, 0.15, 1.13, 0.27, 0.44),
    (0, 0.40, 1.14, 0.255, 0.46),
    (0, 0.58, 1.09, 0.215, 0.39),
]
tube("body", BRONZE, body, sides=14, spacing=0.12, caps=(1, 1))

neck = [
    (0, 0.42, 1.30, 0.20, 0.31),
    (0, 0.60, 1.58, 0.16, 0.27),
    (0, 0.74, 1.84, 0.135, 0.23),
    (-0.01, 0.90, 2.06, 0.115, 0.20),
    (-0.03, 1.04, 2.17, 0.105, 0.18),
]
tube("neck", BRONZE, neck, sides=12, spacing=0.15, caps=(0, 1))

head = [  # turned a little to the horse's left, towards the raised foreleg
    (-0.04, 1.05, 2.15, 0.095, 0.115),
    (-0.06, 1.14, 1.95, 0.100, 0.125),
    (-0.09, 1.21, 1.77, 0.070, 0.090),
    (-0.11, 1.25, 1.62, 0.052, 0.058),
]
tube("head", BRONZE, head, sides=10, spacing=0.12, caps=(1, 1))
for side in (-1, 1):
    ear = [(side * 0.06 - 0.05, 1.01, 2.17, 0.032, 0.017), (side * 0.066 - 0.05, 1.02, 2.30, 0.026, 0.015), (side * 0.066 - 0.05, 1.04, 2.42, 0.008, 0.006)]
    tube(f"ear{side}", BRONZE, ear, sides=4, spacing=0.14, caps=(0, 1))

mane = [
    (0, 0.30, 1.52, 0.045, 0.08),
    (0, 0.56, 1.88, 0.05, 0.085),
    (-0.01, 0.82, 2.18, 0.05, 0.075),
    (-0.03, 1.00, 2.30, 0.045, 0.06),
    (-0.05, 1.10, 2.18, 0.03, 0.04),
]
tube("mane", BRONZE_DARK, mane, sides=6, spacing=0.18, caps=(1, 1))

tail = [
    (0, -0.80, 1.44, 0.06, 0.07),
    (0, -0.98, 1.36, 0.10, 0.10),
    (0, -1.09, 1.10, 0.115, 0.115),
    (0, -1.12, 0.80, 0.10, 0.105),
    (0, -1.07, 0.56, 0.06, 0.07),
    (0, -1.03, 0.42, 0.02, 0.025),
]
tube("tail", BRONZE_DARK, tail, sides=8, spacing=0.16, caps=(0, 1))

hind = [
    (0.18, -0.45, 1.15, 0.14, 0.21),
    (0.19, -0.48, 0.90, 0.12, 0.19),
    (0.18, -0.52, 0.72, 0.09, 0.14),
    (0.165, -0.66, 0.48, 0.065, 0.105),
    (0.16, -0.64, 0.28, 0.050, 0.060),
    (0.16, -0.60, 0.12, 0.055, 0.065),
    (0.16, -0.56, 0.08, 0.050, 0.060),
]
leg("hind_r", BRONZE, hind, hoof_at=(0.16, -0.55))
leg("hind_l", BRONZE, mirror(hind, dy=0.10), hoof_at=(-0.16, -0.45))

fore = [
    (0.16, 0.38, 1.05, 0.125, 0.18),
    (0.16, 0.46, 0.76, 0.095, 0.13),
    (0.155, 0.50, 0.48, 0.072, 0.092),
    (0.15, 0.50, 0.28, 0.050, 0.055),
    (0.15, 0.52, 0.13, 0.055, 0.060),
    (0.15, 0.56, 0.08, 0.050, 0.055),
]
leg("fore_r", BRONZE, fore, hoof_at=(0.15, 0.57))

raised = [  # the left foreleg, lifted: the forearm forward and level, the cannon hanging from the knee
    (-0.16, 0.38, 1.05, 0.125, 0.18),
    (-0.16, 0.55, 0.90, 0.100, 0.13),
    (-0.16, 0.78, 0.80, 0.075, 0.095),
    (-0.16, 0.82, 0.68, 0.055, 0.065),
    (-0.16, 0.82, 0.50, 0.050, 0.055),
    (-0.16, 0.80, 0.38, 0.055, 0.060),
    (-0.16, 0.80, 0.325, 0.050, 0.055),
]
leg("fore_l", BRONZE, raised, hoof_at=(-0.16, 0.81), hoof_lift=0.24)

# --- The rider ------------------------------------------------------------------------------------------------------------
# Everything of his is modelled life size, then enlarged about his seat so a statue's rider is a little heroic.
y0, SEAT, K = -0.02, 1.52, 1.10


def rs(ctrl):  # rider's control points (x, y, z, a, b) -> enlarged
    return [(x * K, y0 + (y - y0) * K, SEAT + (z - SEAT) * K, a * K, b * K) for x, y, z, a, b in ctrl]


def rp(x, y, z):
    return (x * K, y0 + (y - y0) * K, SEAT + (z - SEAT) * K)


blob("saddle_cloth", CLOTH, (0, y0, 1.42), (0.30, 0.42, 0.16), sides=12, rings=5)
torso = [
    (0, y0, 1.58, 0.20, 0.14),
    (0, y0 + 0.01, 1.84, 0.225, 0.15),
    (0, y0, 2.04, 0.26, 0.13),
    (0, y0 - 0.01, 2.13, 0.11, 0.075),
]
tube("torso", COAT, rs(torso), sides=12, spacing=0.14, caps=(1, 1))
tube("neck_rider", SKIN, rs([(0, y0 - 0.01, 2.11, 0.058, 0.058), (0, y0 - 0.01, 2.25, 0.058, 0.058)]), sides=6, spacing=0.2, caps=(0, 0))
hx, hy, hz = rp(0, y0 - 0.005, 2.33)
blob("head_rider", SKIN, (hx, hy, hz), (0.088 * K, 0.103 * K, 0.122 * K), sides=10, rings=6)
brim = [(-0.24, y0, 2.50, 0.07, 0.022), (-0.14, y0, 2.45, 0.14, 0.022), (0, y0, 2.425, 0.155, 0.022), (0.14, y0, 2.45, 0.14, 0.022), (0.24, y0, 2.50, 0.07, 0.022)]
tube("hat_brim", HAT, rs(brim), sides=6, spacing=0.14, hint=(0, 1, 0), caps=(1, 1))
c.lathe("hat_crown", HAT, [(0.105 * K, rp(0, 0, 2.43)[2]), (0.100 * K, rp(0, 0, 2.48)[2]), (0.088 * K, rp(0, 0, 2.54)[2]), (0.0, rp(0, 0, 2.54)[2])],
        sides=8, centre=(0, y0), squash=1.15)
STATS.append(("hat_crown", 8 * 3 * 2))
for side in (-1, 1):
    ex, ey, ez = rp(side * 0.26, y0, 2.09)
    blob(f"epaulette{side}", STEEL, (ex, ey, ez), (0.06 * K, 0.05 * K, 0.03 * K), sides=6, rings=3)

for side in (-1, 1):
    thigh = [(side * 0.13, y0, 1.57, 0.115, 0.115), (side * 0.27, y0 + 0.20, 1.37, 0.09, 0.09)]
    tube(f"thigh{side}", BREECHES, rs(thigh), sides=8, spacing=0.2, caps=(1, 1))
    boot = [
        (side * 0.27, y0 + 0.20, 1.37, 0.085, 0.085),
        (side * 0.29, y0 + 0.16, 1.15, 0.072, 0.072),
        (side * 0.31, y0 + 0.12, 0.96, 0.060, 0.060),
        (side * 0.31, y0 + 0.17, 0.89, 0.056, 0.055),
        (side * 0.31, y0 + 0.29, 0.87, 0.048, 0.040),
    ]
    tube(f"boot{side}", BOOT, rs(boot), sides=8, spacing=0.14, caps=(1, 1))

for name, shoulder, elbow, wrist, hand_end in (
    ("arm_out", (0.25, y0, 2.06), (0.50, y0 + 0.02, 1.99), (0.76, y0 + 0.11, 1.92), (0.90, y0 + 0.14, 1.89)),
    ("arm_in", (-0.25, y0, 2.06), (-0.30, y0 - 0.04, 1.82), (-0.16, y0 + 0.18, 1.74), (-0.11, y0 + 0.26, 1.72)),
):
    tube(name, COAT, rs([(*shoulder, 0.070, 0.070), (*elbow, 0.060, 0.060), (*wrist, 0.047, 0.047)]), sides=6, spacing=0.16, caps=(1, 0))
    tube(name + "_hand", SKIN, rs([(*wrist, 0.040, 0.026), (*hand_end, 0.045, 0.018)]), sides=5, spacing=0.2, caps=(0, 1), hint=(0, 1, 0))

tube("scabbard", STEEL, rs([(-0.31, y0 - 0.02, 1.70, 0.022, 0.022), (-0.36, y0 - 0.45, 1.25, 0.022, 0.022), (-0.39, y0 - 0.80, 0.98, 0.018, 0.018)]),
     sides=4, spacing=0.4, caps=(0, 1))
tube("hilt", STEEL, rs([(-0.29, y0 - 0.02, 1.70, 0.020, 0.020), (-0.27, y0 + 0.02, 1.90, 0.017, 0.017)]), sides=4, spacing=0.4, caps=(1, 1))

cloak = [
    (0, y0 - 0.10, 2.08, 0.25, 0.04),
    (0, y0 - 0.17, 1.82, 0.28, 0.04),
    (0, y0 - 0.30, 1.66, 0.31, 0.04),
    (0, y0 - 0.55, 1.58, 0.31, 0.04),
]
tube("cloak", CLOAK, rs(cloak), sides=6, spacing=0.16, caps=(1, 1))

# The reins run from the bit, along the neck, to the hand that holds them.
for side in (-1, 1):
    rein = [(side * 0.06 - 0.10, 1.215, 1.66, 0.011, 0.011), (side * 0.13, 0.95, 1.73, 0.011, 0.011), (side * 0.19, 0.62, 1.72, 0.011, 0.011)]
    hand = rp(-0.12, y0 + 0.23, 1.73)
    tube(f"rein{side}", BOOT, rein + [(*hand, 0.011, 0.011)], sides=4, spacing=0.35, caps=(1, 1))


# --- Shade by facing, centre, scale and export ----------------------------------------------------------------------------
def tone(obj, strength=0.14):
    """Lighten the vertices that face up and darken those that face down, in the vertex colours."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    attribute = mesh.color_attributes[0]
    for loop in mesh.loops:
        nz = mesh.vertices[loop.vertex_index].normal.z
        item = attribute.data[loop.index]
        f = 1.0 + strength * nz
        item.color = (min(1.0, item.color[0] * f), min(1.0, item.color[1] * f), min(1.0, item.color[2] * f), 1.0)


meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for obj in meshes:
    tone(obj)

ys = [v.co.y for o in meshes for v in o.data.vertices]
zs = [v.co.z for o in meshes for v in o.data.vertices]
shift_y = -(min(ys) + max(ys)) / 2
scale = HEIGHT / (max(zs) - min(zs))
transform = Matrix.Scale(scale, 4) @ Matrix.Translation((0, shift_y, -min(zs)))
for obj in meshes:
    obj.data.transform(transform)
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))  # the viewer shades this layer smooth
    obj.data.update()

print("asset: parts", ", ".join(f"{n}={t}" for n, t in STATS), "| scale", f"{scale:.3f}", "| life-size height", f"{max(zs) - min(zs):.2f}")
c.finish(HEIGHT)

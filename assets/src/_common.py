"""Helpers for the Blender scripts in assets/src/. The scripts import this; the pipeline never does.

A script builds its parts (`lathe`, `box`, `part`), each in one flat sRGB colour, then calls `finish(height)`, which
joins them, checks the asset stands on z = 0 at the declared height, and exports a .glb to the path after `--`:

    blender --background --factory-startup --python assets/src/park_lamp.py -- assets/park-lamp.glb

or, for every script at once, `python assets/build.py`. Conventions (see assets/README.md): 1 unit = 1 metre, z up
while you model, the origin on the ground at the centre of the base, the asset facing +Y (north at heading 0).
Colours are given as sRGB hex, like everywhere in the pipeline, and stored linear.
"""

import math
import os
import sys

import bpy
import bmesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # lets a script `import _common`

GROUND_TOLERANCE = 0.05
HEIGHT_TOLERANCE = 0.03


def linear(hex_colour: str, shade: float = 1.0) -> tuple:
    """An sRGB hex colour as linear RGBA floats, optionally darkened (shade < 1) or lightened."""
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return (*[min(1.0, c * shade) for c in lin], 1.0)


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def part(name: str, verts: list, faces: list, colour: str, shade: float = 1.0):
    """A mesh object from vertex positions and faces (tuples of vertex indices, counter-clockwise seen from
    outside), painted one flat colour. Faces may have more than three vertices."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    attribute = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    rgba = linear(colour, shade)
    for item in attribute.data:
        item.color = rgba
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def lathe(name: str, colour: str, profile: list, sides: int = 8, centre: tuple = (0.0, 0.0), shade: float = 1.0,
          rotate: float = 0.0, squash: float = 1.0, closed: bool = True):
    """A solid of revolution about the vertical axis through `centre`: `profile` is a list of (radius, z) from
    the bottom up. A radius of 0 is a point; with `closed` (the default) a first or last radius above 0 is shut with a
    flat cap. `closed=False` leaves the ends open, for a collar or a cup: its faces then face outward only if the
    profile runs UP (z increasing), and `finish` leaves its winding alone instead of guessing it.
    `squash` stretches the shape along Y (an oval plan); `rotate` turns the sides by that many degrees."""
    verts, faces, rings = [], [], []
    for radius, z in profile:
        if radius <= 1e-9:
            rings.append([len(verts)])
            verts.append((centre[0], centre[1], z))
        else:
            ring = []
            for k in range(sides):
                angle = math.radians(rotate) + 2 * math.pi * k / sides
                ring.append(len(verts))
                verts.append((centre[0] + radius * math.cos(angle), centre[1] + radius * squash * math.sin(angle), z))
            rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for k in range(sides):
            k2 = (k + 1) % sides
            if len(a) == 1:
                faces.append((a[0], b[k2], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
    if closed and len(rings[0]) > 1:
        faces.append(tuple(reversed(rings[0])))
    if closed and len(rings[-1]) > 1:
        faces.append(tuple(rings[-1]))
    obj = part(name, verts, faces, colour, shade)
    if not closed:
        obj["keep_winding"] = True  # `finish` must not recalculate an open mesh's normals: it can turn it inside out
    return obj


def box(name: str, colour: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, shade: float = 1.0):
    """An axis-aligned box."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return part(name, v, f, colour, shade)


def output_path() -> str:
    if "--" not in sys.argv or sys.argv.index("--") + 1 >= len(sys.argv):
        raise SystemExit("usage: blender --background --factory-startup --python <script> -- <out.glb>")
    return sys.argv[sys.argv.index("--") + 1]


def finish(height: float, out: str | None = None) -> None:
    """Join every mesh, check it against `height` and the ground, and export the .glb."""
    out = out or output_path()
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("nothing to export")
    for obj in meshes:  # outward-facing triangles (closed meshes only: on an open one the guess can be wrong)
        if obj.get("keep_winding"):
            continue
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    zs = [v.co.z for v in obj.data.vertices]
    low, top = min(zs), max(zs)
    triangles = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if abs(low) > GROUND_TOLERANCE:
        raise SystemExit(f"lowest point is {low:+.3f} m: the asset must stand on z = 0")
    if abs((top - low) - height) > HEIGHT_TOLERANCE * height:
        raise SystemExit(f"asset is {top - low:.3f} m tall, not {height} m")
    tried = []
    for extra in (
        {"export_vertex_color": "ACTIVE", "export_materials": "NONE", "export_texcoords": False},
        {"export_vertex_color": "ACTIVE", "export_materials": "NONE"},
        {"export_colors": True},
    ):
        try:
            bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True, export_apply=True, **extra)
            print(f"asset: {triangles} triangles, {top - low:.2f} m tall -> {out}")
            return
        except Exception as error:  # noqa: BLE001  (the option names change between Blender versions)
            tried.append((extra, str(error)[:100]))
    raise SystemExit(f"glTF export failed: {tried}")

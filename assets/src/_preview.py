"""Blender side of assets/preview.py: import a .glb and render four views of it with its vertex colours.

    blender --background --factory-startup --python assets/src/_preview.py -- model.glb out_prefix [size]

Writes out_prefix_front.png, _right.png, _back.png and _three_quarter.png. "Front" is the side the asset faces (+Y).
"""

import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
glb, prefix = args[0], args[1]
size = int(args[2]) if len(args) > 2 else 640

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
centre, extent = (low + high) / 2, max((high - low).length, 0.1)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.image_settings.file_format = "PNG"
shading = scene.display.shading
shading.light = "STUDIO"
for colour_type in ("VERTEX", "ATTRIBUTE", "MATERIAL"):  # the name differs between Blender versions
    try:
        shading.color_type = colour_type
        break
    except TypeError:
        continue
shading.show_cavity = False
shading.show_shadows = True
world = bpy.data.worlds.new("preview")
world.color = (0.78, 0.86, 0.95)
scene.world = world

target = bpy.data.objects.new("target", None)
bpy.context.scene.collection.objects.link(target)
target.location = centre
camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
bpy.context.scene.collection.objects.link(camera)
camera.data.type = "ORTHO"
camera.data.ortho_scale = extent * 1.15
constraint = camera.constraints.new("TRACK_TO")
constraint.target, constraint.track_axis, constraint.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
scene.camera = camera

views = {
    "front": (0, 1, 0.1),  # the camera stands in front of the asset (+Y), looking back at it
    "right": (1, 0, 0.1),
    "back": (0, -1, 0.1),
    "three_quarter": (0.7, 0.7, 0.45),
}
for name, (x, y, z) in views.items():
    camera.location = centre + Vector((x, y, z)).normalized() * extent * 3
    scene.render.filepath = f"{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True)
print("preview:", len(views), "views of", len(meshes), "mesh(es),", f"{(high - low).z:.2f} m tall")

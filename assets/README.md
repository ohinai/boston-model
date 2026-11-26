# Assets

**Status: tabled (26 November 2026).** The three models here stay and keep working, but no more are planned: new
shapes are drawn procedurally, in code.

Hand-made models (made in Blender) for the few things worth more than a procedural shape: a statue, a bridge pylon, a
lamp. The city is still generated. The pipeline decides **where** each thing goes from the data and **places** the model
there; an asset only decides what the thing looks like. A missing asset is never an error: the pipeline falls back to
the procedural shape, so a build works with this folder empty.

```
assets/
  park-lamp.glb        the model (committed; what the pipeline reads)
  park-lamp.toml       its sidecar: size, budget, who made it and why
  src/park_lamp.py     the Blender script that makes it (run with: python assets/build.py park-lamp)
  src/_common.py       helpers for those scripts
  build.py             runs the scripts in headless Blender
```

## What the viewer keeps

The viewer draws every tile layer with **one shared material** and keeps only each mesh's **vertex colours**
(`materialFor` in `web/src/city.ts`). So an asset has **no textures and no materials**: its look is the colour painted on
each vertex, plus the layer's roughness and metalness. Paint flat colours per part; a gradient is a few more vertices.

Tiles bake every copy of an asset into their mesh (there is no instancing), so **triangles multiply by the number of
copies**. The pilot's whole district is about 1.3 million triangles. Budget accordingly:

| Kind of asset | Copies | Keep it under |
|---|---|---|
| A park lamp, bench, bin | about a thousand | 100 to 150 triangles |
| A bridge pylon, a monument | a handful | 1,000 to 2,000 |
| A statue or other one-off | one to three | 3,000 to 5,000 |

Every extra triangle is also collision geometry for the walker (only trees are not solid).

## Conventions (checked when an asset loads)

- Standard glTF (y up), 1 unit = 1 metre. Model with z up in Blender: its exporter converts.
- The origin is **on the ground at the centre of the base**. The lowest vertex is at 0 (within 5 cm).
- The asset **faces +Y in Blender**. That is north at heading 0; the pipeline turns it to the feature's `direction`.
- Its height (lowest to highest vertex) is what the sidecar says, within 3%. The pipeline scales an asset to the height
  the data gives (for example a lamp's `height` tag), so model at the typical size.
- Every vertex has a colour in `COLOR_0` (linear). `_common.py` takes sRGB hex, like the rest of the project.
- Triangles only, no textures, no morph targets. Counter-clockwise front faces.

## The sidecar

```toml
file = "park-lamp.glb"
height = 4.9              # metres, lowest to highest vertex
max_triangles = 100       # the budget; the build fails if the model is over it
author = "..."            # who made it
licence = "MIT"           # the project's own, or CC0; say where anything else came from
why = "Which things in the data it stands for, and what it was modelled to look like."
source = "src/park_lamp.py"   # optional: the script that makes it
```

## Making one

```sh
python assets/build.py park-lamp      # needs Blender 3.6 or later: set BLENDER, or have it on the PATH
```

Scripts use `assets/src/_common.py`: `lathe` (a solid of revolution: posts, globes, hats, pylons), `box`, `part` (any
vertices and faces), then `finish(height)`. Start each script with this, so Blender can find the helpers:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c
c.reset()
```

`bpy` and `bmesh` are available for anything else (extrude, bevel, boolean, sculpt); keep each part one flat colour.

Two things the pilot found. **Curves must be real facets:** tiles carry no normals and the viewer shades every layer
flat (only trees are smoothed), so smooth shading or subdivision-for-shading in Blender is discarded; a round globe
needs enough sides to look round on its own. **Open meshes need care:** `finish` makes closed parts face outward but
leaves a part built with `lathe(..., closed=False)` as you wrote it (a profile that runs up faces outward), because the
viewer culls back faces and a flipped collar shows its far wall through. Hand-modelling in Blender's window works too: set the origin to the base, apply
transforms, give every part a colour attribute (Color Attributes, a Corner domain), and export `.glb` with Vertex Color
set to Active and Materials set to None. A hand-made file replaces a scripted one without any other change.

**Where the reference comes from.** The project's rules apply: look at photographs of the real thing to decide what it
should look like, but never trace a position or a dimension from one, and never use a photograph as a texture. Sizes
and positions come from the data. Say in `why` what the model stands for.

## Using one in the pipeline

```python
from . import assets

asset = assets.load("park-lamp")           # None if there is none (or BOSTON_NO_ASSETS is set)
if asset is not None:
    parts.add(*assets.place(asset, x, y, direction, z0, assets.scale_to(asset, height)))
else:
    ...the procedural shape...
```

`place` returns exactly what furniture's `_Parts.add` takes. `BOSTON_NO_ASSETS=1 boston-pipeline build pilot` builds the
procedural version, so a before and after picture is two builds in the same checkout.

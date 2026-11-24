# Using the model in Unreal Engine

> **Not yet tested.** The tiles are written to the 3D Tiles standard that
> Cesium for Unreal reads, and the steps below follow Cesium's documentation,
> but nobody has run this project's tiles in Unreal yet. If you try it, please
> report what happened, working or not.

The model reaches Unreal as data, not as an Unreal project full of imported
meshes. [Cesium for Unreal](https://github.com/CesiumGS/cesium-unreal) is a
free, Apache-licensed plugin that streams 3D Tiles into a level and builds
collision for them, so a character can walk on what it loads.

## What you need

- Unreal Engine 5.6, 5.7 or 5.8 (the versions supported by Cesium for Unreal
  2.31, November 2026).
- The Cesium for Unreal plugin, from its
  [releases page](https://github.com/CesiumGS/cesium-unreal/releases).
- Built tiles: either a local build (`boston-pipeline build pilot`, which writes
  `web/public/tiles/`) or the address of a published `tileset.json`.

## Steps

1. Create a project and enable the Cesium for Unreal plugin.
2. Add a **CesiumGeoreference** to the level if the plugin has not added one,
   and set its origin to the model's origin:
   - latitude `42.358669`
   - longitude `-71.063006`
   - height `-27.74` (metres above the ellipsoid; this is sea level in Boston)
3. Add a **Cesium3DTileset** actor. Set **Source** to *From Url* and **Url** to
   the tileset:
   - local build: `file:///C:/path/to/boston_model/web/public/tiles/tileset.json`
   - published: the `tileset.json` address on your host
4. Leave **Create Physics Meshes** on (it is by default) so the tiles are solid.
5. Drop in a character, for example from the Third Person template, a few
   metres above the ground at the origin.

## Things to expect

- **Scale and axes.** Tiles are in metres with the standard glTF and 3D Tiles
  axis conventions. The plugin converts to Unreal's centimetres and left-handed
  axes; do not rotate or scale the tileset actor to compensate.
- **Plain materials.** Tiles carry flat colours. The window pattern seen in the
  browser is drawn by the viewer, not stored in the tiles. Wall UVs count one
  unit per window bay and storey (see [ARCHITECTURE.md](ARCHITECTURE.md)), so a
  facade material that repeats once per UV unit will line up.
- **Layer names.** Each tile's meshes and materials are named `ground`,
  `roads`, `building_walls` and so on. Use the names to assign Unreal materials.
- **Collision only where loaded.** A character that outruns tile loading can
  fall through. Keep the tileset's loading distance generous around the player.
- **Everything loads at full detail.** The tileset has no simplified levels
  yet, which is fine for the pilot district and will not be for a whole city.

## Unity

Cesium for Unity reads the same tileset and also creates physics meshes. It has
not been tried with this project either.

## Where this is heading

The plan is an Unreal project in this repository that is a thin client: it
points at published tiles and adds a character, lighting and, later, the
simulation layer. The city itself stays in the pipeline so the browser and
desktop versions cannot drift apart.

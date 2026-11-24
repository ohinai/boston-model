# Architecture

This is the reference for how the model is put together. Read it before changing
the pipeline's output or writing a new client.

## The idea

Source data and hand corrections go in, tiles come out, and clients only read
tiles.

```
OpenStreetMap ─┐
overrides/ ────┼─► pipeline ─► tileset.json + tiles/*.glb + tiles/*.json ─► viewers
(more sources) ┘
```

Three consequences follow from this:

- **Nothing built is committed.** Tiles are reproducible from the recipe, so git
  holds text that can be reviewed and merged.
- **Clients stay thin.** The browser viewer and any game-engine project load the
  same files and add only what is specific to them: rendering, controls, physics.
- **Accuracy improves at the source.** A fix made in OpenStreetMap or in
  `overrides/` reaches every client on the next build.

## World frame

Every tool and client shares one coordinate system, defined in
[`pipeline/boston_pipeline/world.py`](../pipeline/boston_pipeline/world.py).

| | |
|---|---|
| Projection | NAD83 / Massachusetts Mainland, metres (EPSG:26986) |
| Origin | State Plane E 236 000, N 901 000, which is on Beacon Hill at 42.358669° N, 71.063006° W |
| Axes | x east, y north, z up, all in metres |
| Heights | metres above NAVD88; the pilot is flat at z = 0 |

A state plane projection was chosen over latitude and longitude or a
tangent-plane frame because Massachusetts publishes its data in it, distances
are true metres across the whole city, and "up" stays vertical everywhere.

WGS84 coordinates from OpenStreetMap are read as NAD83. The two differ by about
a metre here, which is below the accuracy of the source data, and skipping the
datum shift makes the Python pipeline and the browser agree to the millimetre.

Changing the origin or the projection moves everything ever built, including
positions in `overrides/`. Treat these constants as permanent.

## Tiles

The world is cut into 256 m squares. Tile `(i, j)` covers
`[256·i, 256·(i+1))` east and `[256·j, 256·(j+1))` north of the origin. A tile
is named `i_j`, for example `-2_1`.

A building belongs to the one tile that contains its centre point and is never
cut at tile edges. Ground surfaces are cut exactly at tile edges.

`areas.toml` lists named areas as latitude and longitude boxes. A build rounds
the box outward to whole tiles.

### Output files

```
tileset.json        3D Tiles 1.1 tileset: one entry per tile
tiles/-2_1.glb      geometry for tile (-2, 1), binary glTF 2.0
tiles/-2_1.json     what is in that tile, as data
```

**`tileset.json`** follows the [3D Tiles 1.1](https://docs.ogc.org/cs/22-025r4/22-025r4.html)
specification. The root transform places the world frame on the globe for
Cesium-based clients. Each child tile has a translation to its south-west
corner. Project-specific facts live under `extras`: the projection string, the
origin, the tile size, the attribution text and the landmarks from `areas.toml`.

**Tile `.glb` files** are in glTF's axes (x east, y up, z south) relative to the
tile's south-west corner. Each layer is one node, one mesh and one material, all
with the same name:

| Layer | Contents | Height |
|---|---|---|
| `ground` | Paved ground not covered by anything else | 0 |
| `green` | Parks, lawns, cemeteries | 0 |
| `paths` | Footpaths, plazas, piers, platforms | 0 |
| `roads` | Carriageways | −0.12 m |
| `rail` | Surface track | 0 |
| `water` | Rivers, ponds, harbour | −0.6 m |
| `curbs` | Vertical faces around roads and water | up to 0 |
| `building_walls` | Walls, with per-vertex colour and facade UVs | |
| `building_roofs` | Roofs and undersides, with per-vertex colour | |
| `trees` | Mapped trees as simple shapes, with per-vertex colour | |

Clients may rely on these names. The browser viewer uses them to decide what is
solid (everything except `trees`) and to swap in its own materials.

Ground layers never overlap. Where two meet, the higher priority one wins:
roads, then rail, paths, water, green, ground. Carriageways are also kept at
least a metre away from building footprints, so a street estimated too wide
cannot run into a house.

Wall UVs count facade cells rather than metres. Along each wall `u` runs from 0
to the number of window bays, and `v` from 0 to the number of storeys, so one UV
unit is always one whole bay by one whole storey. A client draws windows by
repeating a pattern or texture once per unit. Walls too short for a bay have
`u = 0` throughout.

**Tile `.json` files** describe the same square as data, in world metres:

```json
{
  "tile": [0, 0],
  "streets":   [{"id": "w426472927", "name": "Tremont Street", "kind": "primary",
                 "surface": "road", "width": 10.2, "path": [[256.0, 10.73], [251.62, 0.0]]}],
  "buildings": [{"id": "w29808921", "name": "Frank Sawyer Building", "pos": [92.8, 32.42],
                 "height": 37.0, "height_source": "height"}],
  "pois":      [{"id": "n257535860", "name": "Mildred F. Sawyer Library",
                 "kind": "amenity=library", "pos": [101.44, 34.85]}],
  "signals":   [[120.5, 80.25]]
}
```

Ids are OpenStreetMap ids with a type prefix (`n` node, `w` way, `r` relation).
`height_source` is `height` (a measured height tag), `levels` (storey count),
`default` (a guess by building type) or `override`. This file is where
simulation will hang: signals, shop fronts and transit stops are positions and
facts here long before they are models.

## Pipeline

`boston-pipeline fetch <area>` downloads OpenStreetMap data through the Overpass
API into `data/raw/<area>/`. `boston-pipeline build <area>` then runs three
stages:

1. **Classify** ([`features.py`](../pipeline/boston_pipeline/features.py)).
   Interpret tags: what is a road, how wide, how tall is a building with no
   height. All tagging conventions and default values live here.
2. **Prepare** ([`tiles.py`](../pipeline/boston_pipeline/tiles.py)). Buffer
   street centre lines into surfaces, derive the sea from coastline ways, and
   index everything spatially.
3. **Build tiles.** For each tile, look up what touches it, split the ground
   into layers, extrude buildings, and write the `.glb` and `.json`.

Each tile is built from index lookups alone, so build time grows with the
number of tiles rather than the size of the city, and tiles can later be built
in parallel or one at a time.

[`mesh.py`](../pipeline/boston_pipeline/mesh.py) holds the geometry helpers and
[`gltf.py`](../pipeline/boston_pipeline/gltf.py) a small glTF writer. The
pipeline depends only on numpy, shapely, pyproj and mapbox-earcut.

## Browser viewer

[`web/`](../web) is a static site: TypeScript, three.js and Vite, with no
server-side part. It reads `tiles/tileset.json` from beside the page, or from
any address given as `?tileset=<url>`.

- [`city.ts`](../web/src/city.ts) loads tiles near the player, builds a
  collision structure for each, and answers "what street is this".
- [`player.ts`](../web/src/player.ts) is a capsule-shaped walker with gravity,
  plus a fly mode. Collision is against the real tile geometry, so anything a
  future build adds (terrain, steps, bridges) is walkable without viewer changes.
- [`main.ts`](../web/src/main.ts) sets up lighting, sky and the on-screen text.

The page address carries `#lat,lon,heading` (and altitude and pitch when
flying). Positions are latitude and longitude so links survive any future
change to the world frame.

## Desktop engines

Unreal Engine and Unity can load 3D Tiles through the open-source Cesium
plugins, which also generate collision for the tiles. See [UNREAL.md](UNREAL.md).
The engine project should stay a thin client like the browser viewer: it loads
tiles and adds engine-specific behaviour, and does not become a second place
where the city is edited.

## Known limits of the current design

- **No levels of detail.** Every tile in range loads at full detail. A whole
  city needs a tile hierarchy with simplified parents, which 3D Tiles supports.
- **Flat-earth placement.** The whole tileset is placed on the globe as one
  plane, which drifts from the curved surface by about 8 m at 10 km from the
  origin. Per-tile placement fixes this when the area grows.
- **Uncompressed geometry.** The pilot is about 26 MB. Quantisation and mesh
  compression are standard glTF extensions and are not applied yet.
- **Overpass for downloads.** Fine for a district. A whole-city build should
  read a regional extract instead.

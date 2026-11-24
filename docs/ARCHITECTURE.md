# Architecture

This is the reference for how the model is put together. Read it before changing
the pipeline's output or writing a new client.

## The idea

Source data and the corrections in `fixes/` go in, tiles come out, and clients
only read tiles.

```
OpenStreetMap ───────┐
City tree inventory ─┤
fixes/ ──────────────┼─► pipeline ─► tileset.json + tiles/*.glb + tiles/*.json ─► viewers
(more sources) ──────┘
```

Three consequences follow from this:

- **Nothing built is committed.** Tiles are reproducible from the recipe, so git
  holds text that can be reviewed and merged.
- **Clients stay thin.** The browser viewer and any game-engine project load the
  same files and add only what is specific to them: rendering, controls, physics.
- **Accuracy improves in one place.** A fix added to `fixes/` reaches every
  client on the next build, and survives every later data update.

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
the positions recorded in notes. Treat these constants as permanent.

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
| `green` | Parks, lawns, pitches, playgrounds, cemeteries, with per-vertex colour by kind of surface | 0 |
| `beds` | Planted beds: bare soil at the edge, a low mass of planting in from it, with per-vertex colour. Solid | 0, planting 0.35 m |
| `paths` | Footpaths, plazas, piers, platforms, with per-vertex colour by surface (asphalt, brick, gravel and so on) | 0 |
| `roads` | Carriageways | −0.12 m |
| `rail` | Surface track | 0 |
| `water` | Rivers, ponds, harbour | −0.6 m |
| `curbs` | Vertical faces around roads and water | up to 0 |
| `building_walls` | Walls, with per-vertex colour and facade UVs | |
| `building_roofs` | Roofs and undersides, with per-vertex colour | |
| `trees` | Trees as low-poly shapes whose form, size and colour follow species and trunk size, with per-vertex colour | |
| `furniture` | Benches, lamp posts, litter bins, drinking fountains, flagpoles, with per-vertex colour | |
| `monuments` | Statues, sculptures, memorials, monuments, plaques and fountains. Placeholders of the right kind and size in the right place, not likenesses | |
| `bridges` | Raised footbridge decks with railings, and the stone piers of things standing in the water beside them | up to about 2 m |

Clients may rely on these names. The browser viewer uses them to decide what is
solid (everything except `trees`) and to swap in its own materials.

Vertices in `building_walls` and `building_roofs` carry a `_FEATURE_ID_0`
attribute (declared with the `EXT_mesh_features` extension): the building's
position in the `buildings` list of the tile's data file. Vertices in `trees`,
`furniture`, `monuments` and `bridges` carry it too, as a position in the
`objects` list. That is how a client tells what a click or a line of sight has
hit.

Meshes are written without normals. Every surface is flat-shaded, and glTF
requires a client to shade a mesh with no normals by each face's own normal.
Vertices are shared where position, colour, UV and feature id agree, and
indices are 16-bit wherever a mesh is small enough.

Ground layers never overlap. Where two meet, the higher priority one wins:
roads, then rail, paths, water, beds, green, ground. Where two kinds of path
overlap, the first kind in `COVERS` in `tiles.py` shows. Where two green areas
overlap, the smaller one shows, so a pitch inside a lawn inside a park is a
pitch. Carriageways are also kept at
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
  "objects":   [{"id": "bprd:1416", "kind": "tree", "species": "English elm",
                 "height": 21.4, "pos": [120.1, 64.8]},
                {"id": "n123456789", "kind": "statue", "name": "George Washington Statue",
                 "pos": [130.2, 40.6]}],
  "areas":     [{"id": "w123456", "name": "Boston Public Garden", "kind": "park", "area": 96000,
                 "rings": [[[0.0, 0.0], [256.0, 0.0], [256.0, 90.5], [0.0, 80.2], [0.0, 0.0]]]}],
  "pois":      [{"id": "n257535860", "name": "Mildred F. Sawyer Library",
                 "kind": "amenity=library", "pos": [101.44, 34.85]}],
  "signals":   [[120.5, 80.25]]
}
```

Ids are OpenStreetMap ids with a type prefix (`n` node, `w` way, `r` relation),
`bprd:<n>` for a tree from the City of Boston inventory, or `fix:<file>#<n>`
for a feature added by a file in `fixes/`. A building part also has `part_of`,
the id of the building it belongs to, and takes that building's name.

`objects` lists everything drawn as a thing standing in the tile: each tree,
piece of furniture, monument and raised bridge. Each entry has `id`, `kind`
(`tree`, `bench`, `lamp`, `waste_basket`, `drinking_water`, `flagpole`,
`plaque`, `stele`, `obelisk`, `column`, `statue`, `sculpture`, `monument`,
`fountain` or `bridge`), `pos`, and where known `name`. A tree also has
`species` (a common name) and `height` in metres. The height is an estimate
unless the data gave one. `areas` lists the named parks, gardens, pitches and
playgrounds that overlap the tile: `id`, `name`, `kind`, `area` in square metres
of the whole feature, and `rings`, the outlines of the part inside this tile.
A client uses them to say which park someone is standing in.
`height_source` is `height` (a measured height tag), `levels` (storey count),
`default` (a guess by building type) or `override`. This file is where
simulation will hang: signals, shop fronts and transit stops are positions and
facts here long before they are models.

## Pipeline

`boston-pipeline fetch <area>` downloads OpenStreetMap data through the Overpass
API into `data/raw/<area>/osm.json`, and the City of Boston tree inventory
([`inventory.py`](../pipeline/boston_pipeline/inventory.py)) into
`data/raw/<area>/trees.json`. `boston-pipeline build <area>` then runs four
stages:

0. **Apply fixes** ([`fixes.py`](../pipeline/boston_pipeline/fixes.py)). Turn
   each inventory record into an element shaped like an OpenStreetMap node,
   with the id `bprd:<n>`. Then read every file under [`fixes/`](../fixes) and
   apply it to all the downloaded elements: change tags, delete features, add
   new ones. Fixes use the same tag vocabulary as the source data, so
   everything after this point treats corrected and original data alike.
1. **Classify** ([`features.py`](../pipeline/boston_pipeline/features.py)).
   Interpret tags: what is a road, how wide, how tall is a building with no
   height, what a path is made of, which kind of green area, and which piece of
   furniture or monument a point is. All tagging conventions and default values
   live here. Then reconcile the trees: an inventory tree and an OpenStreetMap
   tree within 4 m of each other are one tree, and the inventory's is kept. In a
   park the inventory covers, OpenStreetMap trees it does not list are dropped.
   Elsewhere OpenStreetMap trees are kept.
2. **Prepare** ([`tiles.py`](../pipeline/boston_pipeline/tiles.py)). Buffer
   street centre lines into surfaces, derive the sea from coastline ways, pick
   out the footbridges that will be raised, and index everything spatially.
3. **Build tiles.** For each tile, look up what touches it, split the ground
   into layers, extrude buildings, place trees, furniture, monuments and
   bridges, and write the `.glb` and `.json`.

Each tile is built from index lookups alone, so build time grows with the
number of tiles rather than the size of the city, and tiles can later be built
in parallel or one at a time.

`boston-pipeline inspect <lat>,<lon>`
([`lookup.py`](../pipeline/boston_pipeline/lookup.py)) runs the first two stages
and reports what is at a position: each source feature's id and tags, and the
layer the spot is drawn in, including trees and objects. It is the starting
point for fixing anything.

Tree size and shape come from [`trees.py`](../pipeline/boston_pipeline/trees.py):
a table of genera gives each a form, a typical mature height and spread, and
leaf and bark colours. Height is the `height` tag if there is one, otherwise it
is estimated from trunk diameter along a curve that levels off at the genus's
mature height. Crown width is `diameter_crown` or the genus's spread. These are
estimates, not measurements. Shapes for furniture and monuments are in
[`furniture.py`](../pipeline/boston_pipeline/furniture.py), and footbridge decks
in [`bridges.py`](../pipeline/boston_pipeline/bridges.py). A footbridge that
crosses water on its own is raised in a gentle arch over the water (2.5% of the
raised length, between 0.3 and 0.8 m), and the water continues beneath it. The
parts of the bridge way that lie on the banks stay ordinary path, so paths that
run along the bank can still cross. Bridges over roads and
rail, road bridges and their sidewalks stay at ground level until the model has
levels.

[`mesh.py`](../pipeline/boston_pipeline/mesh.py) holds the geometry helpers and
[`gltf.py`](../pipeline/boston_pipeline/gltf.py) a small glTF writer. The
pipeline depends only on numpy, shapely, pyproj and mapbox-earcut.

## Browser viewer

[`web/`](../web) is a static site: TypeScript, three.js and Vite, with no
server-side part. It reads `tiles/tileset.json` from beside the page, or from
any address given as `?tileset=<url>`.

- [`city.ts`](../web/src/city.ts) loads tiles near the player, builds a
  collision structure for each, and answers "what street is this" and "what
  did that ray hit".
- [`player.ts`](../web/src/player.ts) is a capsule-shaped walker with gravity,
  plus a fly mode. Collision is against the real tile geometry, so anything a
  future build adds (terrain, steps, bridges) is walkable without viewer changes.
- [`main.ts`](../web/src/main.ts) sets up lighting, sky and the on-screen text.
- [`notes.ts`](../web/src/notes.ts) and [`notepad.ts`](../web/src/notepad.ts)
  handle notes, described next.

The page address carries `#lat,lon,heading` (and altitude and pitch when
flying). Positions are latitude and longitude so links survive any future
change to the world frame.

## Notes

A note is how someone walking the model reports a problem. Pressing `N`
records the position and view at that instant, then asks for a sentence or
two. The viewer has no server, so a note is kept in the browser's local
storage and filed by opening GitHub's new-issue page with the title and text
already filled in.

The ticket ends with a JSON block that tools read:

```json
{
 "boston_note": 1,
 "id": "n-mwuwi5s5-npqu",
 "standing": {"lat": 42.3544, "lon": -71.0702, "x": -590.29, "y": -477.24, "z": 0,
              "heading": 200, "pitch": -6, "flying": false},
 "looking_at": {"lat": 42.3542698, "lon": -71.0702649, "x": -595.56, "y": -491.73, "z": 0,
                "distance": 15.5, "layer": "monuments",
                "object": {"id": "n123456789", "kind": "statue", "name": "George Washington Statue"}},
 "tile": [-3, -2],
 "street": "Lagoon Path",
 "park": "Boston Public Garden",
 "near": "Japanese Lantern",
 "link": "https://ohinai.github.io/boston-model/#42.354400,-71.070200,200,0,-6",
 "data": "2026-11-24T14:25:35Z"
}
```

`standing` is where the author stood; `heading` is degrees clockwise from north
and `pitch` degrees above the horizon. `looking_at` is the first thing straight
ahead: its position, the tile layer it belongs to, for a building its id, name
and height, and for a tree, bench, lamp post, statue, bridge and the like its
`object` (`id`, `kind`, and `name` and `species` where known). `park` is the
named park or garden the author stood in. `data` is the timestamp of the source data the author saw.
These field names are a contract with [AGENTS.md](../AGENTS.md); add fields
rather than renaming them, and raise `boston_note` if the meaning changes.

A workflow labels such tickets `note`. On load the viewer reads the
repository's open issues through GitHub's public API and shows each note as a
marker where it was left.

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
- **Uncompressed geometry.** The pilot is 70 tiles, about 904,000 triangles and
  22 MB. Tiles are already written without normals, with shared vertices and
  16-bit indices where they fit. Quantisation and mesh compression are standard
  glTF extensions and are not applied yet.
- **Trees are not solid** and their sizes are estimates. Bridges over roads and
  rail, road bridges and tunnels are not modelled.
- **Overpass for downloads.** Fine for a district. A whole-city build should
  read a regional extract instead.

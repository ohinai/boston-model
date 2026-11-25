# Working on Boston Model with an AI assistant

This file is for coding assistants (and the people directing them). It explains
how to turn a note about something wrong in the model into a fix.

## What this project is

A walkable 3D model of Boston that should feel realistic at eye level. It is
generated: a Python pipeline turns downloaded map data plus the files in
`fixes/` into 3D tiles, and a browser viewer shows them. Nobody edits 3D scenes
by hand.

**All corrections go through this repository**, as files in `fixes/` or as
changes to the pipeline. Do not send contributors to edit OpenStreetMap.

The current priority is parks and gardens: Boston Common and the Public Garden
first. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup and commands

```sh
python -m venv .venv
.venv\Scripts\activate               # Windows; elsewhere: source .venv/bin/activate
pip install -e "./pipeline[dev]"      # the dev extra includes Pillow, which audit needs
boston-pipeline fetch pilot          # download source data once (cached in data/)
boston-pipeline build pilot          # build tiles into web/public/tiles
boston-pipeline inspect LAT,LON      # what is at a spot, with ids and tags
boston-pipeline audit pilot --park "Boston Public Garden"   # model drawn on aerial photographs (see below)
boston-pipeline ground pilot --park "Boston Public Garden"  # photographs beside the model's views from the same spots (see below)
pytest pipeline                      # pipeline tests

cd web && npm install
npm run dev                          # viewer at http://localhost:5173
npm run build                        # type-check and build the viewer
```

## Notes and tickets

People walking the model press `N` to leave a note. It becomes a GitHub issue
labelled `note`. List them with `gh issue list --label note`.

Each ticket ends with a JSON block (it contains `"boston_note": 1`):

| Field | Meaning |
|---|---|
| `standing` | Where the author stood. `lat`, `lon`; `x`, `y`, `z` in world metres (east, north, up); `heading` in degrees clockwise from north; `pitch` in degrees above the horizon; `flying`. |
| `looking_at` | The first thing straight ahead of them: its position, `distance`, the `layer` it is drawn in, and for buildings the `building` id, name and height. For a tree, bench, lamp post, statue, bridge and the like it also has `object`: `id`, `kind`, and `name` and `species` where known. Absent if they were looking at the sky. |
| `park` | The named park, garden or playground the author stood in, if any. |
| `tile` | The tile they stood in, as `[i, j]`; its files are `web/public/tiles/tiles/i_j.glb` and `.json`. |
| `street`, `near` | Human-readable place names. |
| `link` | Opens the viewer at the same position and view. |
| `data`, `build` | Source-data timestamp and viewer commit the author was seeing. |

The note is usually about `looking_at`, not about where the author stood.

## Fixing a note

1. **Read it.** `gh issue view <number>`. Read the text, then the JSON block.
2. **Look at the data there.**
   `boston-pipeline inspect <looking_at.lat>,<looking_at.lon>` (fall back to
   `standing`; widen with `--radius 30`; add `--json` for full tags). This lists
   every source feature at and near the spot with its id, its tags, and which
   layer it is drawn in. That includes trees (OpenStreetMap or city inventory),
   benches, lamps and statues.
3. **Decide what kind of problem it is.** This matters more than anything else:

   | What is wrong | Where the fix goes |
   |---|---|
   | The data is wrong or missing for one place: a wrong height, a missing path, a pond with the wrong outline, a tree that is gone | A file in `fixes/` |
   | The data is right but drawn wrongly everywhere: all paths too wide, lawns the wrong colour | `pipeline/boston_pipeline/features.py` (interpretation) or `tiles.py` (geometry) |
   | The data has it but nothing draws it: gates, signs, kiosks, steps as steps | New interpretation in `features.py` and geometry in `tiles.py` (or a module like `furniture.py` or `barriers.py`), with a test |
   | It is about look or behaviour in the browser: lighting, shading, controls | `web/src/` |

   If a note needs a pipeline change, say so on the ticket rather than faking it
   with a one-off fix file.

   Trees are a data problem, so they go in `fixes/` by id (see "Fixing a tree"
   below). Fences, walls, hedges, chains, guard rails, kerbs and bollards are
   downloaded and drawn, so a wrong or missing one is a fix file with ordinary
   `barrier=` tags. Gates are not drawn (a gate stands open), so a note about
   one is a request for new drawing code.
4. **Write the fix.** [fixes/README.md](fixes/README.md) has both formats:
   `.toml` to change or delete an existing feature by id, `.geojson` to add or
   redraw one. Every entry needs a `why`.
5. **Rebuild and check.** `boston-pipeline build pilot` prints how many fixes
   applied and warns about any that matched nothing. Run `inspect` again at the
   same spot. Then open the note's `link` against the local viewer: keep
   everything from `#` onward and put `http://localhost:5173/` in front.
6. **Test.** `pytest pipeline`, and `npm run build` in `web/` if you touched it.
7. **Open a pull request** whose description says `Fixes #<number>` and where
   the correct information came from.

## Checking the model against photographs

Notes are not the only way to find mistakes. `boston-pipeline audit` draws the
model on top of an aerial photograph so that a person or an assistant can see
where the two disagree.

```sh
boston-pipeline audit pilot --park "Boston Public Garden"     # or: --around LAT,LON --size METRES
```

It writes one folder per 120 m cell under `data/audit/pilot/`:

| File | What it is |
|---|---|
| `aerial.jpg` | The photograph, north up, in the model's own projection |
| `overlay.jpg` | The photograph with the model drawn on it: numbered tree crowns, path edges, planted beds, water, raised footbridges, buildings, furniture |
| `canopy.jpg` | The photograph with the detected tree canopy outlined (not written with `--no-canopy`) |
| `features.json` | What each number and outline is, and the cell's bounds and `metres_per_pixel` |

The audit also finds tree canopy in the survey's near-infrared band. It
proposes a tree (`C1`, `C2`, ...) where a compact crown stands with no tree in
the data, and measures the crown width of trees that stand alone. It cannot
separate trees inside dense canopy, reads deep shade as canopy and very light
foliage as lawn, so treat every proposal and measurement as a suggestion.

In Claude Code, `/audit "Boston Public Garden"` checks each cell and writes a
`findings.json` beside it, with a verdict on every proposal and measurement.
[.claude/commands/audit.md](.claude/commands/audit.md) is the procedure and the
format of that file. It is written so that a cheaper model can do the checking.

Checking and fixing are separate steps. The audit never changes the model. Look
again at each confident finding, then write it up as a file in `fixes/` as
described above. Cite the photograph in `why`, with its source and date (see
below), and say how it was measured.

### From the ground

The aerial audit cannot show what a walker sees. `boston-pipeline ground` finds
openly licensed, geotagged photographs taken inside a park on Wikimedia
Commons (JPEG, 2015 or later, thinned so the park is covered evenly) and
gets pictures of the model from each photograph's position, standing on the
ground.

```sh
boston-pipeline build pilot                 # the tiles must be current
cd web && npm run build && cd ..            # the pictures are taken with the built viewer (web/dist)
boston-pipeline ground pilot --park "Boston Public Garden"   # options: --limit 40, --no-capture, --force
```

It writes one folder per photograph under `data/ground/pilot/`:

| File | What it is |
|---|---|
| `photo.jpg` | The photograph |
| `model_<heading>.jpg` | The model from the same spot, looking that way (degrees clockwise from north). Three views around the camera's direction if the photograph recorded one, otherwise six all round |
| `sheet.jpg` | The photograph above the model's views |
| `info.json` | Author, licence, date, position, and the camera direction if recorded |

The viewer pictures need a Chrome, Edge or Chromium already installed on the
machine (set `BOSTON_BROWSER` to its path if it is not found). Nothing else is
installed. `--no-capture` writes the photographs and sheets alone.

In Claude Code, `/ground "Boston Public Garden"` checks each sheet and writes a
`findings.json` beside it. [.claude/commands/ground.md](.claude/commands/ground.md)
is the procedure and the format of that file.

Most findings point at something the pipeline does not draw yet (fences,
railings, gates, the look of a bench or a bridge), not at one wrong fact. They
never become fixes by themselves. The photographs are visual reference only, so
a finding shows what to look into, and the fix needs a source the project may
use as data (see below). `info.json` keeps each photograph's licence and author.

### Working out coordinates

Fix files use longitude and latitude. To place something relative to a note,
work in world metres and convert:

```sh
python -c "from boston_pipeline import world; print(world.to_lonlat(-595.5, -491.7))"   # (lon, lat)
python -c "from boston_pipeline import world; print(world.to_local(-71.0702, 42.3544))" # (x, y)
```

Heading 0 is north (+y), 90 is east (+x). A point `d` metres ahead of someone
at `(x, y)` facing `heading` is `(x + d·sin(heading), y + d·cos(heading))`.

The geometry of existing features is in `data/raw/pilot/osm.json` (Overpass
JSON: each element has `type`, `id`, `tags` and `geometry`).

### Where correct information may come from

The contributor's own observation and photos, and the open sources listed in
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md), including the City of Boston tree
inventory (ids like `bprd:1416`). Two aerial surveys may be used as data:
USGS NAIP, 7 July 2023, leaf-on (what `audit` draws on), and MassGIS aerial
imagery, 15 cm, leaf-off, spring 2019 to 2025. Name the survey and date in
`why`, and credit MassGIS when using its imagery. Ground photographs, including openly licensed ones,
are visual reference only: look at them, but do not use them as textures or as
data. Never Google Maps, Street View or Google Earth. If the note does not say
how the author knows, ask on the ticket before inventing detail.

### Fixing a tree

Trees come from the city inventory (`bprd:<n>`) and from OpenStreetMap (`n<n>`);
`inspect` shows both. Heights and crown widths are estimated from species and
trunk size, so they are often a little off.

- A tree that is gone: delete it by id in a `.toml` file.
- A wrong species: set `species` (botanical, for example `Quercus palustris`)
  and `species:en` (the common name, `Pin oak`). The species decides the shape.
- A measured size: set `height` or `diameter_crown` (metres) on the tree. A
  tree with a `diameter_crown` and neither `height` nor a trunk size gets its
  height from the crown, so a tree seen only in a photograph looks right. Tree
  heights cannot be read from aerial photographs.
- A missing tree: add a `.geojson` point tagged `natural=tree`, with `species`
  and `circumference` if known.

## What the pipeline draws today

Tags not listed here are kept in the data and ignored when drawing.

| Thing | Tags read |
|---|---|
| Buildings | `building`, `building:part`, `height`, `building:levels`, `min_height`, `building:min_level`, `building:colour`, `roof:colour`, `building:material` |
| Roads | `highway` (vehicle classes), `lanes`, `width`, `parking:left/right/both`, `area=yes`, `name` |
| Paths | `highway=footway/path/pedestrian/steps/cycleway`, `width`, `area=yes`; `railway=platform`; `man_made=pier`. Coloured by `surface`: `asphalt`, `concrete`, `paving_stones`, `brick`, `sett`, `wood`, `gravel`, `dirt`, `sand`, `metal` (and common synonyms). No `surface` is plain "paved"; a pier is wood. |
| Footbridges | A `highway=footway/path/cycleway/pedestrian` way with `bridge` set, at least 4 m long, that crosses water and is not a road bridge's sidewalk (`footway=sidewalk` or `crossing`, or running beside a carriageway). Drawn as a raised deck with railings over the water only: the ends on the banks stay ordinary path, so paths along the bank still cross. A lamp mapped in the water beside it stands on a stone pier. Other bridges are still drawn at ground level |
| Rail | `railway=rail/light_rail/subway/tram` |
| Water | `natural=water`, `waterway=dock/riverbank`, `natural=coastline`. A water area with `edge=kerb` also gets a low stone kerb along its outer shore (`edge:width`, `edge:height`; 0.4 m wide and 0.15 m high if absent). This is the project's own tag, set through a fix file: OpenStreetMap has no agreed tag for a pond's masonry edge |
| Grass and planting | `leisure=park/garden/pitch/playground/common/dog_park/golf_course`, `landuse=grass/forest/meadow/cemetery/recreation_ground/village_green/flowerbed`, `natural=wood/scrub/grassland`. Told apart as lawn, pitch, artificial turf, hard court, playground surface, bare dirt, garden, scrub, wood and meadow, using `surface`, `leisure`, `landuse` and `natural` |
| Planted beds | `leisure=garden` with `garden:type=flowerbed`, or `landuse=flowerbed`. Drawn in the `beds` layer: soil at the edge, planting 0.35 m high |
| Trees | a point tagged `natural=tree`, from OpenStreetMap, the city inventory or a fix. `species`, `genus`, `species:en`, `leaf_type` (shape and colour); `height`, `diameter_crown`; `circumference`, `denotation` (trunk size) |
| Park furniture | Points: `amenity=bench` (`backrest`, `seats`, `length`), `highway=street_lamp` (`lamp_mount`, `support`, `height`), `amenity=waste_basket`, `amenity=drinking_water`, `man_made=flagpole`, `barrier=bollard`, `barrier=block`. Also `direction`, `colour`, `material` |
| Barriers | Lines: `barrier=fence`, `chain`, `wall`, `city_wall`, `retaining_wall`, `hedge`, `guard_rail`, `kerb`, `bollard` (a row of bollards). `height`, `width`, `colour`, `material` or `wall` (stone, granite, dry_stone, concrete, brick). Default heights and widths are in `BARRIERS` in `features.py`. `barrier=gate` ways are not drawn. Drawn in the `barriers` layer. A barrier is opened where a path crosses it and cut where it lies in a carriageway, inside a building or under a raised footbridge deck; a path that only runs beside it does not cut it |
| Monuments | Points: `historic=monument` (drawn as a statue when `memorial` or `artwork_type` says it is one); `historic=memorial` or `tourism=artwork` with `memorial` or `artwork_type` (`statue`, `bust`, `plaque`, `obelisk`, `column`, `stele`, `war_memorial`; anything else becomes a sculpture or memorial stone); `amenity=fountain` (`fountain=nozzle`, `diameter`). `height` and `name` are used. These are placeholders of the right kind and size, not likenesses |
| Named places (data only) | `shop`, `amenity`, `tourism`, `historic`, stations, with a `name` |
| Traffic signals (data only) | `highway=traffic_signals` |

Tree height and crown width are estimates from species and trunk size unless
`height` or `diameter_crown` is set. Willows, and any tree whose common name
says "weeping" or whose botanical name has the cultivar 'Pendula', are drawn as
a broad dome with a hanging skirt that a walker can stand under (`Betula
pendula`, silver birch, is not one). Trees are not solid; everything else is.

Anything with `tunnel`, `indoor`, `location=underground` or a negative `layer`
is skipped. Where ground layers overlap, the order is roads, rail, paths,
water, planted beds, grass, plain ground. Where grass areas overlap, the
smaller one shows (a pitch inside a lawn inside a park).

## Rules

- Never commit `data/` or `web/public/tiles/`. Both are generated.
- Never change the origin, projection or tile size in `world.py`.
- Tile layer names (`ground`, `green`, `paths`, ...) are a contract with
  clients; add layers, do not rename them. Document new ones in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- Defaults and tag interpretation belong in `features.py`, with a comment.
- A pipeline change needs a test in `pipeline/tests/`.
- Look at the result in the viewer before calling a visible change done.

## Where things are

| Path | Contents |
|---|---|
| `fixes/` | Corrections applied on top of the source data |
| `areas.toml` | Areas that can be built, and their landmarks |
| `pipeline/boston_pipeline/features.py` | Tags to typed features: what is a road, how tall is a building |
| `pipeline/boston_pipeline/tiles.py` | Features to tile geometry and data files |
| `pipeline/boston_pipeline/fixes.py` | Loads and applies `fixes/` |
| `pipeline/boston_pipeline/trees.py` | Tree species and trunk size to shape, size and colour |
| `pipeline/boston_pipeline/inventory.py` | Downloads the city tree inventory and merges it with OpenStreetMap trees |
| `pipeline/boston_pipeline/furniture.py` | Shapes for benches, lamps, bins, statues, fountains and the rest |
| `pipeline/boston_pipeline/bridges.py` | Raised footbridge decks and railings |
| `pipeline/boston_pipeline/barriers.py` | Shapes for fences, walls, hedges, chains, guard rails, kerbs and rows of bollards |
| `pipeline/boston_pipeline/lookup.py` | The `inspect` command |
| `pipeline/boston_pipeline/audit.py` | The `audit` command: aerial photographs, overlays, canopy detection |
| `.claude/commands/audit.md` | How to check an audit and write `findings.json` |
| `pipeline/boston_pipeline/ground.py` | The `ground` command: Commons photographs, sheets |
| `pipeline/boston_pipeline/capture.py` | Serves the built viewer locally and gets pictures from it through a headless browser |
| `web/src/capture.ts` | The viewer's capture mode (`?capture=`) |
| `.claude/commands/ground.md` | How to check a ground sheet and write `findings.json` |
| `web/src/city.ts` | Tile loading, collision, picking |
| `web/src/notes.ts`, `notepad.ts` | Notes: data format, tickets, on-screen form |
| `docs/ARCHITECTURE.md` | Coordinate system and file formats |

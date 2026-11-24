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
pip install -e "./pipeline[dev]"
boston-pipeline fetch pilot          # download source data once (cached in data/)
boston-pipeline build pilot          # build tiles into web/public/tiles
boston-pipeline inspect LAT,LON      # what is at a spot, with ids and tags
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
| `looking_at` | The first thing straight ahead of them: its position, `distance`, the `layer` it is drawn in, and for buildings the `building` id, name and height. Absent if they were looking at the sky. |
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
   layer it is drawn in.
3. **Decide what kind of problem it is.** This matters more than anything else:

   | What is wrong | Where the fix goes |
   |---|---|
   | The data is wrong or missing for one place: a wrong height, a missing path, a pond with the wrong outline, a tree that is gone | A file in `fixes/` |
   | The data is right but drawn wrongly everywhere: all paths too wide, lawns the wrong colour | `pipeline/boston_pipeline/features.py` (interpretation) or `tiles.py` (geometry) |
   | The data has it but nothing draws it: benches, fountains, fences, statues | New interpretation in `features.py` and geometry in `tiles.py`, with a test |
   | It is about look or behaviour in the browser: lighting, shading, controls | `web/src/` |

   If a note needs a pipeline change, say so on the ticket rather than faking it
   with a one-off fix file.
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
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Never Google Maps, Street View or
Google Earth. If the note does not say how the author knows, ask on the ticket
before inventing detail.

## What the pipeline draws today

Tags not listed here are kept in the data and ignored when drawing.

| Thing | Tags read |
|---|---|
| Buildings | `building`, `building:part`, `height`, `building:levels`, `min_height`, `building:min_level`, `building:colour`, `roof:colour`, `building:material` |
| Roads | `highway` (vehicle classes), `lanes`, `width`, `parking:left/right/both`, `area=yes`, `name` |
| Paths | `highway=footway/path/pedestrian/steps/cycleway`, `width`, `area=yes`; `railway=platform`; `man_made=pier` |
| Rail | `railway=rail/light_rail/subway/tram` |
| Water | `natural=water`, `waterway=dock/riverbank`, `natural=coastline` |
| Grass and planting | `leisure=park/garden/pitch/playground/common/dog_park/golf_course`, `landuse=grass/forest/meadow/cemetery/recreation_ground/village_green/flowerbed`, `natural=wood/scrub/grassland` |
| Trees | a point tagged `natural=tree` |
| Named places (data only) | `shop`, `amenity`, `tourism`, `historic`, stations, with a `name` |
| Traffic signals (data only) | `highway=traffic_signals` |

Anything with `tunnel`, `indoor`, `location=underground` or a negative `layer`
is skipped. Where ground layers overlap, the order is roads, rail, paths,
water, grass, plain ground.

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
| `pipeline/boston_pipeline/lookup.py` | The `inspect` command |
| `web/src/city.ts` | Tile loading, collision, picking |
| `web/src/notes.ts`, `notepad.ts` | Notes: data format, tickets, on-screen form |
| `docs/ARCHITECTURE.md` | Coordinate system and file formats |

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
boston-pipeline fetch boston         # the published area, Back Bay to the Brookline Reservoir and Coolidge Corner: 495 tiles, fetched in strips (a few minutes); also downloads the Town of Brookline's two layers and the City of Boston's building survey. After changing an area's bbox fetch with --force: the cache does not know the box
boston-pipeline build boston         # writes the same web/public/tiles; the pilot's tiles come out identical
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
   | The data has it but nothing draws it: signs, kiosks, steps as steps | New interpretation in `features.py` and geometry in `tiles.py` (or a module like `furniture.py` or `barriers.py`), with a test |
   | It is about look or behaviour in the browser: lighting, shading, controls | `web/src/` |

   If a note needs a pipeline change, say so on the ticket rather than faking it
   with a one-off fix file.

   Trees are a data problem, so they go in `fixes/` by id (see "Fixing a tree"
   below). Fences, walls, hedges, chains, guard rails, kerbs and bollards are
   downloaded and drawn, so a wrong or missing one is a fix file with ordinary
   `barrier=` tags. Gates are drawn open; a wrong or missing one is a fix with
   ordinary `barrier=gate` tags (a `.geojson` line across the opening).

   A bench that faces the wrong way is a data problem too. Which way an
   unmarked bench faces is a guess (see the table of what is drawn), so set
   `direction` on it by id in a `.toml` file: degrees clockwise from north, or
   a compass point such as `NE`, the way a person sitting on it looks.
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
inventory (ids like `bprd:1416`) and, in the `boston` area, the Town of Brookline's
tree inventory and building footprints (ids like `brk:8838`; used on the project
owner's decision although they state no licence: docs/DATA_SOURCES.md) and the City of Boston
Planning Department's building survey (heights; PDDL). Two aerial surveys may be used as data:
USGS NAIP, 7 July 2023, leaf-on (what `audit` draws on), and MassGIS aerial
imagery, 15 cm, leaf-off, spring 2019 to 2025. Name the survey and date in
`why`, and credit MassGIS when using its imagery. Ground photographs, including openly licensed ones,
are visual reference only: look at them, but do not use them as textures or as
data. Never Google Maps, Street View or Google Earth. If the note does not say
how the author knows, ask on the ticket before inventing detail.

### Fixing a tree

Trees come from the city inventory (`bprd:<n>`), the Town of Brookline's inventory
(`brk:<n>`, Brookline only) and from OpenStreetMap (`n<n>`);
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
| Buildings | `building`, `building:part`, `height`, `building:levels`, `min_height`, `building:min_level`, `building:colour`, `roof:colour`, `building:material`. `building=roof` with `roof:shape` hipped, gabled or pyramidal and no `min_height` is an open pavilion: posts (`colour` or `building:colour`) and a shaped roof (`height` to the ridge, `roof:height`); other roofs stay slabs. The viewer paints each wall's facade in a shader (`web/src/facade.ts`) from the wall's colour and its UVs: reddish is brick (coursing, recessed sash windows, stone sills and lintels, a cornice, shop fronts), cool grey-blue on 8 or more storeys is a glass curtain wall, other buildings of 8 or more storeys get ribbon windows between precast panels, and the rest is ashlar stone with moulded window surrounds. The upper part of a building on a podium (`min_height` above 0.5 m) gets no shop front. Nothing is tagged for this: the wall colour decides (`building:colour`, or `building:material` through the palettes in `features.py`). A `roof:shape` of gabled, hipped, half-hipped, pyramidal, skillion, gambrel, mansard (and `quadruple_saltbox`), round or dome is drawn as a roof (`roofs.py`) from `roof:height` (else `roof:levels` x 2.5 m, else about 30 degrees), `roof:direction` (a skillion slopes down toward it), `roof:orientation`, `roof:colour` and `roof:material`, and the walls stop at the eaves. A footprint that fills under 80% of its enclosing rectangle stays flat. A mansard omits the slope toward a neighbour, so a row is one roof. Buildings tagged `amenity=shelter` default to 3 m. In an area with `extra_sources = ["brookline"]` (the `boston` area), a building with no `height`, `building:levels` or `min_height` that a Town of Brookline footprint covers gets that footprint's height (`height` plus `height:source=Town of Brookline building footprints`; `height_source` is `town`). In an area with `extra_sources` listing `city_heights` (`pilot` and `boston`), a building with no `height`, `min_height` or `building:min_level`, not built after 2010 by its `start_date`, that the City of Boston's roof-break survey covers gets that survey's height (the part that covers most of its area; `height:source=City of Boston building roof breaks (2010 survey)`; `height_source` is `city`). A building with only `building:levels` takes it too where it is within 0.7 to 1.4 times what its levels give: a building the survey contradicts has been rebuilt and keeps its levels |
| Roads | `highway` (vehicle classes), `lanes`, `width`, `parking:left/right/both`, `area=yes`, `name`. Asphalt is a dark warm grey with slow variation and a few repair patches. Paint is drawn in the `markings` layer by `markings.py`: crosswalks from `footway=crossing` ways and `highway=crossing` nodes (`crossing:markings` ladder, zebra or lines, two plain lines if a marked crossing says nothing more; `unmarked` or `surface` gets none), double yellow centre lines on two-way roads, dashed white lane lines, and stop lines before signalised or stop-sign crosswalks. Nothing on service roads or roads under 5 m wide. Lines stop at junctions |
| Paths | `highway=footway/path/pedestrian/steps/cycleway`, `width`, `area=yes`; `railway=platform`; `man_made=pier`. Coloured by `surface`: `asphalt`, `concrete`, `paving_stones`, `brick`, `sett`, `wood`, `gravel`, `dirt`, `sand`, `metal` (and common synonyms). No `surface` is plain "paved"; a pier is wood. A pier area with a `colour` tag has its sides painted that colour (the Swan Boat landing). |
| Footbridges | A `highway=footway/path/cycleway/pedestrian` way with `bridge` set, at least 4 m long, that crosses water and is not a road bridge's sidewalk (`footway=sidewalk` or `crossing`, or running beside a carriageway). Drawn as a raised deck with railings over the water only: the ends on the banks stay ordinary path, so paths along the bank still cross. A lamp mapped in the water beside it stands on a stone pier. `bridge:structure=suspension` (set by a fix) draws lattice railings, granite pylons with globe lamps, cables and stone piers and abutments; the pylons stand at the lamps mapped in the water beside the deck, and those lamps are drawn as part of the bridge. The pylon is `assets/bridge-pylon.glb` when it exists (scaled to the lamp's height within 0.8 to 1.25), otherwise a procedural one. Other bridges are still drawn at ground level |
| Rail | `railway=rail/light_rail/subway/tram` |
| Water | `natural=water`, `waterway=dock/riverbank`, `natural=coastline`. A pond, lagoon or basin (`water=pond/lagoon/basin`, or water of no stated kind under 20,000 m²) is dark olive; rivers, canals, docks, the harbour and the sea stay blue. A water area with `edge=kerb` also gets a low stone kerb along its outer shore (`edge:width`, `edge:height`; 0.4 m wide and 0.15 m high if absent). This is the project's own tag, set through a fix file: OpenStreetMap has no agreed tag for a pond's masonry edge |
| Grass and planting | `leisure=park/garden/pitch/playground/common/dog_park/golf_course`, `landuse=grass/forest/meadow/cemetery/recreation_ground/village_green/flowerbed`, `natural=wood/scrub/grassland`. Told apart as lawn, pitch, artificial turf, hard court, playground surface, bare dirt, garden, scrub, wood and meadow, using `surface`, `leisure`, `landuse` and `natural`. Lawn, garden and woodland floor vary slowly in colour from place to place (vertex colours keyed to world position, so tiles agree at their edges) |
| Planted beds | `leisure=garden` with `garden:type=flowerbed`, or `landuse=flowerbed`. A polygon with holes gives ring beds round lawns (the Kelleher Rose Garden). Drawn in the `beds` layer: soil at the edge, planting 0.35 m high, mottled foliage with bare mulch and a few patches of bloom picked per bed |
| Trees | a point tagged `natural=tree`, from OpenStreetMap, the city inventory or a fix. `species`, `genus`, `species:en`, `leaf_type` (shape and colour); `height`, `diameter_crown`; `circumference`, `denotation` (trunk size). Drawn in a form chosen by species: round, oval, vase, columnar, small (a lumpy skin with leaf-clump lobes and a flared, forking trunk, smooth-shaded), weeping (a dome with hanging strands), conifer (tiers), pine (pads on a bare trunk), spire (a column), clipped (a dome). An unnamed `leaf_type=needleleaved` tree whose `diameter_crown` and `height` give a crown/height ratio of 0.65 or more is a clipped dome. A tree on the lawn of a named park, garden or common of 20,000 m² or more stands in a mulch ring (`tree_surround=none` removes it). Trunks are solid in the viewer: each tree's tile entry lists `trunk`, the drawn trunk's radius in metres at 1.3 m (absent for a clipped evergreen, which draws no trunk). Crowns and a willow's skirt stay passable. `brk:<ID>` trees come from the Town of Brookline inventory (DBH in inches, aged for trees planted after 1994); they merge with OpenStreetMap trees within 4 m, and OpenStreetMap trees are never dropped in a Brookline park |
| Park furniture | Points: `amenity=bench` (`backrest`, `seats`, `length`), `highway=street_lamp` (`lamp_mount`, `support`, `height`), `amenity=waste_basket`, `amenity=drinking_water`, `man_made=flagpole`, `barrier=bollard`, `barrier=block`. Also `direction`, `colour`, `material`. A bench with no `direction` is turned to face the nearest path within 20 m; inside a paved area (a plaza, a wharf) it looks in from the nearest edge; with water within 10 m on the side away from the path it looks at the water. With none of these it faces north. A street lamp that is not `lamp_mount=arm` is drawn from `assets/park-lamp.glb` (a slim black post carrying a white globe, 94 triangles) when it exists, otherwise procedurally. A lamp with `lamp_mount` bent_mast or arm, or taller than 5.5 m, is a street lamp: a tall pole with a bracket arm and a cobra head, turned to the nearest road if it has no `direction`. A bench is a slatted seat and back on cast-iron ends with armrests (backless if `backrest=no`) |
| Street furniture | `streetfurniture.py` places each thing on the verge beside the road it serves, off every carriageway and clear of walls, and draws nothing for a node on no road. `highway=traffic_signals`: a pole to the right of each approach, facing it (`traffic_signals:direction` picks the approach, else every inbound approach, one-way respected), a post head, or a mast arm (heads above 4.6 m) on carriageways of 9 m or more; none on cycleways or roads under 5 m. Footway crossings with `crossing=traffic_signals` or `crossing:signals=yes`: a pedestrian head at each end. `highway=stop` with `direction` forward or backward: a red octagon (none without a direction). `amenity=post_box`: blue (UPS brown), door facing away from the road. `amenity=bicycle_parking`: `capacity` gives up to 6 hoops; `bicycle_parking=bollard` a post and ring, `shed` or `lockers` a box. `highway=bus_stop`: a sign facing the buses; `shelter=yes` a glass shelter and `bench=yes` a bench, narrowed to clear walls, none if the map has the shelter as a building. Lenses are static, all three colours drawn. In the `furniture` layer; an object's `pos` is the pole, not the map node |
| Barriers | Lines: `barrier=fence`, `chain`, `wall`, `city_wall`, `retaining_wall`, `hedge`, `guard_rail`, `kerb`, `bollard` (a row of bollards). `height`, `width`, `colour`, `material` or `wall` (stone, granite, dry_stone, concrete, brick). Default heights and widths are in `BARRIERS` in `features.py`. `barrier=gate` ways are the opening in a fence or wall and are drawn open: a granite pier outside each end (panelled shaft, projecting cap, a pyramid, ball or urn) and iron leaves hinged on the piers and swung back, so the whole opening stays clear. One leaf up to 1.3 m wide, else two of at most 2.6 m. `height` (0.9 to 3 m), `colour` and `material` (`wood`, `wire`) are read. Ways that meet end to end are one gate. The leaves swing to the side with less path, road, water or building in the way, then to the lawn. `gate=ticket_barrier` and `turnstile` are skipped. A `fence` with `material` iron, steel or metal (and not a `fence_type` of chain_link, wire_mesh or similar) up to 1.8 m high is drawn as railings: pointed bars every 0.25 m between two rails on capped posts; a chain has ball-topped posts and a sagging chain. Drawn in the `barriers` layer. A barrier is opened where a path crosses it and cut where it lies in a carriageway, inside a building or under a raised footbridge deck; a path that only runs beside it does not cut it |
| Monuments | Points: `historic=monument` (drawn as a statue when `memorial` or `artwork_type` says it is one); `historic=memorial` or `tourism=artwork` with `memorial` or `artwork_type` (`statue`, `bust`, `plaque`, `obelisk`, `column`, `stele`, `war_memorial`; anything else becomes a sculpture or memorial stone); `amenity=fountain` (`fountain=nozzle`, `diameter`). `height` and `name` are used. `statue=equestrian` (set by a fix) is drawn as a horse and rider, `animal=*` as a low animal sized by `length`, `support=ground` as a statue with no pedestal; other statues stand on a stepped granite pedestal (base, die, cap). A `statue=equestrian` with bronze (or no) material and no `colour` is drawn from `assets/equestrian-figure.glb` (a hand-made horse and rider, 3,224 triangles) in the `figures` layer, on the procedural pedestal; marble, granite, a `colour` tag or `BOSTON_NO_ASSETS` give the procedural figure. These are placeholders of the right kind and size, not likenesses. Project tags (set by `fixes/`; `monuments.py` draws them): `pedestal=bell` or `shaft`, `figures=2` or `3`, `figure=child`, `animal=duck`, `dog`, `lion`, `panther`, `turtle`, a fountain with `figure` or `animal` carries a bronze figure, `fountain=nozzle` or `jet_water` throws jets, `artwork_type=lantern` is a Japanese stone lantern, and a `memorial=column` with `man_made=flagpole` and `flag:wikidata=Q42537` flies the US flag. A statue 1.6 m or taller gets a full figure (about 330 triangles) |
| Pitches and courts | `leisure=pitch` with `sport` (the first of tennis, basketball, soccer, american_football, volleyball, beachvolleyball, baseball, softball that fits; nothing for `multi`, no sport, more than three sports, a covered, indoor or rooftop pitch, or one mostly hidden). `sports.py` fits the sport's standard layout to the pitch's outline (standard sizes are interpretation, like a tree's size from its species): white lines in the `markings` layer (5 cm above the surface, not solid), nets as a mesh band and tape (a walker passes through), and objects `hoop`, `goal`, `goalpost`, `net` (the posts) and `backstop` in `furniture`. A diamond is drawn only where home plate can be placed: from an infield polygon (Fenway Park's `surface=dirt`/grass pair) or from the project tags `home_plate` (`"lat,lon"`), `direction` (home to second base, degrees), `base_distance` (60, 70 or 90 ft) and `infield=dirt`, set in `fixes/sports/` (25 diamonds, fitted to the infield dirt in the NAIP photograph of 7 July 2023): infield dirt, base paths, bases, plate, mound, a backstop. Where pitches overlap the smaller shows; paint stops at roads, paths, water, beds and buildings. A pitch for hard-court games (tennis, basketball...) with no `surface` is a hard court |
| Playgrounds and small park things | `leisure=playground` areas get seeded generic equipment (`playgrounds.py`: swings, slides, climbing towers, seesaws, spring riders, sandpits; the pieces say `generic=yes`, are never surveyed, and keep 1.5 m inside the outline and clear of buildings, paths, water and fences). A `playground=*` point is drawn instead and wins. A playground with a paving `surface` is grey, one of sand is sand. `landuse=allotments` and community gardens get plots in the `beds` layer (soil, rows of planting, sheds, bean poles) except where plots are already mapped. `leisure=bandstand` and gazebo shelters of 8 to 150 m² are open structures (round ones octagonal with a conical roof). Points: `amenity=bbq`, `leisure=picnic_table` or `amenity=picnic_table`, `amenity=table` with `sport=chess`, `amenity=toilets` (a hut, only where no building covers it) |
| Stadiums and ballparks | `stadium.py`, for any `leisure=stadium`, grandstand or bleacher part with a pitch within 30 m: raked stands built from the outline minus its parts (0.9 m rows, 0.4 m risers, stairs every 8 m; `stand=terrace\|wall\|deck\|none`, `seat:colour`, `seat:tread` override), a deck fascia on the field side, and a ballpark outside wall (brick, pilasters, arched gate openings and windows, a dark base, no shop fronts: the shader's wall style 1). A thin bleachers part with a `height` is a field wall with seats on top and a scoreboard (Fenway's Green Monster). Points: `baseball=foul_pole`, `man_made=floodlight` (with `min_height`, `direction`) or a floodlight mast, and landmark signs (`advertising=sign` or `billboard` with `tourism=attraction` and a `height` of 15 m or more; `size`, `colour`, `sign:face=triangle`: the Citgo sign, as a board with no lettering). `fixes/fenway/` sets Fenway Park's colours, foul poles, the three light towers the aerials show, and the sign |
| Named places (data only) | `shop`, `amenity`, `tourism`, `historic`, stations, with a `name` |
| Traffic signals (data only) | `highway=traffic_signals` |

Tree height and crown width are estimates from species and trunk size unless
`height` or `diameter_crown` is set. Willows, and any tree whose common name
says "weeping" or whose botanical name has the cultivar 'Pendula', are drawn as
a broad dome with a hanging skirt that a walker can stand under (`Betula
pendula`, silver birch, is not one). Tree crowns are not solid; a tree's trunk is. Everything else is solid. A tree standing inside a building's outline (a building of 2.5 m or more that starts at the ground, not an open roof) is dropped when the scene is built unless a fix put it there (`inventory.drop_trees_in_buildings`): in the `boston` area that was 190 inventory points on roofs and in yards a footprint covers. A park the city inventory does not cover (Columbus Park, the Greenway, the Cambridge parks) has no data trees: fixes add trees by their canopy in the NAIP photograph (`fixes/*/*-trees-*.geojson`: `natural=tree` with `diameter_crown`, no species, so they are drawn as round broadleaf trees; about 3,800 so far, found by a detector and checked by eye, with roughly half of the crowns caught). The Town of Brookline's records can outlive their trees (a rebuilt school, an emptied tree pit): check the leaf-off MassGIS photograph before trusting a trunk size.

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
- Hand-made models live in `assets/` (see [assets/README.md](assets/README.md)): vertex colours only, inside their
  triangle budget. The pipeline falls back to the procedural shape without them, and `BOSTON_NO_ASSETS=1` turns them
  off. Tests run procedurally unless the test module is named `test_asset*` (`pipeline/tests/conftest.py`).
  Assets are tabled (26 November 2026): the three that exist stay, but make new shapes procedurally, in code.
- The far-view copy of a wall takes its colour from the far tint in `web/src/facade.ts`; `far_wall_colour` in `lod.py`
  must stay in step with it, or distant buildings change colour where the copy swaps for the full tile.

## Where things are

| Path | Contents |
|---|---|
| `fixes/` | Corrections applied on top of the source data |
| `areas.toml` | Areas that can be built (`pilot`, the downtown district; `boston`, the published area from Back Bay to Coolidge Corner), and their landmarks |
| `pipeline/boston_pipeline/features.py` | Tags to typed features: what is a road, how tall is a building |
| `pipeline/boston_pipeline/tiles.py` | Features to tile geometry and data files |
| `pipeline/boston_pipeline/fixes.py` | Loads and applies `fixes/` |
| `pipeline/boston_pipeline/trees.py` | Tree species and trunk size to shape, size and colour |
| `pipeline/boston_pipeline/inventory.py` | Downloads the city tree inventory and merges it with OpenStreetMap trees |
| `pipeline/boston_pipeline/brookline.py` | The Town of Brookline's trees and building heights: download, field mapping, matching footprints to buildings |
| `pipeline/boston_pipeline/furniture.py` | Shapes for benches, lamps, bins, statues, fountains and the rest |
| `pipeline/boston_pipeline/monuments.py` | People, animals, pedestals, fountain figures, the lantern and the flag |
| `pipeline/boston_pipeline/extras.py` | The hook through which `sports.py`, `playgrounds.py` and `stadium.py` add kinds of object (`SHAPES`, `interpret(tags)`, `objects(scene)`) without editing the shared modules |
| `pipeline/boston_pipeline/sports.py` | Lines, nets, hoops, goals, backstops and baseball diamonds on pitches |
| `pipeline/boston_pipeline/playgrounds.py` | Playground equipment, allotment plots, bandstands and gazebos, grills, picnic and chess tables, toilet huts |
| `pipeline/boston_pipeline/stadium.py` | Raked stands, field walls, foul poles, light towers, landmark signs and the ballpark wall |
| `pipeline/boston_pipeline/cityheights.py` | The City of Boston's surveyed building heights (roof-break parts, 2010): download, matching, `height_source` "city" |
| `pipeline/boston_pipeline/streetfurniture.py` | Shapes and placement of traffic signals, stop signs, post boxes, bicycle racks and bus stops |
| `pipeline/boston_pipeline/markings.py` | Road paint (crosswalks, centre and lane lines, stop lines) and the asphalt colour |
| `pipeline/boston_pipeline/roofs.py` | Pitched roofs from the roof tags |
| `pipeline/boston_pipeline/assets.py` | Reads the hand-made models in `assets/` and places them |
| `assets/` | Blender models (`.glb`, their sidecars and the scripts that make them), `build.py`, `preview.py`; see its README |
| `pipeline/boston_pipeline/facing.py` | Which way a bench with no `direction` faces |
| `pipeline/boston_pipeline/bridges.py` | Raised footbridge decks and railings, and the suspension footbridge's pylons and cables |
| `pipeline/boston_pipeline/pavilions.py` | Open pavilions on posts and the painted edge of a dock |
| `pipeline/boston_pipeline/barriers.py` | Shapes for fences, iron railings, walls, hedges, chains, guard rails, kerbs, rows of bollards and gates |
| `pipeline/boston_pipeline/groundlook.py` | Slow colour variation in lawns, planted beds' foliage and mulch, and tree mulch rings |
| `pipeline/boston_pipeline/lookup.py` | The `inspect` command |
| `pipeline/boston_pipeline/audit.py` | The `audit` command: aerial photographs, overlays, canopy detection |
| `.claude/commands/audit.md` | How to check an audit and write `findings.json` |
| `pipeline/boston_pipeline/ground.py` | The `ground` command: Commons photographs, sheets |
| `pipeline/boston_pipeline/capture.py` | Serves the built viewer locally and gets pictures from it through a headless browser |
| `web/src/capture.ts` | The viewer's capture mode (`?capture=`) |
| `.claude/commands/ground.md` | How to check a ground sheet and write `findings.json` |
| `web/src/city.ts` | Tile loading, collision, picking |
| `pipeline/boston_pipeline/lod.py` | The simplified far-view copy of each tile (`lod1/` super-tile files) |
| `web/src/lod.ts` | Loads the far copy and shows a cell only while its full tile is not loaded. Address switches: `?lod=0`, `?lod=tint`, `?radius=`, `?fog=` |
| `web/src/trunks.ts` | Solid tree trunks: cylinders from each tile's per-tree trunk radius, tested in `City.collide`; not drawn, not pickable |
| `web/src/facade.ts` | Facade shader: brick, stone, ribbon-window and curtain-wall facades from a wall's UVs and colour; no textures |
| `web/src/notes.ts`, `notepad.ts` | Notes: data format, tickets, on-screen form |
| `docs/ARCHITECTURE.md` | Coordinate system and file formats |

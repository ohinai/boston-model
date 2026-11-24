# Roadmap

The aim is a city that feels real to walk through. Work is ordered by what a
person on foot notices most. Sources named here are described in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Now: parks and gardens

Boston Common and the Public Garden first, then the other parks in the pilot
district. Done means someone who knows these places walks through and
recognises them at eye level.

### Done so far

- **Trees from the city inventory**, merged with OpenStreetMap's. The pilot has
  5,791 trees, 4,728 of them from the inventory. Each is drawn by species and
  trunk size. Heights and crown widths are estimates, not measurements.
- **Ground by surface.** Paths are coloured by what they are made of. Lawns,
  sports pitches, artificial turf, hard courts, playgrounds, bare dirt,
  gardens, scrub, woods and meadows are told apart. 100 planted beds are drawn
  as beds, not lawn.
- **Park furniture.** 1,191 lamp posts, 988 benches, 394 litter bins and 30
  drinking fountains.
- **Statues, sculptures, memorials, monuments and fountains** (48 statues, 36
  sculptures, 33 plaques, 17 fountains, 15 monuments), as placeholders of the
  right kind and size in the right place.
- **Footbridges over water** raised as arched decks with railings, with the
  water continuing beneath them. 35 are raised in the pilot, including the
  Public Garden's.
- **Checking against aerial photographs.** `boston-pipeline audit` draws the
  model on a leaf-on photograph, finds tree canopy from its infrared band, and
  proposes missing trees and measures crowns. A person or an assistant checks
  each cell before anything becomes a fix. The first audit of the Public Garden
  showed the footbridge was 4.9 m wide, not 3.2 m, and that roads and asphalt
  paths were drawn far too dark, so the palette was lightened. It also found
  rows of small clipped evergreens along the walks that are in neither the city
  inventory nor OpenStreetMap. These are being added as fixes, and crown
  widths of free-standing trees set from measurements, all from the photograph.
- **Know where you are.** The viewer shows the name of the park you are in, and
  notes record it and what you were looking at, down to the tree or statue.

### Still to do

**Draw what the data has, and what it does not**

- Fences, walls, hedges and gates. They are not in the downloaded data yet, so
  most will come from notes and fixes.
- Likenesses for statues, and hand-made models of landmarks, in place of the
  placeholders.

**Checking against photographs**

- Audit Boston Common and the rest of the pilot's parks, and turn what the
  checks confirm into fixes.
- Build the MassGIS leaf-off imagery into the audit. With the leaves off, path
  widths, beds, benches and trunk positions can be checked, which the leaf-on
  photograph cannot show.

**Trees that read as trees**

- Measured heights and crown widths, from the 2021 USGS lidar point cloud. This
  needs a reader for LAZ files. Heights cannot be read from the aerial
  photographs, which only give crown width and whether a crown exists.
- Trunks you can bump into. Trees are not solid yet.
- More varied, less blocky canopies.

**Water**

- The Public Garden lagoon with a soft bank instead of a wall, and the Swan
  Boat dock.
- Frog Pond on the Common, which is not mapped as water.

**Ground**

- Real terrain from USGS elevation, so the Common slopes up to Beacon Street.
  This is the largest single item and benefits the whole model.
- Grass that is not one flat colour.

After that, parks improve through notes: walk, press N, fix.

## Next: the rest of the pilot district

- **Measured building heights.** Fill untagged heights from the city's
  roof-break dataset. 48% are a guess today.
- **Roof shapes.** Read `roof:shape`, then take detailed roofs from the city's
  3D model where it has them.
- **Bridges and tunnels.** Footbridges over water are raised already. Raise
  bridges over roads and rail, road bridges and elevated roads to their real
  level and let walkers pass under them. Open up building passages.
- **Sidewalks.** Use mapped sidewalks and the city's sidewalk inventory instead
  of treating everything beside a road as pavement.
- **Street furniture.** Lamp posts, benches and bins are drawn from
  OpenStreetMap. Traffic signals, the city's street lights and hydrants are not,
  and should come from the city inventories.
- **Facades.** Storey counts and materials from data; then a way to contribute
  a hand-made facade or a whole landmark model.

## Then: the whole city

- **Pinned source data.** Build from a dated snapshot rather than a fresh
  download, so feature ids cannot shift under the fixes that refer to them.
- **Regional extract.** Read OpenStreetMap from the Geofabrik extract rather
  than the Overpass API.
- **Levels of detail.** Simplified parent tiles so a skyline is cheap to draw
  and only nearby tiles load in full.
- **Smaller tiles.** glTF quantisation and mesh compression.
- **Hosting.** Somewhere that can hold a whole city of tiles, since they do not
  belong in git.
- **Globe placement.** Place each tile on the globe individually so the model
  follows the Earth's curve.

## Later: a living city

- **Transit.** MBTA trains and buses running to the schedule, then to the live
  feed.
- **Traffic signals** that cycle, then vehicles that obey them.
- **Places you can enter or use.** Shops and stations as interactive points.
- **Time of day and weather.**

## Alongside

- **Notes without a GitHub account.** A small relay service that files tickets
  on a visitor's behalf.
- **Touch controls** so the viewer works on phones.
- **Unreal Engine project** that loads the published tiles through Cesium for
  Unreal, with a walking character. See [UNREAL.md](UNREAL.md).

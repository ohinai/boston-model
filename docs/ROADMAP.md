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
- **Checking from the ground.** `boston-pipeline ground` collects openly
  licensed photographs taken inside a park and shows the model from the same
  spots, standing on the ground, so what a walker would notice can be compared.
  The first run covered the Public Garden. It also found a bug: planted beds had
  no ground under the planting, so a walker who arrived inside one fell through
  the world. Beds now have soil under all of them, and a walker arriving on foot
  starts a metre up and drops onto whatever is there, such as a bridge deck or a
  bed.
- **Fences, walls and hedges.** The pipeline had never asked OpenStreetMap for
  `barrier` features. It does now, and draws them in a `barriers` layer: 227
  fences (12.0 km), 90 walls (4.2 km), 52 retaining walls (4.9 km), 51
  post-and-chain fences (2.2 km), 19 guard rails (3.8 km) and rows of bollards
  (8 rows and 58 single bollards), plus 11 stone blocks. They are opened where
  paths cross them and cut where they lie in a carriageway, a building or under
  a footbridge deck. Gates are not drawn.
- **The lagoon's stone edge.** The Public Garden lagoon has a low stone kerb
  along its shore, set in `fixes/` with the project's own `edge=kerb` tag. It
  comes from MassGIS spring 2023 imagery, where a light band about 0.45 m wide
  sits on the shoreline at 64 of 100 places sampled.
- **Weeping trees.** Willows and other weeping trees are drawn as a broad dome
  with a hanging skirt a walker can stand under.
- **Water seen from the air.** Water used to break into stripes from high up
  because the depth buffer ran out of precision. The camera's near plane now
  moves out as it climbs.

  The fences, the lagoon's edge and the willows came out of the first ground
  check of the Public Garden (23 usable photographs of 29), where they were
  among the gaps seen most often.
- **Know where you are.** The viewer shows the name of the park you are in, and
  notes record it and what you were looking at, down to the tree or statue.

### Still to do

**Draw what the data has, and what it does not**

- Gates. A gate stands open for now, and is not drawn.
- Lamp posts in their real styles. The ones in the parks have globes.
- Pedestals and likenesses for statues.
- The skyline and landmark buildings, such as the Arlington Street Church
  spire.
- Check whether asphalt paths are now drawn too light.
- Likenesses for statues, and hand-made models of landmarks, in place of the
  placeholders.

**Checking against photographs**

- Audit Boston Common and the rest of the pilot's parks, and turn what the
  checks confirm into fixes.
- Run the ground check on Boston Common, and act on what the Public Garden
  check finds. Most of it will be things the pipeline does not draw yet.
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

- The Swan Boat dock. It is drawn as a brick box; it is an open pavilion on a
  floating dock.
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

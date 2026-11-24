# Roadmap

The aim is a city that feels real to walk through. Work is ordered by what a
person on foot notices most. Sources named here are described in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Now: parks and gardens

Boston Common and the Public Garden first, then the other parks in the pilot
district. Done means someone who knows these places walks through and
recognises them at eye level.

The map data for the two parks already holds far more than the model draws:
about 1,180 trees, 570 lamp posts, 400 benches, nearly 70 planted beds, 28
statues and artworks, 20 memorials, 7 monuments, 4 fountains, and a surface
for nearly every path.

**Draw what the data already has**

- Planted beds as beds, not lawn. Sports pitches, playgrounds and dog parks
  with their own surfaces.
- Path surfaces: asphalt, concrete, brick, paving stones, wood and gravel each
  look different.
- Benches, waste baskets, drinking fountains and lamp posts as shared models.
- Fountains, statues, memorials and monuments, at first as plinths of the
  right size in the right place.
- Fences, walls, hedges and gates. These are barely mapped, so most will come
  from notes and fixes.

**Trees that read as trees**

- Shape by leaf type (broadleaved or needleleaved is known for almost every
  tree; species for very few).
- Species, trunk size and missing trees from the city's tree inventory.
- More varied, less blocky canopies.

**Water**

- The Public Garden lagoon with a soft bank instead of a wall, the footbridge
  raised over the water, and the Swan Boat dock.
- Frog Pond on the Common.

**Ground**

- Real terrain from USGS elevation, so the Common slopes up to Beacon Street.
  This is the largest single item and benefits the whole model.
- Grass that is not one flat colour.

**Know where you are**

- Show the park's name while you are in it.

After that, parks improve through notes: walk, press N, fix.

## Next: the rest of the pilot district

- **Measured building heights.** Fill untagged heights from the city's
  roof-break dataset. 48% are a guess today.
- **Roof shapes.** Read `roof:shape`, then take detailed roofs from the city's
  3D model where it has them.
- **Bridges and tunnels.** Raise bridges and elevated roads to their real level
  and let walkers pass under them. Open up building passages.
- **Sidewalks.** Use mapped sidewalks and the city's sidewalk inventory instead
  of treating everything beside a road as pavement.
- **Street furniture.** Traffic signals, street lights and hydrants from the
  city inventories.
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

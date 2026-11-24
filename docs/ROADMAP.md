# Roadmap

The order is by what makes the city more true to life for someone walking it.
Sources named here are described in [DATA_SOURCES.md](DATA_SOURCES.md).

## 1. Make the pilot district accurate

- **Terrain.** Build a ground mesh from USGS 3DEP elevation and set buildings,
  streets and trees on it. Beacon Hill needs this most.
- **Measured building heights.** Fill untagged heights from the city's
  roof-break dataset, and report what is still guessed.
- **Roof shapes.** Honour `roof:shape` from OpenStreetMap (gabled, hipped,
  pyramidal), then take detailed roofs from the city's 3D model where it has them.
- **Bridges and tunnels.** Raise bridges and elevated roads to their real level
  and let walkers pass under them. Open up building passages.
- **Sidewalks.** Use mapped sidewalks and the city's sidewalk inventory instead
  of treating everything beside a road as pavement.

## 2. Cover the whole city

- **Regional extract.** Read OpenStreetMap from the Geofabrik extract rather
  than the Overpass API.
- **Levels of detail.** Add simplified parent tiles so a skyline is cheap to
  draw and only nearby tiles load in full.
- **Smaller tiles.** Apply glTF quantisation and mesh compression.
- **Hosting.** Publish built tiles somewhere that can hold a whole city (a
  release archive or object storage), since they do not belong in git.
- **Globe placement.** Place each tile on the globe individually so the model
  follows the Earth's curve.

## 3. Street-level detail

- **Street furniture.** Traffic signals, street lights, hydrants and trees from
  the city inventories, as shared models placed by the data layer.
- **Facades.** Storey counts and materials from data; then a way to contribute
  a hand-made facade or a whole landmark model for a specific building.
- **Lane markings and crossings** from OpenStreetMap lane and crossing tags.
- **Touch controls** so the viewer works on phones.

## 4. A living city

- **Transit.** MBTA trains and buses running to the GTFS schedule, and later to
  the live feed.
- **Traffic signals** that cycle, then vehicles that obey them on the street
  graph.
- **Places you can enter or use.** Shops and stations as interactive points
  from the data layer.
- **Time of day and weather.**

## Desktop

- **Unreal Engine project** that loads the published tiles through Cesium for
  Unreal, with a walking character. See [UNREAL.md](UNREAL.md).

## Good first contributions

- Add building heights, storey counts or roof shapes to OpenStreetMap for a
  block you know.
- Add a landmark to `areas.toml`.
- Draw traffic signals in the viewer from the `signals` list in each tile's
  data file.
- Make crossings visible where a footpath crosses a road.

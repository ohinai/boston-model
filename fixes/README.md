# Fixes

Every correction to the model lives in this folder as plain text. The pipeline
applies these files on top of the downloaded map data each time it builds, so a
fix is permanent, reviewable, and survives data updates.

Put files in any sub-folder under any name; one file per fix or per place keeps
pull requests from colliding. For example `fixes/public-garden/lagoon-bank.geojson`.

There are two kinds of file, shown below with made-up examples. Both describe things with
[OpenStreetMap tags](https://wiki.openstreetmap.org/wiki/Map_features), the same
vocabulary the source data uses. [AGENTS.md](../AGENTS.md) lists the tags the
pipeline currently draws.

## Change or remove something that exists: `.toml`

Each entry is keyed by a feature id. Find ids with
`boston-pipeline inspect <lat>,<lon>`, or in a note's position data.

```toml
["w123456789"]
why = "Height is 61 m per the 2010 city roof survey (Analyze Boston, roof breaks)."
at = [42.35875, -71.06190]            # optional: where it is, to re-find it if the id changes
set = { height = "61", "building:colour" = "#b5651d" }
unset = ["building:levels"]

["n987654321"]
why = "This tree came down in the March 2025 storm; seen on site."
delete = true
```

- `why` is required: what was wrong, and where the right value comes from.
- `set` adds or replaces tags. `unset` removes tags. `delete = true` removes the
  feature entirely.
- A feature can be changed in one file only.

Trees from the City of Boston tree inventory have ids like `bprd:1416` and are
changed or deleted in exactly the same way, for example with `delete = true` for
a tree that has been cut down. `inspect` shows these ids.

## Add something, or redraw it: `.geojson`

A standard GeoJSON `FeatureCollection` in longitude and latitude (WGS84). Each
feature's `properties` are its tags, plus:

- `why` (required): what this adds and where it comes from.
- `replaces` (optional): an id, or a list of ids, of existing features this one
  redraws. They are removed.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "natural": "tree",
        "leaf_type": "broadleaved",
        "why": "Large elm missing by the Arlington Street gate; seen on site, November 2026."
      },
      "geometry": { "type": "Point", "coordinates": [-71.07105, 42.35395] }
    },
    {
      "type": "Feature",
      "properties": {
        "highway": "footway",
        "surface": "gravel",
        "width": "2",
        "why": "Path along the north bank is missing; traced from the 2023 city orthophoto."
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [[-71.07010, 42.35420], [-71.06995, 42.35428], [-71.06975, 42.35431]]
      }
    }
  ]
}
```

Points, lines, polygons (with holes) and their multi forms are accepted. Added
features get ids like `fix:public-garden/lagoon-bank#0`: the file path without
its extension, then the feature's position in the file.

Any GeoJSON editor works, for example <https://geojson.io>. Do not trace from
Google Maps, Street View or Google Earth; see
[docs/DATA_SOURCES.md](../docs/DATA_SOURCES.md) for what may be used.

Fences, walls, hedges and the like can be added or corrected the same way, with
ordinary `barrier=` tags (`fence`, `wall`, `hedge`, `chain` and so on).

A water area can be given a low stone edge along its outer shore with a tag the
project made up, `edge=kerb`, because OpenStreetMap has no agreed tag for a
pond's masonry edge. `edge:width` and `edge:height` (metres) are optional.

```toml
["r1404341"]
why = "Narrow masonry edge round the lagoon, seen in the 2023 MassGIS aerial photographs."
set = { edge = "kerb", "edge:width" = "0.45" }
```

## Check a fix

```sh
boston-pipeline build pilot          # reports how many fixes applied, and any that matched nothing
boston-pipeline inspect <lat>,<lon>  # confirm the spot now reads the way you intended
```

Then look at it in the viewer. A malformed file stops the build with a message
naming the file and what to change.

A fix that "matched nothing" names a feature the current data does not contain:
either it is outside the area being built, or the feature was redrawn upstream
and has a new id. Use `at` and `inspect` to find the new one.

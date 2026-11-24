# Data sources

Every source the model uses or plans to use, with its terms. Facts here were
checked against the publishers' pages on 24 November 2026. Before adding a new
source, confirm its licence allows redistribution of derived work and add it to
this page in the same pull request.

## In use

### OpenStreetMap

- **Used for:** building footprints and heights, streets, paths, parks, water,
  coastline, trees, shops and other named places, traffic signals.
- **Licence:** [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
- **Credit required:** "© OpenStreetMap contributors", linked to
  <https://www.openstreetmap.org/copyright>. For games and simulations a
  credits screen is acceptable
  ([attribution guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines)).
- **Share-alike:** the built tiles are distributed under ODbL as well. Whether a
  3D mesh counts as a "produced work" or a "derivative database" is not settled
  by any OSM Foundation guideline, so this project takes the cautious reading.
  The pipeline being public satisfies the requirement to share the method.
- **How it is fetched:** the [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
  for single districts. Its usage policy asks for fewer than 10 000 queries and
  1 GB per day, and no parallel requests. For the whole city, use the
  [Geofabrik Massachusetts extract](https://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf)
  (about 300 MB) instead.

## Planned

These are open, verified and not yet wired into the pipeline. They are listed
roughly in the order they would improve accuracy.

### Terrain: USGS 3DEP elevation

- **Would fix:** the flat ground.
- **What:** 1 m elevation model from 2021 lidar, heights in metres above NAVD88.
- **Licence:** public domain
  ([USGS terms](https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map)).
- **Access:** the [3DEP ImageServer](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer)
  `exportImage` endpoint returns a GeoTIFF for a bounding box (up to
  8000 × 8000 pixels, 32-bit float). 1 m source tiles are listed by the
  [TNM Access API](https://tnmaccess.nationalmap.gov/api/v1/products).
- **Alternative:** [MassGIS lidar terrain](https://www.mass.gov/info-details/massgis-data-lidar-terrain-data),
  0.5 m, public domain.

### Building heights: Boston buildings with roof breaks

- **Would fix:** the 48% of buildings whose height is currently a guess.
- **What:** city building polygons split at roof-height changes, with
  `BLDG_HGT_2010`, `ROOF_ELEV_2010` and `GRND_ELEV_2010` attributes. Units are
  not confirmed (probably feet).
- **Licence:** Open Data Commons Public Domain Dedication and License (PDDL).
- **Access:** <https://data.boston.gov/dataset/boston-buildings-with-roof-breaks>;
  [feature service](https://gis.bostonplans.org/hosting/rest/services/Boston_Buildings/FeatureServer/9)
  in EPSG:6492.

### Building shapes: Boston Planning Department 3D model

- **Would fix:** flat roofs, on the buildings the city has modelled in detail.
- **What:** a citywide 3D model in 5 000 ft tiles. Per tile: SketchUp scene,
  per-building OBJ with JSON attributes, terrain mesh, ground plan.
- **Caveat:** most of it is block massing, not detailed roofs. Of about 165 000
  models, roughly 150 000 are extruded footprints or roofprints and about
  14 500 have real roof detail. These counts were read off the city's scene
  layer statistics and should be re-checked before relying on them.
- **Licence:** PDDL, stated on the download page.
- **Access:** <https://www.bostonplans.org/3d-data-maps/3d-smart-model/3d-data-download>.
  Files follow
  `https://maps.bostonplans.org/3d/Bos3d_BldgModels_<date>_OBJ/<tile>_BldgModels_OBJ.zip`;
  the date changes with each release.
- **Coordinates:** Massachusetts State Plane feet (EPSG:2249), NAVD88 feet, with
  models offset from a local origin at X 731 100 ft, Y 2 902 900 ft.

### Street detail: Analyze Boston

All PDDL, at `https://data.boston.gov/dataset/<name>`:

| Dataset | Use |
|---|---|
| `bprd-trees` | Street and park trees with species and trunk size |
| `traffic-signals` | Signal locations |
| `streetlight-locations` | About 74 000 street lights |
| `sidewalk-inventory`, `sidewalk-centerline` | Sidewalk geometry (2014) |
| `pedestrian-ramp-inventory` | Curb ramps |
| `fire-hydrants` | Hydrants (2019) |
| `parking-meters` | Meters |
| `2011-contours-1ft` | One-foot contours |

No open curb-line dataset was found.

### Transit: MBTA

- **What:** [GTFS schedule feed](https://cdn.mbta.com/MBTA_GTFS.zip) with route
  shapes, stops, station entrances and in-station pathways. Line and station
  geometry is also in the MassGIS
  [MBTA Rapid Transit shapefile](https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/mbta_rapid_transit.zip).
- **Licence:** the [MassDOT Developers License Agreement](https://cdn.mbta.com/sites/default/files/2023-08/mbta-massdot-develop-license-agreement.pdf).
  It allows use and redistribution with credit to MassDOT and no use of logos.
  The grant is revocable and does not mention derived works, so keep MBTA data
  as a separately built layer that can be removed, rather than merging it into
  other files.

### Other

- **MassGIS Building Structures (2-D):** statewide roof outlines, public
  domain, but with no height attribute, so of limited use here.
  <https://www.mass.gov/info-details/massgis-data-building-structures-2-d>
- **Overture Maps buildings:** ODbL, combines OSM with other footprints.
  Height fields exist but are optional and their coverage in Boston has not
  been measured. <https://docs.overturemaps.org/attribution/>

## Not usable

- **Google Maps, Street View, Google Earth and Photorealistic 3D Tiles.** Their
  terms forbid deriving data from them. Do not trace, measure or copy from them.
- **Anything with no stated licence.** Ask the publisher or leave it out. Two
  Analyze Boston datasets fall here today: `tree-canopy-change-assessment` and
  `2016-land-cover`.

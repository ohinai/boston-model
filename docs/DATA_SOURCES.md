# Data sources

Every source the model uses or plans to use, with its terms. Facts here were
checked against the publishers' pages on 24 November 2026. Before adding a new
source, confirm its licence allows redistribution of derived work and add it to
this page in the same pull request.

## In use

### OpenStreetMap

- **Used for:** building footprints and heights, streets, paths, parks, water,
  coastline, trees, fences, walls, hedges, chains, guard rails, kerbs and
  bollards (`barrier` features), shops and other named places, traffic signals.
- **Licence:** [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
- **Credit required:** "© OpenStreetMap contributors", linked to
  <https://www.openstreetmap.org/copyright>. For games and simulations a
  credits screen is acceptable
  ([attribution guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines)).
- **Share-alike:** the built tiles are distributed under ODbL as well. Whether a
  3D mesh counts as a "produced work" or a "derivative database" is not settled
  by any OSM Foundation guideline, so this project takes the cautious reading.
  The pipeline being public satisfies the requirement to share the method.
- **Corrections:** errors and gaps are fixed in this repository's `fixes/`
  folder, not upstream, and those files are ODbL as well.
- **How it is fetched:** the [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
  for single districts. Its usage policy asks for fewer than 10 000 queries and
  1 GB per day, and no parallel requests. For the whole city, use the
  [Geofabrik Massachusetts extract](https://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf)
  (about 300 MB) instead. Three public Overpass servers are tried in turn. A
  server whose copy of the map is more than 3 days old is passed over, and
  the download fails if none has a current copy.

### City of Boston tree inventory

- **Used for:** the position, species and trunk size of street and park trees.
  Drawn by species and trunk size; see [ARCHITECTURE.md](ARCHITECTURE.md).
- **Licence:** Open Data Commons Public Domain Dedication and License (PDDL).
- **What it gives:** position, botanical and common name, and trunk diameter at
  breast height in inches. It has no height and no crown width.
- **How it is fetched:** `boston-pipeline fetch <area>` pages through the
  Parks and Recreation (BPRD Trees)
  [feature service](https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/BPRD_Trees/FeatureServer/0)
  for the area's box and saves it as `data/raw/<area>/trees.json`. The same
  data is listed on data.boston.gov as `bprd-trees`.
- **Ids:** `bprd:<n>`, so fixes can change or delete a tree like any other
  feature.
- **Merging with OpenStreetMap:** an inventory tree and an OpenStreetMap tree
  within 4 m are one tree, and the inventory's is kept. In a park the inventory
  covers, OpenStreetMap trees it does not list are dropped. Elsewhere they are
  kept. In the pilot this gives 5,791 trees: 4,728 from the inventory and 1,063
  from OpenStreetMap only (1,480 duplicates merged, 267 dropped).
- **Caveats:** positions are good to about a metre. Species names are
  inconsistent between records, and some records are only a genus. Records for
  empty planting sites and stumps are ignored. It is a living dataset edited
  continuously, so a fresh download can differ from the one a build used. Trees
  missing from it or removed since are corrected in `fixes/`.

### Town of Brookline: tree inventory and building footprints

- **Used for:** the `boston` area only (the line `extra_sources = ["brookline"]` in its table in `areas.toml`). Trees: position,
  species and trunk size of Brookline's street and park trees, which the city's inventory does not cover. Buildings: the height
  of OpenStreetMap buildings that have no `height` or `building:levels` of their own. Code: `pipeline/boston_pipeline/brookline.py`.
- **Licence: none stated.** Neither ArcGIS layer carries a licence, terms of use or copyright text (checked 26 November 2026).
  They are used on the project owner's judgement of 26 November 2026, knowing there is no explicit grant. This is the one
  exception to "Anything with no stated licence" below. Credit: "Town of Brookline" (written into every tile and into
  `tileset.json` as "Town of Brookline tree inventory and building footprints"). To remove the data, delete the
  `extra_sources` line from `[boston]` in `areas.toml` and rebuild: nothing else changes and the pilot never used it.
- **Services (public, no key):**
  - Trees: https://services1.arcgis.com/Oknk0tvfHOElpgGU/arcgis/rest/services/Brookline_Tree_Viewer_Web_WFL1/FeatureServer/0
    (16,348 records in the whole town; 7,383 in the area's box).
  - Buildings: https://services1.arcgis.com/Oknk0tvfHOElpgGU/arcgis/rest/services/BuildingFootprint/FeatureServer/0
    (13,708 in the whole town; 7,218 in the box).
  `boston-pipeline fetch boston` pages through both (2,000 records a page) into `data/raw/boston/brookline_trees.json` and
  `brookline_buildings.json`.
- **Fields used.** Trees: ID, OBJECTID, ScientificName, CommonName, Cultivar, DBH, YR_PLANT, IsStump, CONDITION, PARKTREE,
  CG_NOTES. Buildings: TYPE, NUMSTORIES, BLDGHEIGHT, TOP_GL, COMMENT, COMMENTS.
- **Units, as checked.** DBH is in inches (trees planted in 2007 to 2009 read 1.5 to 2, a planting caliper; per-species medians
  match Boston's inventory, which is inches). BLDGHEIGHT is in feet and is the Town's storey count times 13 ft (2.55 storeys is
  33.15 ft): an assessed figure, not a survey. TOP_GL is a LiDAR top above ground in feet.
- **Trees.** Ids `brk:<ID>` (the 109 usable records with no ID get `brk:<1,000,000 + OBJECTID>`), so fixes can change or delete
  one like any tree. Stumps, dead trees (CONDITION 4) and records with neither species nor size are left out (47 of 7,383), and a
  DBH above 100 inches is a typing error (455 and 1,721 occur) and gives no size. For trees planted after the 1994 baseline
  the DBH is the caliper at planting and was never updated, so 0.3 inch a year since planting is added (this makes the median 3
  inches into 11, close to the 12 of the 1994 baseline trees). The 122 newest records have no ScientificName: it is taken from the same common name elsewhere in the layer.
  Records whose species is "Unknown" take it from the Town's own free-text note where that is one of 24 exact notes ("pine",
  "european beech"). Merging with OpenStreetMap: within 4 m is one tree and the Town's is kept (42 merges); unlike Boston's
  inventory it does not drop OpenStreetMap trees in parks (the Town's inventory is not a survey of every park tree, and 112 of the
  123 OpenStreetMap-only trees there are more than 15 m from any Town tree). The result is 7,336 trees.
- **Buildings.** BLDGHEIGHT in feet times 0.3048, or NUMSTORIES times the Town's typical height for that many storeys when the
  height is blank; a footprint marked "DEMOLISHED 2021" (526), a covered walkway or foundation is ignored; an outbuilding (TYPE
  OUTBLDG, whose figure is copied from the house) takes its LiDAR top when that is 7 to 20 ft; a claim above 60 ft is dropped
  where the LiDAR top is under 55% of it (32 of the 37 are Village Way rowhouses). A footprint matches a building when it overlaps
  half of the smaller of the two; several footprints give the figure that covers most of the building's area. OpenStreetMap's own
  `height`, `building:levels` and `min_height` win; fixes win over everything. The built tile entry says `"height_source": "town"`.
  In the `boston` build 4,095 of 24,946 buildings (16%) get a height this way: "from default" falls from 68% to 52%.
- **Cannot tell:** when a DBH was measured (every record is stamped 6 March 2026, a migration), whether the 1994 baseline trees
  have been remeasured, which stems a multi-trunk tree's DBH counts, or what is under the 183 footprints with no height.

### USGS NAIP aerial imagery

- **Used for:** checking the model against a leaf-on photograph: whether a tree
  crown exists and how wide it is, and the colour of surfaces. The `audit`
  command draws the model on it and finds tree canopy from its near-infrared
  band (see [ARCHITECTURE.md](ARCHITECTURE.md)). It is not built into the tiles.
- **What:** National Agriculture Imagery Program, 0.3 m, acquired 7 July 2023,
  four bands. The service gives natural colour, false colour and a vegetation
  index through its `renderingRule`. Only this one acquisition is served for
  Boston.
- **Licence:** public domain.
- **Access:** the
  [USGS NAIP ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer),
  `exportImage`. The audit asks for EPSG:26986, so the picture lines up with the
  model with no reprojection.
- **Caveats:** tall things lean a few metres away from the camera. Shadows can
  be mistaken for crowns, so work out which way they fall in each picture
  before judging. A tree planted since 2023 is not in it. Tree heights cannot
  be read from it.

### MassGIS aerial imagery

- **Used for:** measuring things that are clear with the leaves off, such as the
  width of the Public Garden footbridge and of paths, and telling evergreen
  trees from bare ones in early spring (a dense evergreen is solid, with a sharp
  shadow). It is not yet built into the `audit` command; that is on the
  [roadmap](ROADMAP.md).
- **What:** 15 cm, leaf-off, spring 2019, 2021, 2023 and 2025. Services
  `USGS_Orthos_2019`, `orthos2021`, `orthos2023` and
  `Massachusetts_Aerial_Imagery_2025` under
  `https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/`.
- **Licence:** the ArcGIS items say "No restrictions apply to these data", and
  that acknowledgement of MassGIS is appreciated. Credit MassGIS.
- **Access:** Web Mercator map tiles only, with no token. Level 20 is the native
  resolution.

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

### Tree heights and crowns: USGS 2021 lidar point cloud

- **Would fix:** the estimated heights and crown widths of trees. Today both
  come from species and trunk size, not from measurement.
- **What:** the 2021 airborne lidar for eastern Massachusetts, project
  `MA_CentralEastern_1_2021`. It was flown with the leaves off, so it shows the
  tops of branches rather than full summer crowns.
- **Licence:** public domain
  ([USGS terms](https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map)).
- **Access:** point cloud tiles listed by the
  [TNM Access API](https://tnmaccess.nationalmap.gov/api/v1/products).
- **Caveat:** the files are LAZ, so the pipeline needs a LAZ reader before it
  can use them.

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

All PDDL, at `https://data.boston.gov/dataset/<name>`. The tree dataset,
`bprd-trees`, is already in use (see above).

| Dataset | Use |
|---|---|
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

## Reference only

### Photographs

Ground photographs are for looking at, to see what a tree, statue or fence is
like. Openly licensed ones, such as those on Wikimedia Commons (mostly
CC BY-SA), are fine as visual reference. Do not use a photograph as a texture
or as data.

### Wikimedia Commons

- **Used for:** the ground check. `boston-pipeline ground` collects geotagged
  photographs taken inside a park and lays them beside the model's view from the
  same spot (see [ARCHITECTURE.md](ARCHITECTURE.md)). They are looked at, never
  used as data. Each sheet records the photograph's author and licence.
- **Access:** the Commons API, which needs no key. The pipeline keeps JPEG files
  from 2015 or later, inside the park's outline, and thins them so the park is
  covered evenly.
- **Licences:** mostly CC BY-SA and CC BY, a few public domain or CC0.
- **Coverage (24 November 2026):** 250 geotagged files within 230 m of the
  Public Garden's centre, and at least 500 around Boston Common (the query caps
  at 500). Of 325 geotagged files found over the Public Garden, 89 passed the
  filters.
- **Caveats:** positions are often tens of metres out, and most files do not
  record which way the camera faced.

Looked at the same day and not used: iNaturalist has thousands of plant
observations with photographs in the two parks, but almost all carry a
non-commercial licence. KartaView's API answers without a token, but its
coverage in the parks was not established.

### Mapillary

Mapillary has street-level photographs along most paths of the Public Garden
and across parts of Boston Common (seen on its public coverage map, November
2026). The photographs are CC BY-SA 4.0, so they may be looked at like any
other openly licensed photograph. They may not be used as data here: Mapillary's
terms (effective 15 February 2024) allow deriving data from its images for
OpenStreetMap and the Overture Maps Foundation only, and reserve its own
detected objects (benches, street lights and so on). Do not copy positions,
measurements or detected objects from Mapillary into `fixes/` without written
permission from Mapillary. Its API needs an access token from a registered
application.

Panoramax has a single image in the parks. For data, the aerial surveys above
are the practical check.

## Not usable

- **Google Maps, Street View, Google Earth and Photorealistic 3D Tiles.** Their
  terms forbid deriving data from them. Do not trace, measure or copy from them.
- **Anything with no stated licence.** Ask the publisher or leave it out. Two
  Analyze Boston datasets fall here today: `tree-canopy-change-assessment` and
  `2016-land-cover`. The one exception, by the project owner's decision of
  26 November 2026, is the Town of Brookline's tree inventory and building
  footprints (see above); it can be removed with one line in `areas.toml`.

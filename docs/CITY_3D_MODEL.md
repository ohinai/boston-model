# The City of Boston 3D model: what its building files hold

A feasibility study written on 27 November 2026 for drawing the city's building shapes (real roofs, stepped roofs) in the model. Nothing here is built yet; `docs/DATA_SOURCES.md` has the licence (PDDL) and `docs/ROADMAP.md` the plan. Paths in `<scratch>` were files on the author's machine and are not in the repository.

Studied 27 November 2026. Read-only on the repository. Everything below is measured unless it says "estimate" or "not established".

## 0. Bottom line

* It is usable, and the registration is good: the model sits on OSM to within about 1 m (median offset OSM minus city: +0.11 m east, +0.68 m north, which is the NAD83 versus WGS84 difference the repository already ignores on purpose).
* The real roofs are far more common than the repository believed. 86.5% of the 11,235 existing city buildings inside our bbox are made by CyberCity 3D and carry roof geometry, although the city labels almost all of them "LOD 1.5". Of the 1,508 existing buildings I parsed (two Back Bay / Beacon Hill tiles), 27% have pitched or sloped roofs, 31% have stepped flat roofs (2 to 9 levels: podium, tower, penthouses) and only 41% are a single flat extrusion. The city's own LOD label cannot tell them apart; `Model_Cred` plus a geometry test can.
* Three things in the data are traps: `Gnd_El_Ft` and `Height_Ft` are unreliable (do not use them); 9.5% of the building models in our bbox are demolished ("History"); the model has no building at all in most of Brookline.
* There is no key shared with OSM or with `Boston_Buildings/FeatureServer/9`. Matching has to be by geometry, and only 58% of models fall one to one inside an OSM polygon.
* Cost if we draw the upper shell (roof plus every vertical face above the lowest eave) for all 11,235 buildings: about 1.76 M triangles and 58 MB of GLB, against today's 546 k building triangles and 28 MB (whole project: 7.81 M triangles, 195 MB). Roughly half of that with simplification (my guess).

## 1. The claims in `docs/DATA_SOURCES.md`, checked

| Claim | Verdict | Evidence |
|---|---|---|
| 5,000 ft tiles | True | The tile index has 103 tiles, every one exactly 5,000.0 x 5,000.0 ft (`BOS_<col>_<row>`, columns A to O, rows 1 to 13; column A starts at x = 736,100 ft, row 1 ends at y = 2,972,900 ft; 103 of the 195 cells hold a tile). |
| Per tile: SketchUp scene, per-building OBJ with JSON attributes, terrain mesh, ground plan | True, incomplete | The download page and the web map also offer an aerial photograph (Nearmap 2023, 1 ft/pixel) and a vector ground plan (DXF). The JSON is not loose: each building is a nested zip (see section 3). |
| State Plane feet, EPSG:2249 | True | Header comment of every OBJ: `NAD83 / Massachusetts Mainland (ftUS)`, EPSG 2249. Unit confirmed as US survey feet: model footprints agree with OSM areas (median ratio 1.03). |
| NAVD88 feet | Not confirmed | The documentation PDF says so. Roof levels in the OBJ agree with the roof-break layer's `ROOF_ELEV_2010` to about 3 ft (section 6), so the vertical frame is consistent with that layer. But the `Gnd_El_Ft` attribute is not (section 6). |
| Offset from local origin X 731,100, Y 2,902,900 | True for OBJ vertices | Overlaid on OSM with exactly that origin and no rotation, the leftover offset is 0.11 m E, 0.68 m N (section 5). The "rotation 0.34 degrees clockwise" in the PDF is a SketchUp note and is not applied to the OBJs (it would move a building 65,000 ft from the origin by about 380 ft). The tile index field `BosShift_X` is not the OBJ offset (it differs by 1,100 ft in X from centre minus origin); ignore it. The PDF's latitude of the origin is 1 m off; irrelevant. |
| Most models are extruded footprints; about 14,500 of 165,000 have real roof detail | Counts right, conclusion wrong | Citywide catalogs (102 of 103 tile zips; BOS_A_9 returns 404): 169,030 models, of which 49,021 are `Wall`, 445 `Dock`, 121 `Bridge`, 119,443 `Building` (115,485 existing). Models with `Model_LOD` of 2 or more: 15,140 (the "14,500"; 14,428 of them are LOD 2.5). But CyberCity 3D made 32,230 buildings (17,811 labelled LOD 1.5 and 14,428 LOD 2.5), and the LOD 1.5 ones carry roofs too. Only `Model_Cred` = "Extruded from 2012 Photogrammetric Footprints and DTM by pbcGIS" (85,284 citywide, all LOD 1) are plain extrusions: 370 of 370 in my two tiles have one flat roof. |
| File pattern `.../Bos3d_BldgModels_<date>_OBJ/<tile>_BldgModels_OBJ.zip` | True | Current: `https://maps.bostonplans.org/3d/Bos3d_BldgModels_20260624_OBJ/BOS_H_4_BldgModels_OBJ.zip`. The older 20251204 folder is still downloadable. `<tile>` is `BOS_H_4`. |

## 2. The tile grid and where the real URLs are

The download page embeds nothing; it links to an ArcGIS Experience (item `d8f824291f294fddbb8d211047563c47`). Its configuration (`https://www.arcgis.com/sharing/rest/content/items/<id>/data?f=json`) points to the web map `cc79d1bd7bdd448c8cb891d344d015d9` ("3D Data Download Map", `boston.maps.arcgis.com`). That web map holds two tile-grid layers:

* `Bos3D_TileGrid_PopUps_20251204` (hidden; the service now answers "Invalid URL"), and
* `Bos3D_TileGrid_PopUps_20260624` = `https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/Bos3D_TileGrid_PopUps_20260624/FeatureServer/70` (item `8533388b888e41059c9944603141c276`, edited 29 June 2026). Fields: `tile_id`, `tile_label`, `grid_col`, `grid_row`, `town_code` (always BOS), `BosShift_X/Y`, `Lat`, `Lon`, `MASP_X/Y`. Spatial reference EPSG:2249.

The download links are in that layer item's popup template (`https://www.arcgis.com/sharing/rest/content/items/8533388b888e41059c9944603141c276/data?f=json`, key `popupInfo`), with `{tile_id}` as the placeholder. The release date appears in the layer name (20260624) and in the OBJ folder (20260624); the SketchUp folder has a different date (20260625), so do not derive one from the other.

Query that selects our tiles (tested; returns 22 tiles, 103 in total without the filter):

```
.../FeatureServer/70/query?where=1%3D1&geometry=-71.1470,42.3315,-71.0480,42.3640&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=tile_id&returnGeometry=false&f=json
```

An exact polygon intersection of the projected bbox gives 21 tiles; the server also returns `BOS_D_3`, a corner sliver (it does not hold a single building of ours). Release 20260624 sizes and census of the 22 tiles (existing building = status Current or Approved Demo, counted by centroid inside the bbox):

| Tile | fraction of tile inside bbox | zip MB | models in tile | existing buildings in bbox |
|---|---|---|---|---|
| BOS_D_3 | 0.00 | 3.5 | n/a | 0 |
| BOS_D_4 | 0.90 | 15.5 | 3163 | 1931 |
| BOS_D_5 | 0.90 | 13.7 | 2835 | 1531 |
| BOS_D_6 | 0.33 | 3.3 | 6 | 0 |
| BOS_E_3 | 0.00 | 5.1 | 439 | 0 |
| BOS_E_4 | 1.00 | 8.7 | 1363 | 934 |
| BOS_E_5 | 1.00 | 8.6 | 1322 | 1282 |
| BOS_E_6 | 0.37 | 4.0 | 214 | 4 |
| BOS_F_4 | 1.00 | 3.8 | 101 | 22 |
| BOS_F_5 | 1.00 | 10.5 | 1290 | 516 |
| BOS_F_6 | 0.36 | 16.0 | 3685 | 710 |
| BOS_G_4 | 1.00 | 6.7 | 768 | 214 |
| BOS_G_5 | 1.00 | 13.2 | 2205 | 674 |
| BOS_G_6 | 0.36 | 13.4 | 2926 | 237 |
| BOS_H_3 | 0.01 | 11.4 | 2057 | 5 |
| BOS_H_4 | 1.00 | 12.3 | 1774 | 797 |
| BOS_H_5 | 1.00 | 10.3 | 1541 | 711 |
| BOS_H_6 | 0.35 | 9.3 | 1856 | 207 |
| BOS_I_3 | 0.01 | 8.4 | 1238 | 6 |
| BOS_I_4 | 0.45 | 7.4 | 875 | 261 |
| BOS_I_5 | 0.45 | 9.8 | 1624 | 563 |
| BOS_I_6 | 0.16 | 13.2 | 2717 | 630 |
| total, 22 tiles | 12.67 tile-equivalents | 208.0 | about 34,000 | 11,235 |

16 tiles hold more than 6 of our buildings: 172.4 MB. A full 22-tile download is 208 MB (the observed speed was 12.3 MB in 2.1 s). Every tile zip carries about 3.3 MB of fixed overhead (viewer files, tile polygons). The whole city (102 files) is 898.7 MB. The release cadence is "semiannual" per the documentation; two releases are live (20251204, 20260624).

## 3. Internal structure (BOS_H_4 and BOS_H_5, downloaded: 12.31 MB and 10.29 MB)

Outer zip (1,810 entries in H_4):

* `catalog.csv` (1.6 MB, 45 columns), `catalog_jsonp.js` (5.1 MB, a JSONP-wrapped GeoJSON FeatureCollection with one footprint polygon per model in lon/lat with 6 decimals, plus every attribute), `ModelFinder.htm` and `catalog_resources/` (Leaflet viewer, tile polygons).
* `objz/BOS-XXXXX-XXXXX-XXXXX_OBJ.zip`, one nested zip per model (1,774 in H_4, 1,541 in H_5). The model id is `BOS-` plus 15 characters, not `BOS_` plus 7 as the PDF says.

Each nested zip holds exactly three files: `<Model_ID>.obj`, `materials.mtl`, `<Model_ID>.json`.

* **JSON** (a one-element array, 50 keys): `OBJECTID`, `Name` (often an address, e.g. "255 265 CHARLES ST", or null), `Status`, `Appear_Dt`, `Disapp_Dt`, `Model_ID`, `Model_Cred`, `Model_LOD`, `Survey_Src`, `Survey_Dt`, `Model_Note`, `Model_Dt`, `ModelBatch`, `Model_File`, `Centr_Lat/Lon/X_Ft/Y_Ft`, `Gnd_El_Ft`, `Z_Max_Ft`, `Z_MIn_Ft`, `Height_Ft`, `Tile_ID`, `Parcel_ID` (present for 94.7% of H_4 buildings), `Parcel_Lnk`, `Project_ID`, `A80_Status`, `StructType` (Building, Wall, Bridge, Dock), `StructUse` (always empty), `Struct_ID` (empty) and edit-tracking fields. No roof type, no OSM id, no id from the roof-break layer. The three ids that look usable do not overlap: model `OBJECTID` runs 1 to 164,622, layer 9 `OBJECTID` runs 644,205 to 772,795 (overlap: 0).
* **OBJ**: plain text from FME 2025.1.2. `v x y z` in US survey feet, offset from the origin above; z is up and absolute (NAVD88-like, not above ground); `vn` normals, one per face; no `vt` (UVs) in 3,260 of 3,315 models (the other 55 are hand-made LOD 3 and similar); every face is a triangle; one object per file. Vertex count per model (BOS_H_4, all types): median 26, p95 489, max 14,071.
* **Materials**: every `Kd` is white (1 1 1) apart from two exceptions in 80 files sampled (the multi-material files, 441 of 3,315, are sub-objects of pbcGIS walls, not colours). No colours, no textures. A building's wall and roof colour must keep coming from OSM tags or our palettes.
* **Shell**: CyberCity models are open shells with almost no floor faces (3,486 down-facing triangles of 282,935 in existing buildings, 2,196 of them in single-flat buildings) whose walls run straight down from the roof outline to below ground (lowest vertex of Back Bay towers: about 48 ft below z = 0), so the bottom has to be cut at our ground.
* **Footprints**: `catalog_jsonp.js` polygons agree with the plan silhouette of the OBJ (n = 405 models over 300 ft²: IoU median 0.995, centroid offset 0.00 ft), so footprints can come from that file without parsing OBJs. (Its `Centr_Lat/Lon` columns, in contrast, are 0.96 m north of a plain conversion of `Centr_X/Y_Ft`; do not use them for placement.)

Triangles per building (all 3,315 models in two tiles; fan triangulation counted from the faces): total 581,542 (H_4 327,628; H_5 253,914). By type: Building 1,753 models (392,429 triangles; median 49, p95 859, p99 1,862, max 33,284), Wall 1,524 (156,768), Bridge 29, Dock 9. Existing buildings only (status Current or Approved Demo, n = 1,508): 282,935 triangles, median 48, p95 780, p99 1,502. Of those, 31% face upward (roof surfaces, 87,205), 68% are vertical (192,244) and 1% face down. Parse time: 0.4 to 0.6 s per 700 to 800 models.

`Status` values among Buildings in the two tiles: Current 1,468, History 210, Approved Demo 40, Board Approved 18, Under Construction 11, Complete 6. In the whole bbox: Current 11,072, Approved Demo 163, History 1,197 (9.5%), Board Approved 98, Under Construction 27, Complete 65. `History` means the building is gone or replaced; drawing it would resurrect it.

The `Wall` models (7,462 inside our bbox, 49,021 citywide) are the city's measured walls ("Height established from measured infotech height. Depth established from DTM. Width from Infotech Wall Type"). Not used for roofs; possibly useful for our barriers layer, which I did not study.

## 4. Detailed roofs versus extruded blocks

How to tell, from the data, without drawing anything:

1. Metadata: `Model_Cred`. "Extruded from 2012 Photogrammetric Footprints and DTM by pbcGIS" (LOD 1) is a plain extrusion; "CyberCity 3D" (LOD 1.5, 2.5, rarely 3) and the Planning Department / BPDA hand-made models (LOD 3, projects) carry roof geometry. `Model_LOD` alone does not work: LOD 1.5 holds 435 pitched and 531 stepped roofs among 1,319 buildings in my two tiles.
2. Geometry, per parsed model (these thresholds are mine): pitched if at least 10% of the up-facing triangle area has normal z between 0.1 and 0.98; stepped if at least 2 distinct flat-roof levels (rounded to 0.5 ft) each hold at least 2% of the flat roof area; otherwise a single flat roof.

Numbers, existing buildings (Current + Approved Demo), BOS_H_4 + BOS_H_5, n = 1,508:

| Model maker | n | pitched or sloped | stepped flat | single flat | median triangles |
|---|---|---|---|---|---|
| CyberCity 3D (LOD 1.5 and 3) | 1,151 | 412 | 449 | 290 | 165 (pitched), 164 (stepped), 10 (single flat); 86 overall |
| pbcGIS extrusion (LOD 1) | 331 | 0 | 0 | 331 | 20 |
| Planning Dept / BPDA hand-made | 26 | 1 | 24 | 1 | 292 |
| all | 1,508 | 413 (27%) | 473 (31%) | 622 (41%) | 48 overall; 166 (pitched), 167 (stepped), 12 (single flat) |

Per tile: BOS_H_4 270 / 222 / 305, BOS_H_5 143 / 251 / 317 (pitched / stepped / single flat). Stepped buildings have on average 5.1 flat levels, many of them small rooftop boxes. I rendered two views (`<scratch>\city3d\out\render_beacon_hill.png` and `...\render_parksq.png`): the Beacon Hill rows have hipped and mansard roofs with dormers and chimneys, and the Park Square towers show podiums, set-backs, a curved tower and rooftop mechanical boxes.

Our whole bbox (exact counts from the catalogs; the roof split is extrapolated from the two tiles, so treat it as indicative): existing buildings 11,235 = CyberCity 9,716 (86.5%), pbcGIS extruded 1,336 (11.9%), other 183 (1.6%). By `Model_LOD`: 1: 1,453; 1.5: 9,411; 2: 8; 2.5: 317; 3: 46. Extrapolated split: about 3,485 pitched, 3,959 stepped, 3,791 single flat (66% detailed). The two tiles are Back Bay, Beacon Hill and the Financial District fringe; the Allston and Brookline-edge tiles (D, E) are probably richer in triple-decker gables, so the real split may differ.

LOD 2.5 is CyberCity 3D from a newer survey ("Aerial stereo survey, Quantum geospatial issued by MassGIS"); 330 are inside our bbox. I fetched three of them (single models, a few KB each) and they look like the others (10, 48 and 206 triangles).

## 5. Placement against OSM

Conversion: the world frame uses the same projection parameters as EPSG:2249, so no reprojection is needed. Checked on 1,000 random points: world metres = (x_ft + 731,100) x 1200/3937 - 236,000 east and (y_ft + 2,902,900) x 1200/3937 - 901,000 north, equal to `world.to_local` within 0.1 mm. (Equivalently x_world = (x_local_ft - 43,176.667) x 0.3048006096 and y_world = (y_local_ft - 53,130.833) x 0.3048006096.) pyproj is installed in the project environment but is not needed.

Method: OSM from `data/raw/boston/osm.json` (the centroid-offset statistics use the 23,592 building ways; the matching-share statistics use all 28,307 polygons: ways, building:part ways and relation outers), city footprints from the OBJ plan silhouette (existing buildings only, n = 1,508; 913 of them over 300 ft²), both in local feet, WGS84 read as NAD83 as `world.py` does.

* 392 pairs with IoU of at least 0.6: centroid offset OSM minus city median (+0.35 ft, +2.24 ft) = (+0.11 m E, +0.68 m N); scatter standard deviation 6.8 ft (2 m) from different footprint definitions. Per tile: H_4 (+0.5, +2.45) ft, H_5 (+0.06, +1.92) ft. A grid search for the one shift that maximises the mean IoU picks (0, +3) ft. The catalog's own lat/lon columns are 0.96 m north of a plain conversion of its feet columns, which is the same NAD83 to WGS84 effect. So add roughly (+0.1, +0.7) m to city geometry to sit on OSM in our world; the figure from the best-fit shifts of seven single buildings is similar (mean +0.25 m E, +0.65 m N).
* Seven buildings near the Public Garden (the Garden's own centre has no building; `inspect` there lists the Lagoon and paths only; `inspect` at 42.35134,-71.06740 shows One Charles as `building:part` w104205228, height 56.5, 19 levels, part of w485200676; at 42.35390,-71.07401 the First Lutheran Church w29945923, height 9.7 m "from city"):

| OSM way | City model | IoU raw / after best shift | best shift of the model (ft) | catalog `Height_Ft` (ft) | layer 9 parts (height ft) |
|---|---|---|---|---|---|
| w405398143 (unnamed, Park Plaza side) | BOS-00NUE-OWDJF-XX898 | 0.90 / 0.98 | (-2.5, +3.0) | 99 | 2 parts, 65.5 |
| w29806475 Saunders Building | BOS-BMEPQ-1S9NQ-ECPTX | 0.81 / 0.82 | (+2.0, 0.0) | 216 | 185.2, 176.4, 155.4, 155.4, 50.9 |
| w29802369 Park Square Building | BOS-KN9QC-NQH0R-D5ZP4 | 0.95 / 0.97 | (+5.5, +2.5) | 173 | 126.4 |
| w485200676 One Charles | BOS-FPK0Y-3J52O-EX83X | 0.92 / 0.93 | (+0.5, +3.0) | 227 | 185.2, 184.7, 184.0 |
| w29943256 Motormart Garage | BOS-A1AOD-4UKPC-6F96U | 0.98 / 0.98 | (+0.5, 0.0) | 123 | 89.2 |
| w29945923 First Lutheran Church | BOS-KTYPQ-MAYS7-YM9HL | 0.89 / 0.96 | (-1.0, +4.0) | 44 | 31.8 |
| w29719069 9 Arlington Street | BOS-GY22Z-AWKL8-BYAVI | 0.54 / 0.56 | (+1.0, +2.5) | 92 | 69.8, 69.8, 68.4 |

(`Height_Ft` is the catalog's own figure; see section 6 for why it runs too high.) 9 Arlington Street is the weak case: the model silhouette (7,907 ft²) is far larger than the OSM way (4,921 ft²).

* Matching quality over 913 models with a footprint above 300 ft² (after the shift): 528 (58%) sit at least 50% inside one OSM polygon (1:1 IoU median 0.80, 48% at 0.8 or more, 23% at 0.9 or more); 254 (28%) span several OSM polygons (OSM splits where the city model merges, e.g. row houses or building parts); 131 (14%) have no OSM building under them. 16 OSM polygons receive 2 to 5 models. Of 820 models under 300 ft² (stoops, bays, sheds), 465 have no OSM building at all. A one-to-one match by id is impossible and a 1:1 geometry match would lose a third of the buildings.
* Reverse direction: not measured cleanly. Both tiles include Cambridge, where the city has no models.

## 6. Heights: use the geometry, not the attributes

* `Height_Ft` is Z_Max_Ft minus `Gnd_El_Ft`. In our bbox the lowest vertex sits more than 3 ft above `Gnd_El_Ft` in 57% of existing buildings, more than 10 ft in 37%, more than 30 ft in 17%. Against the roof-break layer's ground (`GRND_ELEV_2010`, same frame as the OBJ z), `Gnd_El_Ft` is lower by a median 8.7 ft (p10 to p90: -31.8 to -5.9; n = 612), so `Height_Ft` runs a median 22 ft above the layer-9 height of the same building (ratio median 1.38; 15 ft for layer-9 heights under 40 ft, 31 ft above 150 ft). A LOD 2.5 model at 160 Roxbury St says 165 ft; its own z range is 11 ft.
* The geometry is consistent with layer 9: roof levels in the OBJ against `ROOF_ELEV_2010`: Motormart 101 and 103.5 ft against 100.1; Park Square 134.5 and 139 against 136.4; First Lutheran 44 to 45.5 against 43.2; One Charles 199.5 against 196.5; Saunders podium 64 against 61.1. Over 616 matched buildings the model's highest point is a median 10.2 ft above layer 9's highest roof (p10 to p90: 0.9 to 25.9) because the model includes mechanical boxes, spires and ridges. Layer 9 itself matches official heights (685 ft for the Millennium Tower, 788 ft for the Hancock Tower, per `cityheights.py`).
* `Z_MIn` penetrates the ground by a median 10.7 ft (p10 to p90: -58 to -1 ft) against layer 9's ground, so the lowest vertex is no ground reference either.

Recommended vertical rule: height above ground = (model z - layer-9 ground elevation of the overlapping parts) x 0.3048, using the same area-weighted median the pipeline already uses for layer 9; where no layer-9 part overlaps, anchor the model's dominant roof level (area-weighted median of the up-facing triangles) to the building's existing `height` and take only relative z from the model. Not established: whether layer 9's `BLDG_HGT_2010` is to the eave, the ridge or the middle of a pitched roof (it matters by 1 to 3 m).

## 7. Recommendation for the pipeline

New module (name suggestion `city3d.py`), switched on by `extra_sources = ["city3d"]`, like `cityheights.py` and `brookline.py`.

**(a) Find and download the tiles.** Read the web map item data, take the newest `Bos3D_TileGrid_PopUps_<yyyymmdd>` layer, fetch that item's data and read the OBJ link template from `popupInfo` (the template carries the release date; do not hard-code 20260624, because the previous service has already vanished). Query the layer with the envelope above (22 tiles). Download each `<tile_id>_BldgModels_OBJ.zip` to `data/raw/<area>/city3d/<release>/`, tolerate 404 (the index lists BOS_A_9, which has no zip; it is outside our area). Optional saving of about 36 MB (172 instead of 208 MB): read the central directory and `catalog.csv` of each zip by HTTP Range (about 0.3 to 0.5 MB per tile, which I did for all 102 tiles) and skip tiles with no building inside the bbox; the benefit is small. The fetch step should then write one compact cache (footprint polygon and upper-shell triangles in world metres, `Model_ID`, cred, status, parcel) so builds never reopen 34,000 nested zips. Credit string: "City of Boston Planning Department 3D model"; the models are credited per building to CyberCity 3D, pbcGIS and the Planning Department in `Model_Cred`.

**(b) Read the OBJs.** Plain text, one object, triangles only, `v x y z` in feet (see section 3), `python zipfile` reading each nested zip from memory. Keep: `StructType == "Building"`; status Current, Approved Demo and Complete; drop History, Board Approved and Under Construction (the first is demolished, the others proposals); drop footprints under about 28 m² (300 ft²). Convert with the linear rule in section 5 and the +0.1 m E, +0.7 m N correction. The plan footprint can come from `catalog_jsonp.js` instead of the OBJ.

**(c) Match each model to an OSM building.** Geometry only, in the style of `brookline.py` and `cityheights.py`: a model belongs to the OSM way, relation or part that holds at least 50% of the model's footprint area; a model that spans several OSM buildings is split by giving each roof triangle to the OSM polygon that contains its centroid; models with no OSM polygon under at least half of them are skipped (the building stays OSM plus layer 9). Expect about 58% direct, 28% split, 14% skipped. Do not try to match ids: the model, OSM and layer 9 share none.

**(d) Geometry for `building_roofs`.**
* Anchor heights as in section 6.
* Pitched roofs (27 to 31%) and stepped roofs (31 to 35%): keep the pipeline's own facade walls (`add_walls`, shader windows) from the OSM outline up to the model's lowest roof edge; add the model's upper shell above that line to `building_roofs` with vertex colours: up-facing triangles in the roof colour, vertical ones (gable ends, dormer cheeks, set-back and penthouse walls) in the wall colour with zero UVs, as `roofs.py` already does for gable-end fills. Roof-only is not enough: 64% of the model's vertical triangles lie entirely above its lowest roof edge, and without them every step and dormer would be a see-through gap. Measured per building (existing, two tiles): stepped 242 triangles (roof plus above-eave faces), pitched 223, single flat 6.8.
* Single flat roofs (41%; includes every pbcGIS extrusion): leave as they are; the model adds nothing.
* Extruded blocks and anything outside the city model: keep what `cityheights.py` does, and add steps from the roof-break parts where they matter. In the OSM ways I checked (13,102 with parts of at least 200 ft²), 14% have two or more parts of at least 5% of the area, 5% have a part at least 3 m off the dominant height (624 buildings) and 2% at least 9 m (294). Only the pbcGIS extrusions (1,336) and buildings with no city model need this; the CyberCity models already have their steps, more accurately. Each part becomes a part with its own height (the `part_of` mechanism).
* Brookline: the city model does not cover it. Of 7,888 Brookline footprints (over 300 ft²) from `brookline_buildings.json`, 2,063 contain a city-model centroid; a map of 1,300 ft cells shows coverage only along the north edge next to Boston and none in Brookline proper (5,825 of the 7,888 footprints have no model centroid). Keep the Town's own heights there.

**Cost for our 495 tiles (estimates built on measured per-building numbers).**
* Upper shell for all 11,235 buildings: 3,959 stepped x 242 + 3,485 pitched x 223 + 3,791 flat x 7 = about 1.76 M triangles. Measured 33.2 bytes per triangle in today's `building_roofs` layer (positions, normals, colours, feature index, uncompressed GLB): about 58 MB.
* Today: `building_roofs` 159,497 triangles (5.3 MB), `building_walls` 386,734 (22.7 MB), whole project 7,812,311 triangles, 195.1 MB over 495 GLBs. Net addition about +1.7 M triangles and +55 MB (slightly less than the 58 MB, because the model's roofs replace the flat slabs those 11 k buildings have today), i.e. about +22% triangles and +28% bytes on the whole tile set.
* Drawing the model's own walls as well (1.46 M triangles at 58.6 B/triangle with UVs) would add about 94 MB. Not recommended.
* If tiny features are dropped (chimneys, boxes under about 1 m, faces under about 0.5 m high) I expect roughly half of the shell, so +0.8 M triangles and about +27 MB. That 50% is a guess, not measured.
* Time: parse 0.5 s per 750 models, so about 25 s for 34,000 models; matching with an STRtree is seconds. Download 208 MB at about 6 MB/s is under a minute.
* The far-view copies (`lod.py`) are separate; I did not estimate them.

## 8. What I could not establish, and what went beyond the brief

* Absolute vertical datum: no independent survey; only relative consistency with layer 9 and with official heights of two towers.
* Whether layer 9 heights are eave, ridge or mean for pitched roofs.
* Geometry was parsed only for two tiles (1,774 and 1,541 models). Counts per `Model_LOD`, maker and status are exact for the bbox (from the catalogs), but the pitched/stepped/flat split for the bbox is extrapolated from those two central tiles.
* OSM matching used the project's cached `osm.json`; the 254 models that span several OSM polygons were not inspected one by one.
* Not downloaded or read: terrain mesh, SketchUp, DXF, ground plan, orthophoto. (The terrain mesh, `Bos3d_Terrain_2011_OBJ/<tile>_TerrainMesh_2011_OBJ.zip`, would also give real ground; not studied.)
* I did not render the ArcGIS Experience in the browser pane; I read its configuration and the web map through the REST item endpoints, which is where the layers and links came from. The PDF documentation was read with `pdftotext`.
* Licence: I read only the download page's PDDL sentence (via a page fetch), as the project owner said to rely on it. I did not check whether the CyberCity 3D credit implies other rights.
* Beyond "download two zips": besides the two tile zips (22.6 MB) I fetched, by HTTP Range requests, the `catalog.csv` of all 102 reachable tile zips (about 39 MB of ranges in total) to get exact citywide and bbox counts, plus three single-model zips (a few KB each) from BOS_G_6 to see LOD 2.5. I also downloaded the documentation PDF (16.0 MB) and `Bos3d_TileGrid.zip` (3.8 MB, a SketchUp tile frame, not needed). Total traffic about 82 MB, under the 300 MB cap.

## 9. Where things are

* Two downloaded zips: `<scratch>\city3d\tiles\` (BOS_H_4 and BOS_H_5); extracted in `...\city3d\extract\`.
* Documentation, grid layer, web map JSON: `...\city3d\docs\`. Catalogs of all tiles: `...\city3d\census\` and `...\city3d\census_all\`.
* Per-model parse results: `...\city3d\out\BOS_H_4.jsonl`, `BOS_H_5.jsonl`, `classified.json`, and the two renders.
* Scripts (all read untrusted data with `python -I`): `...\city3d\scripts\` (`analyze_tile.py`, `classify.py`, `match_osm.py`, `manyone2.py`, `five2.py`, `remote_catalog.py`, `census_city.py`, `l9elev.py`, `render.py` and others).

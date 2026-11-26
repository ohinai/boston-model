"""The Town of Brookline's public ArcGIS layers: a tree inventory and building footprints with heights.

Boston's own tree inventory stops at the Brookline line, and across the `boston` area two buildings in three have a
height that is only a guess from the building type. The Town publishes both layers openly on ArcGIS Online
(https://services1.arcgis.com/Oknk0tvfHOElpgGU/), with no key. They state no licence; the project owner decided on
26 November 2026 to use them anyway, with credit to the Town of Brookline (see docs/DATA_SOURCES.md). The use is
switched on by `extra_sources = ["brookline"]` in the area's table in areas.toml and off by deleting that line.

Trees become source elements with ids `brk:<ID>`, shaped like OpenStreetMap nodes (as inventory.py does for Boston's
`bprd:<id>`), and are merged with OpenStreetMap trees by `inventory.reconcile`. Footprint heights are written into the
`height` tag of OpenStreetMap buildings that have none (`apply_heights`), with `height:source` saying where the number
came from, so that features.py can report it honestly.

What the layers hold, as checked on 26 November 2026 (the evidence is in the report this module was written from):

* Trees: DBH is in inches (trees planted in 2007 to 2009 read 1.5 to 2, a planting caliper; the species medians match
  Boston's inventory, whose DBH is in inches). For trees planted after the 1994 baseline the figure is the caliper
  at planting and was never updated, so it is aged here (`PLANTED_GROWTH`). 122 records have no ID.
* Buildings: BLDGHEIGHT is in feet and is the Town's storey count times 13 ft (2.55 storeys is 33.15). It is therefore
  an assessed figure, not a survey, but it agrees with the storey counts mapped in OpenStreetMap and, within a quarter, with
  the Town's own LiDAR top (TOP_GL) for four buildings in five. 526 footprints are marked "DEMOLISHED 2021", and an
  outbuilding's figure is copied from the house it serves.
"""

from __future__ import annotations

import datetime
import http.client
import json
import math
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import STRtree, unary_union
from shapely.geometry import Polygon

from . import inventory, world
from .features import (
    DEFAULT_HEIGHTS, HEIGHT_FROM_TOWN, _coords, _is_underground, _relation_polygons, _way_polygons, polygons_of,
)

SERVICE = "https://services1.arcgis.com/Oknk0tvfHOElpgGU/arcgis/rest/services"
TREES_ENDPOINT = f"{SERVICE}/Brookline_Tree_Viewer_Web_WFL1/FeatureServer/0/query"
BUILDINGS_ENDPOINT = f"{SERVICE}/BuildingFootprint/FeatureServer/0/query"
CREDIT = "Town of Brookline"  # tiles.BROOKLINE_ATTRIBUTION says what for
PAGE_SIZE = 2000  # the server's limit on records per reply
ATTEMPTS = 3  # tries per page before giving up
RETRY_PAUSE = 5  # seconds between tries
USER_AGENT = "city-model-pipeline/0.1"
TREES_FILE = "brookline_trees.json"
BUILDINGS_FILE = "brookline_buildings.json"
TREE_SOURCE_TAG = "Town of Brookline tree inventory"
TREE_PREFIX = "brk:"

# ----------------------------------------------------------------------------------------------------- trees

INCH = 0.0254  # metres
MAX_DBH = 100.0  # inches. Above this the record is a typing error (the box holds 455 and 1721); trees.describe only trusts a girth up to 8 m, which is about 100 in
NO_ID_BASE = 1_000_000  # the 122 records with no ID take 1,000,000 + their OBJECTID, so every id is `brk:<digits>` as fixes/ expects
PLANTED_GROWTH = 0.3  # inches of trunk diameter a street tree adds each year: the low end of the 0.3 to 0.6 that urban-forestry studies give, so that a planted tree is never made too big
PLANTING_CALIPER = 2.5  # inches: a nursery street tree, for planted trees that have no DBH at all
BASELINE_YEAR = 1994  # YR_PLANT of every tree that stood when the Town started its inventory: its DBH is not a planting caliper
THIS_YEAR = 2026
AGED_BELOW = 10.0  # a tree planted after the baseline whose DBH is below this is still the planting caliper; a larger figure was measured later
NOT_TREES = ("stump", "empty", "vacant", "planting site")
DEAD = 4  # CONDITION code of a dead tree (the layer's domain: 0 Excellent, 1 Good, 2 Fair, 3 Poor, 4 Dead)
TYPOS = {"fraxinus pennsylvanicum": "Fraxinus pennsylvanica", "carpinus caroliana": "Carpinus caroliniana",
         "crataegus phaenopyru": "Crataegus phaenopyrum"}  # misspelt in the layer
# What the Town's free-text CG_NOTES says on records whose species is "Unknown": (botanical or genus, common name).
# Only exact notes, so that nothing is guessed from a sentence.
NOTE_SPECIES = {
    "european beech": ("Fagus sylvatica", "European beech"), "pine": ("Pinus", "Pine"),
    "white pine": ("Pinus strobus", "Eastern white pine"), "black oak": ("Quercus velutina", "Black oak"),
    "southern red oak": ("Quercus falcata", "Southern red oak"), "english oak": ("Quercus robur", "English oak"),
    "walnut": ("Juglans", "Walnut"), "butter nut": ("Juglans cinerea", "Butternut"), "magnolia": ("Magnolia", "Magnolia"),
    "white mulberry": ("Morus alba", "White mulberry"), "black mulberry": ("Morus nigra", "Black mulberry"),
    "mulberry": ("Morus", "Mulberry"), "princeton elm": ("Ulmus americana 'Princeton'", "American elm 'Princeton'"),
    "juniper": ("Juniperus", "Juniper"), "spruce": ("Picea", "Spruce"), "norway spruce": ("Picea abies", "Norway spruce"),
    "hemlock": ("Tsuga", "Hemlock"), "tupelo": ("Nyssa sylvatica", "Tupelo"), "poplar": ("Populus", "Poplar"),
    "sassafras": ("Sassafras albidum", "Sassafras"), "silver fir": ("Abies alba", "Silver fir"),
    "weeping cherry": ("Prunus", "Weeping cherry"), "purple leaf plum": ("Prunus cerasifera", "Purple leaf plum"),
    "carolina silver bell": ("Halesia tetraptera", "Carolina silverbell"),
}


def _text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _botanical(props: dict, names: dict[str, tuple[str, str]]) -> tuple[str | None, str | None, str | None]:
    """(species or None, genus or None, common name or None) for one record.

    The Town's ScientificName sometimes carries the cultivar (`Pyrus calleryana Aristocrat`, with Cultivar
    `Aristocrat`) and sometimes only a genus (`Malus`). The 122 newest records have a common name and no
    ScientificName: `names` (built from the other records) says which species a common name is.
    """
    scientific, common, cultivar = _text(props.get("ScientificName")), _text(props.get("CommonName")), _text(props.get("Cultivar"))
    if common.lower() == "unknown":
        common = ""
    if not scientific and common:
        found = names.get(common.lower())
        if found is None:  # "Honeylocust Skyline": the species name with a cultivar after it
            words = common.split(" ")
            for cut in range(len(words) - 1, 0, -1):
                found = names.get(" ".join(words[:cut]).lower())
                if found:
                    cultivar = cultivar or " ".join(words[cut:])
                    common = " ".join(words[:cut])
                    break
        if found:
            scientific = found[0]
    if cultivar and scientific.lower().endswith(cultivar.lower()):
        scientific = scientific[: -len(cultivar)].strip()  # the cultivar is added back in quotes
    scientific = TYPOS.get(scientific.lower(), scientific)
    if not scientific:
        return None, None, inventory.common_name(common)
    words = scientific.split(" ")
    genus = words[0].capitalize()
    species = scientific if len(words) > 1 else None  # "Malus" names only a genus
    if species and cultivar and "'" not in species:
        species = f"{species} '{cultivar}'"
    name = inventory.common_name(re.sub(re.escape(cultivar) + r"$", "", common).strip() if cultivar else common)
    if name and cultivar:
        name = f"{name} '{cultivar}'"
    return species, genus, name


def _diameter(props: dict) -> float | None:
    """Trunk diameter at breast height now, in inches; None when the record gives no usable figure."""
    try:
        dbh = float(props.get("DBH") or 0)
    except (TypeError, ValueError):
        dbh = 0.0
    year = props.get("YR_PLANT")
    age = THIS_YEAR - year if isinstance(year, int) and BASELINE_YEAR < year <= THIS_YEAR else None
    if dbh > MAX_DBH:
        return None
    if dbh <= 0:
        return PLANTING_CALIPER + PLANTED_GROWTH * age if age is not None else None
    if age is not None and dbh < AGED_BELOW:
        return dbh + PLANTED_GROWTH * age  # the planting caliper, never remeasured
    return dbh


def tree_elements(payload: dict) -> list[dict]:
    """Source elements (OpenStreetMap-style nodes, ids `brk:<n>`) for the trees in a downloaded payload."""
    records = [f for f in payload.get("features", []) if (f.get("geometry") or {}).get("coordinates")]
    names: dict[str, tuple[str, str]] = {}
    for feature in records:
        p = feature.get("properties") or {}
        if _text(p.get("ScientificName")) and _text(p.get("CommonName")) and not _text(p.get("Cultivar")):
            names.setdefault(_text(p["CommonName"]).lower(), (_text(p["ScientificName"]), _text(p["CommonName"])))
    out, seen = [], set()
    for feature in records:
        props = feature.get("properties") or {}
        coordinates = feature["geometry"]["coordinates"]
        common = _text(props.get("CommonName")).lower()
        if props.get("IsStump") == 1 or props.get("CONDITION") == DEAD or any(word in common for word in NOT_TREES):
            continue
        species, genus, name = _botanical(props, names)
        if genus is None:
            note = NOTE_SPECIES.get(_text(props.get("CG_NOTES")).lower())
            if note:
                species, name = (note[0], note[1]) if " " in note[0] else (None, note[1])
                genus = note[0].split(" ")[0]
        dbh = _diameter(props)
        if genus is None and not name and dbh is None:
            continue  # a record with no species and no size: nothing says a tree stands there
        if props.get("ID") not in (None, ""):
            number = int(re.sub(r"\D", "", str(props["ID"])) or 0)
        else:
            number = NO_ID_BASE + int(props.get("OBJECTID") or 0)
        ident = f"{TREE_PREFIX}{number}"
        if not number or ident in seen:
            continue
        seen.add(ident)
        tags = {"natural": "tree"}
        if species:
            tags["species"] = species
        if genus:
            tags["genus"] = genus
        if name:
            tags["species:en"] = name
        if dbh:
            tags["circumference"] = f"{math.pi * dbh * INCH:.2f}"  # OSM: trunk girth in metres at 1.3 m
        else:
            tags["denotation"] = "park" if props.get("PARKTREE") == 1 else "street"  # the size to assume when there is none (trees.py)
        tags["source"] = TREE_SOURCE_TAG
        out.append({"type": "node", "ident": ident, "lon": float(coordinates[0]), "lat": float(coordinates[1]), "tags": tags})
    return out


# ----------------------------------------------------------------------------------------------------- buildings

FOOT = 0.3048  # metres
MIN_FEET, MAX_FEET = 8.0, 250.0  # a figure outside this is not a building height (the box holds 13 ft to 154 ft)
STOREY_FEET = 13.0  # the Town's own conversion, used where its data lacks a typical figure for a storey count
MAX_STOREYS = 25
TALL_FEET = 60.0  # a claim above this must be corroborated by the LiDAR top
CORROBORATION = 0.55  # the LiDAR top must be at least this share of a tall claim ...
LIDAR_MIN_FEET = 8.0  # ... unless it is below this, when the LiDAR is plainly broken (an 8-storey tower "5.4 ft" tall)
OUT_LIDAR = (7.0, 20.0)  # feet: an outbuilding's LiDAR top is believed between these
MIN_OVERLAP = 0.5  # a footprint matches an OpenStreetMap building when it overlaps this share of the smaller of the two
MIN_COVER = 0.3  # and the matching footprints together cover at least this share of the OpenStreetMap building
SKIP_FOOTPRINT_TYPES = ("OVHD-WALKWAY", "FOUNDATION")  # TYPE values that are not a building to draw
SMALL_KINDS = 5.0  # an OpenStreetMap building type whose own default is below this (garage, shed, kiosk) keeps it


@dataclass
class Footprint:
    polygon: Polygon  # world metres
    feet: float  # height above ground, feet
    outbuilding: bool  # a garage or shed (the Town's OUTBLDG); its height is the LiDAR top, never the house's storey figure


def _sane(feet) -> float | None:
    try:
        feet = float(feet)
    except (TypeError, ValueError):
        return None
    return feet if MIN_FEET <= feet <= MAX_FEET else None


def footprints(payload: dict) -> tuple[list[Footprint], dict]:
    """Footprints worth matching, with a height in feet each, and counts of what was set aside."""
    rows = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        polygons = [geometry["coordinates"]] if geometry.get("type") == "Polygon" else geometry.get("coordinates", []) if geometry.get("type") == "MultiPolygon" else []
        shapes = []
        for rings in polygons:
            projected = []
            for ring in rings:
                x, y = world.to_local(np.array([c[0] for c in ring], float), np.array([c[1] for c in ring], float))
                projected.append(list(zip(x.tolist(), y.tolist())))
            if projected and len(projected[0]) >= 4:
                shapes.extend(polygons_of(Polygon(projected[0], projected[1:])))
        if shapes:
            rows.append((feature.get("properties") or {}, shapes))
    # How tall a storey count is in the Town's own figures: the median BLDGHEIGHT of each count.
    by_storeys = defaultdict(list)
    for props, _ in rows:
        stories, feet = props.get("NUMSTORIES"), _sane(props.get("BLDGHEIGHT"))
        if props.get("TYPE") != "OUTBLDG" and feet and isinstance(stories, int) and 1 <= stories <= MAX_STOREYS:
            by_storeys[stories].append(feet)
    typical = {n: statistics.median(v) for n, v in by_storeys.items() if len(v) >= 5}
    out_tops = [props.get("TOP_GL") for props, _ in rows if props.get("TYPE") == "OUTBLDG"]
    out_tops = [t for t in out_tops if isinstance(t, (int, float)) and OUT_LIDAR[0] <= t <= OUT_LIDAR[1]]
    typical_outbuilding = statistics.median(out_tops) if out_tops else None

    kept, skipped = [], defaultdict(int)
    for props, shapes in rows:
        kind = props.get("TYPE")
        remark = f"{props.get('COMMENT') or ''} {props.get('COMMENTS') or ''}".lower()
        if "demolish" in remark:
            skipped["demolished"] += 1
            continue
        if kind in SKIP_FOOTPRINT_TYPES:
            skipped["not a building"] += 1
            continue
        lidar = props.get("TOP_GL")
        if kind == "OUTBLDG":
            feet = lidar if isinstance(lidar, (int, float)) and OUT_LIDAR[0] <= lidar <= OUT_LIDAR[1] else typical_outbuilding
        else:
            feet = _sane(props.get("BLDGHEIGHT"))
            stories = props.get("NUMSTORIES")
            if feet is None and isinstance(stories, int) and 1 <= stories <= MAX_STOREYS:
                feet = typical.get(stories, STOREY_FEET * stories)  # no height, but the Town counted the storeys
            if feet and feet > TALL_FEET and isinstance(lidar, (int, float)) and lidar >= LIDAR_MIN_FEET and lidar < CORROBORATION * feet:
                skipped["tall, not confirmed by the LiDAR top"] += 1
                continue
        if not feet:
            skipped["no height"] += 1
            continue
        kept.extend(Footprint(shape, float(feet), kind == "OUTBLDG") for shape in shapes)
    return kept, dict(skipped)


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """The value (first of each pair) below which half of the weight (second) lies."""
    ordered = sorted(pairs)
    half = sum(w for _, w in ordered) / 2
    total = 0.0
    for value, weight in ordered:
        total += weight
        if total >= half:
            return value
    return ordered[-1][0]


def _building_shape(element: dict):
    """The polygons of a building element as one shapely geometry, or None."""
    if element["type"] == "way":
        polygons = _way_polygons(_coords(element["geometry"])) if element.get("geometry") else []
    elif element["type"] == "relation" and (element.get("tags") or {}).get("type") == "multipolygon":
        polygons = _relation_polygons(element)
    else:
        return None
    return unary_union(polygons) if polygons else None


def wants_height(tags: dict) -> bool:
    """Whether a building element is one the Town's figures may give a height to: it has none of its own and is not a
    small structure whose type already says how tall it is, a roof alone, or a part that starts above the ground."""
    if "building" not in tags and "building:part" not in tags:
        return False
    kind = tags.get("building") or tags.get("building:part")
    if kind in ("no", "roof") or tags.get("amenity") == "shelter" or _is_underground(tags):
        return False
    if any(tags.get(k) for k in ("height", "building:height", "building:levels", "min_height", "building:min_level")):
        return False  # OpenStreetMap's own figure wins
    return DEFAULT_HEIGHTS.get(kind, SMALL_KINDS) >= SMALL_KINDS


def apply_heights(elements: list[dict], feet: list[Footprint]) -> tuple[list[dict], dict]:
    """Elements with a `height` (metres) and `height:source` tag added to the buildings the Town's footprints cover.

    A footprint matches a building when it overlaps at least half of the smaller of the two; the matching footprints
    together must cover 30% of the building. Several footprints on one building give the figure that covers most of its
    area (the area-weighted median: a house with a small tall annex stays a house). A building's own outbuilding
    footprints count only when it has no other. Buildings that already have `height`, `building:levels` or a `min_height`
    are left as they are. Returns the new list (the input is not changed) and counts.
    """
    wanted = []
    for index, element in enumerate(elements):
        if element["type"] in ("way", "relation") and wants_height(element.get("tags") or {}):
            shape = _building_shape(element)
            if shape is not None and shape.area >= 2.0:
                wanted.append((index, shape))
    stats = {"footprints": len(feet), "buildings": len(wanted), "given": 0, "from_outbuilding": 0, "uncovered": 0}
    if not wanted or not feet:
        return elements, stats
    tree = STRtree([shape for _, shape in wanted])
    votes: dict[int, list[tuple[Footprint, float]]] = defaultdict(list)
    for footprint in feet:
        for j in tree.query(footprint.polygon, predicate="intersects"):
            shape = wanted[int(j)][1]
            overlap = footprint.polygon.intersection(shape).area
            if overlap >= MIN_OVERLAP * min(footprint.polygon.area, shape.area):
                votes[int(j)].append((footprint, overlap))
    out = list(elements)
    for j, ballots in votes.items():
        index, shape = wanted[j]
        main = [b for b in ballots if not b[0].outbuilding]
        chosen = main or ballots
        if sum(overlap for _, overlap in chosen) < MIN_COVER * shape.area:
            stats["uncovered"] += 1
            continue
        feet_tall = _weighted_median([(f.feet, overlap) for f, overlap in chosen])
        element = elements[index]
        out[index] = {**element, "tags": {**element["tags"], "height": f"{feet_tall * FOOT:.1f}", "height:source": HEIGHT_FROM_TOWN}}
        stats["given"] += 1
        stats["from_outbuilding"] += 0 if main else 1
    return out, stats


# ----------------------------------------------------------------------------------------------------- download


def fetch(bbox_wsen: tuple[float, float, float, float], folder: Path, force: bool = False) -> list[Path]:
    """Download both layers for the box into `folder` (JSON). Reuses files already there unless forced."""
    out = []
    for name, endpoint in ((TREES_FILE, TREES_ENDPOINT), (BUILDINGS_FILE, BUILDINGS_ENDPOINT)):
        out.append(_fetch_layer(bbox_wsen, folder / name, endpoint, force))
    return out


def _fetch_layer(bbox_wsen, dest: Path, endpoint: str, force: bool) -> Path:
    if dest.exists() and not force:
        print(f"using cached {dest} (pass --force to refresh)")
        return dest
    features: list[dict] = []
    while True:
        page, more = _page(endpoint, bbox_wsen, len(features))
        features.extend(page)
        if not page or not more:
            break
    payload = {
        "source": endpoint,
        "credit": CREDIT,
        "licence": "none stated; used on the project owner's decision of 26 November 2026 (docs/DATA_SOURCES.md)",
        "fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "bbox_wsen": list(bbox_wsen),
        "features": features,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest} ({len(features)} records, {dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _page(endpoint: str, bbox_wsen, offset: int) -> tuple[list[dict], bool]:
    """One page of records starting at `offset`, and whether the server has more."""
    query = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": ",".join(str(v) for v in bbox_wsen),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "orderByFields": "OBJECTID",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
    })
    request = urllib.request.Request(f"{endpoint}?{query}", headers={"User-Agent": USER_AGENT})
    layer = endpoint.split("/")[-4]
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(RETRY_PAUSE)
        try:
            print(f"querying {layer} from record {offset} ...")
            with urllib.request.urlopen(request, timeout=120) as response:
                reply = json.loads(response.read())
            if "features" not in reply:
                raise ValueError(f"unexpected reply: {str(reply.get('error', reply))[:200]}")
        except (OSError, http.client.HTTPException, ValueError) as error:
            print(f"  failed: {error}")
            last_error = error
            continue
        features = reply["features"]
        exceeded = reply.get("exceededTransferLimit") or (reply.get("properties") or {}).get("exceededTransferLimit")
        return features, bool(exceeded) or len(features) >= PAGE_SIZE
    raise RuntimeError(f"the Brookline layer {layer} failed {ATTEMPTS} times; the last error was: {last_error}")

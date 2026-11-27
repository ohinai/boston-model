"""The City of Boston's building heights: the Planning Department's "buildings with roof breaks" layer.

Across the `boston` area half of the buildings have a height that is only a guess from the building type or a storey
count. The Planning Department publishes the city's buildings split at roof-height changes, with each part's height above the
ground measured from the 2010 LiDAR survey (its 3D Smart Model is built from the same). Its download page states the Open Data
Commons Public Domain Dedication and License (PDDL), so there is no condition on use and the credit is a courtesy
(https://www.bostonplans.org/3d-data-maps/3d-smart-model/3d-data-download; the feature service is
https://gis.bostonplans.org/hosting/rest/services/Boston_Buildings/FeatureServer/9). It is switched on by
`extra_sources = ["city_heights"]` in the area's table in areas.toml, and off by deleting that word.

What the layer holds, as checked on 27 November 2026:

* `BLDG_HGT_2010` is in FEET above the ground (`ROOF_ELEV_2010` minus `GRND_ELEV_2010`). Against the 451 OpenStreetMap buildings
  of the area that carry a metric `height`, the ratio is 3.29 (median; 74% within 15% of 3.2808), and the tallest part is 788 ft,
  the John Hancock Tower. A level is 11.8 ft in the median (3.6 m), which agrees with the 3.4 m plus a parapet the pipeline uses.
* A building is several parts where its roof steps (a tower on a podium, a penthouse). Each part is a polygon with its own height.
* It is a 2010 survey: a building put up since is missing, and one pulled down since is still there. OpenStreetMap buildings
  with a `start_date` after 2010 are therefore left alone, and a building whose own `building:levels` disagrees with the survey
  by more than a third keeps its levels (it has probably been rebuilt; the mapper counted the floors).

Heights go into the `height` tag of OpenStreetMap buildings that have no `height` of their own (`apply_heights`), with
`height:source` saying where the number came from, so features.py reports it honestly. The matching is the one brookline.py uses
for the Town's footprints: a part belongs to a building when it overlaps at least half of the smaller of the two; the parts
together must cover 30% of the building; the figure that covers most of its area (the area-weighted median) is the building's, so
a rooftop bulkhead does not make a house a tower.
"""

from __future__ import annotations

import datetime
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon

from . import brookline, world
from .features import (
    DEFAULT_HEIGHTS, HEIGHT_FROM_CITY, LEVEL_HEIGHT, ROOF_ALLOWANCE, _is_underground, parse_number, polygons_of,
)

ENDPOINT = "https://gis.bostonplans.org/hosting/rest/services/Boston_Buildings/FeatureServer/9/query"
FILE = "city_heights.json"
CREDIT = "City of Boston Planning Department"  # tiles.CITY_ATTRIBUTION says what for
FOOT = 0.3048  # metres
MIN_FEET, MAX_FEET = 8.0, 900.0  # a part outside this is no building height (the box holds 8 ft to 788 ft; 975 parts say 0)
SURVEY_YEAR = 2010  # a building mapped as built after this is not in the survey
LEVELS_AGREE = (0.7, 1.4)  # the survey's height over the height the building's own levels give: keep the levels outside this
MAIN_TYPES = ("BLDG", "bldg")  # IEL_TYPE of a building part; OUTBLDG (a garage or shed) counts only where nothing else does
OUT_TYPES = ("OUTBLDG",)
SMALL_KINDS = brookline.SMALL_KINDS


def footprints(payload: dict) -> tuple[list[brookline.Footprint], dict]:
    """The parts worth matching, each with a height in feet, and counts of what was set aside."""
    kept, skipped = [], defaultdict(int)
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        kind = props.get("IEL_TYPE")
        if kind not in MAIN_TYPES + OUT_TYPES:
            skipped["not a building"] += 1
            continue
        try:
            feet = float(props.get("BLDG_HGT_2010"))
        except (TypeError, ValueError):
            skipped["no height"] += 1
            continue
        if not MIN_FEET <= feet <= MAX_FEET:
            skipped["no usable height"] += 1
            continue
        geometry = feature.get("geometry") or {}
        polygons = [geometry["coordinates"]] if geometry.get("type") == "Polygon" else geometry.get("coordinates", []) if geometry.get("type") == "MultiPolygon" else []
        for rings in polygons:
            projected = []
            for ring in rings:
                x, y = world.to_local(np.array([c[0] for c in ring], float), np.array([c[1] for c in ring], float))
                projected.append(list(zip(x.tolist(), y.tolist())))
            if projected and len(projected[0]) >= 4:
                for polygon in polygons_of(Polygon(projected[0], projected[1:])):
                    kept.append(brookline.Footprint(polygon, feet, kind in OUT_TYPES))
    return kept, dict(skipped)


def _year(tags: dict) -> int | None:
    found = re.match(r"\s*(\d{4})", str(tags.get("start_date") or tags.get("building:start_date") or ""))
    return int(found.group(1)) if found else None


def wants_height(tags: dict) -> bool:
    """Whether the survey may give this building a height: it states none of its own (a count of levels is allowed: apply_heights
    checks it against the survey), it is not a small structure whose type already says how tall it is, a roof alone, a part
    that starts above the ground, or newer than the survey."""
    if "building" not in tags and "building:part" not in tags:
        return False
    kind = tags.get("building") or tags.get("building:part")
    if kind in ("no", "roof") or tags.get("amenity") == "shelter" or _is_underground(tags):
        return False
    if any(tags.get(k) for k in ("height", "building:height", "min_height", "building:min_level")):
        return False  # OpenStreetMap's own figure wins
    if (year := _year(tags)) is not None and year > SURVEY_YEAR:
        return False
    return DEFAULT_HEIGHTS.get(kind, SMALL_KINDS) >= SMALL_KINDS


def apply_heights(elements: list[dict], parts: list[brookline.Footprint]) -> tuple[list[dict], dict]:
    """Elements with a `height` (metres) and `height:source` tag added to the buildings the survey's parts cover; the input is
    not changed. Returns the new list and counts (`given`, of which `over_levels` had only a count of levels; `kept_levels` where
    the count and the survey disagree; `uncovered` where the matching parts cover too little of the building)."""
    wanted = []
    for index, element in enumerate(elements):
        if element["type"] in ("way", "relation") and wants_height(element.get("tags") or {}):
            shape = brookline._building_shape(element)
            if shape is not None and shape.area >= 2.0:
                wanted.append((index, shape))
    stats = {"parts": len(parts), "buildings": len(wanted), "given": 0, "over_levels": 0, "kept_levels": 0, "uncovered": 0}
    if not wanted or not parts:
        return elements, stats
    tree = STRtree([shape for _, shape in wanted])
    votes: dict[int, list[tuple[brookline.Footprint, float]]] = defaultdict(list)
    for part in parts:
        for j in tree.query(part.polygon, predicate="intersects"):
            shape = wanted[int(j)][1]
            overlap = part.polygon.intersection(shape).area
            if overlap >= brookline.MIN_OVERLAP * min(part.polygon.area, shape.area):
                votes[int(j)].append((part, overlap))
    out = list(elements)
    for j, ballots in votes.items():
        index, shape = wanted[j]
        main = [b for b in ballots if not b[0].outbuilding]
        chosen = main or ballots
        if sum(overlap for _, overlap in chosen) < brookline.MIN_COVER * shape.area:
            stats["uncovered"] += 1
            continue
        metres = brookline._weighted_median([(p.feet, overlap) for p, overlap in chosen]) * FOOT
        tags = elements[index]["tags"]
        levels = parse_number(tags.get("building:levels"))
        if levels and levels > 0:
            expected = levels * LEVEL_HEIGHT + ROOF_ALLOWANCE
            if not LEVELS_AGREE[0] <= metres / expected <= LEVELS_AGREE[1]:
                stats["kept_levels"] += 1
                continue
            stats["over_levels"] += 1
        out[index] = {**elements[index], "tags": {**tags, "height": f"{metres:.1f}", "height:source": HEIGHT_FROM_CITY}}
        stats["given"] += 1
    return out, stats


# ----------------------------------------------------------------------------------------------------- download


def fetch(bbox_wsen: tuple[float, float, float, float], folder: Path, force: bool = False) -> Path:
    """Download the parts for the box into `folder`/city_heights.json. Reuses the file there unless forced."""
    dest = folder / FILE
    if dest.exists() and not force:
        print(f"using cached {dest} (pass --force to refresh)")
        return dest
    features: list[dict] = []
    while True:
        page, more = brookline._page(ENDPOINT, bbox_wsen, len(features), who="the City of Boston building layer")
        features.extend(page)
        if not page or not more:
            break
    payload = {
        "source": ENDPOINT,
        "credit": CREDIT,
        "licence": "Open Data Commons Public Domain Dedication and License (PDDL), stated on the Planning Department's 3D data download page",
        "fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "bbox_wsen": list(bbox_wsen),
        "features": features,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest} ({len(features)} records, {dest.stat().st_size / 1e6:.1f} MB)")
    return dest

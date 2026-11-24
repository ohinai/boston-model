"""The City of Boston's tree inventory: download it and merge it with OpenStreetMap trees.

The inventory (Parks and Recreation, ODC PDDL public domain dedication) gives
each tree a surveyed position, species and trunk size. Records become source
elements shaped like OpenStreetMap nodes with ids `bprd:<id>`, so fixes can
change or delete them like any other feature.
"""

from __future__ import annotations

import datetime
import http.client
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from shapely import STRtree, points
from shapely.prepared import prep

ENDPOINT = "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/BPRD_Trees/FeatureServer/0/query"
LICENCE = "ODC PDDL"
PAGE_SIZE = 2000  # the server's limit on records per reply
ATTEMPTS = 3  # tries per page before giving up
RETRY_PAUSE = 5  # seconds between tries
USER_AGENT = "city-model-pipeline/0.1"
SOURCE_TAG = "City of Boston tree inventory"

NOT_TREES = ("empty", "planting site", "stump", "vacant")  # common-name words of records that hold no tree
MATCH_RADIUS = 4.0  # metres within which an inventory and an OpenStreetMap tree are the same tree
MIN_PARK_TREES = 20  # inventory trees a park needs before the inventory is trusted over OpenStreetMap there
PARK_KINDS = ("park", "common", "garden")  # leisure values that count as a park
KEEP_FROM_OSM = ("name", "inscription", "start_date", "height", "diameter_crown", "denotation", "leaf_cycle")


def fetch(bbox_wsen: tuple[float, float, float, float], dest: Path, force: bool = False) -> Path:
    """Download every inventory record in the box into `dest` (JSON). Reuses an existing file unless forced."""
    if dest.exists() and not force:
        print(f"using cached {dest} (pass --force to refresh)")
        return dest
    features: list[dict] = []
    while True:
        page, more = _page(bbox_wsen, len(features))
        features.extend(page)
        if not page or not more:
            break
    payload = {
        "source": ENDPOINT,
        "licence": LICENCE,
        "fetched": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "bbox_wsen": list(bbox_wsen),
        "features": features,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest} ({len(features)} records, {dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _page(bbox_wsen, offset: int) -> tuple[list[dict], bool]:
    """One page of records starting at `offset`, and whether the server has more."""
    query = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": ",".join(str(v) for v in bbox_wsen),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "orderByFields": "ObjectId",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
    })
    request = urllib.request.Request(f"{ENDPOINT}?{query}", headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(RETRY_PAUSE)
        try:
            print(f"querying the tree inventory from record {offset} ...")
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
    raise RuntimeError(f"the city tree inventory failed {ATTEMPTS} times; the last error was: {last_error}")


def common_name(text: str | None) -> str | None:
    """"Linden, Littleleaf" becomes "Littleleaf linden"; names without a comma are only tidied."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"\s+(spp|sp)\.?$", "", text, flags=re.IGNORECASE)
    if not text:
        return None
    if "," in text:
        head, _, rest = text.partition(",")
        text = f"{rest.strip()} {head.strip()}"
    return text[0].upper() + text[1:].lower()


def elements(payload: dict) -> list[dict]:
    """Source elements (OpenStreetMap-style nodes) for the trees in a downloaded payload."""
    out, seen = [], set()
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        coordinates = (feature.get("geometry") or {}).get("coordinates")
        common = props.get("spp_com") or ""
        if props.get("id") is None or not coordinates or any(word in common.lower() for word in NOT_TREES):
            continue
        ident = f"bprd:{props['id']}"
        if ident in seen:
            continue
        seen.add(ident)
        tags = {"natural": "tree"}
        botanical = re.sub(r"\s+", " ", (props.get("spp_bot") or "").strip())
        if botanical and botanical.lower() != "unknown":
            # "Ulmus species" names only a genus, so it is not a species tag.
            if not re.search(r"\b(species|spp?\.?)$", botanical, re.IGNORECASE):
                tags["species"] = botanical
            tags["genus"] = botanical.split()[0].capitalize()
        if name := common_name(common):
            tags["species:en"] = name
        try:
            dbh = float(props.get("dbh") or 0)
        except ValueError:
            dbh = 0.0
        if dbh > 0:
            tags["circumference"] = f"{math.pi * dbh * 0.0254:.2f}"  # OSM: trunk girth in metres at 1.3 m
        if props.get("park"):
            tags["bprd:park"] = str(props["park"])
        tags["source"] = SOURCE_TAG
        out.append({"type": "node", "ident": ident, "lon": float(coordinates[0]), "lat": float(coordinates[1]), "tags": tags})
    return out


def reconcile(scene) -> dict:
    """Merge inventory trees and OpenStreetMap trees in `scene.trees`, in place; returns counts."""
    inventory = [t for t in scene.trees if t.id.startswith("bprd:")]
    osm = [t for t in scene.trees if not t.id.startswith("bprd:")]
    inv_pts = points([(t.x, t.y) for t in inventory])
    osm_pts = points([(t.x, t.y) for t in osm])

    # Nearest pairs first, each tree used once.
    pairs = []
    if inventory and osm:
        found = STRtree(osm_pts).query(inv_pts, predicate="dwithin", distance=MATCH_RADIUS)
        distances = np.hypot(
            np.array([inventory[i].x - osm[j].x for i, j in found.T]), np.array([inventory[i].y - osm[j].y for i, j in found.T])
        )
        pairs = sorted(zip(distances.tolist(), found[0].tolist(), found[1].tolist()))
    matched_inv, matched_osm = {}, set()
    for _, i, j in pairs:
        if i not in matched_inv and j not in matched_osm:
            matched_inv[i] = j
            matched_osm.add(j)

    merged_trees = []
    for i, tree in enumerate(inventory):
        if i in matched_inv:
            theirs = osm[matched_inv[i]]
            tags = dict(tree.tags)
            for key in KEEP_FROM_OSM:
                if key in theirs.tags and key not in tags:
                    tags[key] = theirs.tags[key]
            tags["osm"] = theirs.id
            tree = type(tree)(tree.id, tree.x, tree.y, name=tags.get("name") or tree.name, kind=tree.kind, tags=tags)
        merged_trees.append(tree)

    # In a park the inventory covers, an OpenStreetMap tree it does not list is not there.
    parks = [
        prep(area.polygon)
        for area in scene.areas.get("green", [])
        if area.tags.get("leisure") in PARK_KINDS
    ]
    inventoried = [
        park for park in parks if sum(1 for t, p in zip(inventory, inv_pts) if "bprd:park" in t.tags and park.contains(p)) >= MIN_PARK_TREES
    ]
    kept, dropped = [], 0
    for j, tree in enumerate(osm):
        if j in matched_osm:
            continue
        if any(park.contains(osm_pts[j]) for park in inventoried):
            dropped += 1
        else:
            kept.append(tree)

    scene.trees[:] = merged_trees + kept
    return {"inventory": len(inventory), "osm": len(osm), "merged": len(matched_inv), "dropped": dropped, "kept_osm": len(kept)}

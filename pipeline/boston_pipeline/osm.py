"""Download OpenStreetMap data for a bounding box through the Overpass API."""

from __future__ import annotations

import gzip
import http.client
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Public Overpass servers, tried in order. Each is run by volunteers or a
# company as a courtesy and any of them can be down or busy at a given moment.
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
ROUNDS = 3  # passes over the whole server list before giving up
ROUND_PAUSE = 30  # seconds to wait before the second pass; doubled for the third
USER_AGENT = "city-model-pipeline/0.1"
# A server whose copy of the map is older than this many days is passed over.
# Servers normally run minutes behind, but in November 2026 one of them
# answered with a copy four months old, and the site was built from it.
MAX_AGE_DAYS = 3

# Everything the builder knows how to use. `out geom` inlines coordinates so no
# second pass over nodes is needed.
QUERY = """
[out:json][timeout:240][maxsize:536870912];
(
  nwr["building"]({{bbox}});
  nwr["building:part"]({{bbox}});
  way["highway"]({{bbox}});
  way["railway"~"^(rail|light_rail|subway|tram|platform)$"]({{bbox}});
  way["natural"~"^(water|coastline|wood|scrub|grassland)$"]({{bbox}});
  relation["natural"~"^(water|wood)$"]({{bbox}});
  way["waterway"~"^(dock|riverbank)$"]({{bbox}});
  way["man_made"~"^(pier|breakwater|groyne)$"]({{bbox}});
  relation["man_made"="pier"]({{bbox}});
  way["leisure"~"^(park|garden|pitch|playground|common|dog_park|golf_course)$"]({{bbox}});
  relation["leisure"~"^(park|garden|common)$"]({{bbox}});
  way["landuse"~"^(grass|forest|meadow|cemetery|recreation_ground|village_green|flowerbed)$"]({{bbox}});
  relation["landuse"~"^(grass|forest|meadow|cemetery|recreation_ground)$"]({{bbox}});
  way["barrier"~"^(fence|wall|retaining_wall|hedge|chain|guard_rail|kerb|gate|bollard|city_wall)$"]({{bbox}});
  node["natural"="tree"]({{bbox}});
  node["barrier"~"^(bollard|block)$"]({{bbox}});
  node["highway"~"^(traffic_signals|crossing|street_lamp|bus_stop|stop)$"]({{bbox}});
  node["shop"]({{bbox}});
  node["amenity"]({{bbox}});
  node["tourism"]({{bbox}});
  node["historic"]({{bbox}});
  node["railway"~"^(station|subway_entrance|tram_stop|halt)$"]({{bbox}});
);
out geom qt;
"""


def build_query(bbox_wsen: tuple[float, float, float, float]) -> str:
    w, s, e, n = bbox_wsen
    return QUERY.replace("{{bbox}}", f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}")


def fetch(bbox_wsen: tuple[float, float, float, float], dest: Path, force: bool = False) -> Path:
    """Fetch the box into `dest` (JSON). Reuses an existing file unless forced."""
    if dest.exists() and not force:
        print(f"using cached {dest} (pass --force to refresh)")
        return dest
    body = urllib.parse.urlencode({"data": build_query(bbox_wsen)}).encode()
    last_error: Exception | None = None
    for attempt in range(ROUNDS):
        if attempt:
            pause = ROUND_PAUSE * 2 ** (attempt - 1)
            print(f"every server failed; waiting {pause} s before trying again ...")
            time.sleep(pause)
        for url in ENDPOINTS:
            try:
                payload = _query(url, body)
            # OSError covers HTTP errors, refused connections and timeouts;
            # ValueError covers a reply that is not the JSON we asked for.
            except (OSError, http.client.HTTPException, ValueError) as error:
                print(f"  failed: {error}")
                last_error = error
                continue
            payload["bbox_wsen"] = list(bbox_wsen)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            print(f"wrote {dest} ({len(payload['elements'])} elements, {dest.stat().st_size / 1e6:.1f} MB)")
            return dest
    raise RuntimeError(f"every Overpass server failed {ROUNDS} times; the last error was: {last_error}")


def data_age(payload: dict, now: datetime | None = None) -> float | None:
    """How many days old the server's copy of the map is, or None if the reply does not say."""
    stamp = (payload.get("osm3s") or {}).get("timestamp_osm_base")
    try:
        base = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return ((now or datetime.now(timezone.utc)) - base).total_seconds() / 86400


def _query(url: str, body: bytes) -> dict:
    print(f"querying {url} ...")
    request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    payload = json.loads(raw)
    if not payload.get("elements"):
        raise ValueError(f"no elements returned: {payload.get('remark', 'empty response')}")
    age = data_age(payload)
    if age is None:
        raise ValueError("the reply does not say how old its copy of the map is")
    if age > MAX_AGE_DAYS:
        stamp = payload["osm3s"]["timestamp_osm_base"]
        raise ValueError(f"its copy of the map is {age:.0f} days old ({stamp}); the limit is {MAX_AGE_DAYS} days")
    return payload


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

"""Download OpenStreetMap data for a bounding box through the Overpass API."""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
USER_AGENT = "city-model-pipeline/0.1"

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
  node["natural"="tree"]({{bbox}});
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
    for attempt, url in enumerate(ENDPOINTS * 2):
        if attempt:
            time.sleep(5 * attempt)
        print(f"querying {url} ...")
        request = urllib.request.Request(
            url, data=body, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            payload = json.loads(raw)
            if not payload.get("elements"):
                raise ValueError(f"no elements returned: {payload.get('remark', 'empty response')}")
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            print(f"  failed: {error}")
            last_error = error
            continue
        payload["bbox_wsen"] = list(bbox_wsen)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {dest} ({len(payload['elements'])} elements, {dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    raise RuntimeError(f"all Overpass endpoints failed: {last_error}")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

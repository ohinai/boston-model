"""Draw what the model contains on top of an aerial photograph, cell by cell.

A person or an assistant can then look at the photograph and the overlay and
spot where the model is wrong: a tree that is not there, a path in the wrong
place, a pond with the wrong outline. Each cell also gets a `features.json`
that explains every label on the picture.

Networking (`fetch_photo`) is kept apart from drawing (`render_cell`) so the
drawing can be tested with a plain image.
"""

from __future__ import annotations

import http.client
import json
import math
import time
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

from . import trees, world
from .features import Scene

USER_AGENT = "city-model-pipeline/0.1"
ATTEMPTS = 3
RETRY_PAUSE = 5  # seconds between tries


@dataclass(frozen=True)
class PhotoSource:
    name: str
    url: str  # template: {bbox} is E0,N0,E1,N1 in the world CRS (EPSG:26986), {size} is the pixel width and height
    attribution: str


# NAIP is served in EPSG:26986, the model's own CRS shifted by an origin, so a
# north-up image lines up with the model with no reprojection. A sharper source can be added here.
NAIP = PhotoSource(
    name="naip",
    url="https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
    "?bbox={bbox}&bboxSR=26986&imageSR=26986&size={size}&format=jpg&f=image",
    attribution="USGS NAIP aerial imagery, 7 July 2023 (public domain)",
)
SOURCE = NAIP
# The same survey as a vegetation-index picture (near-infrared against red), colour-ramped by the service:
# dense tree canopy is dark green, lawn lighter green, pavement and water cream or orange.
NDVI = PhotoSource(
    name="naip_ndvi",
    url=NAIP.url + "&renderingRule=%7B%22rasterFunction%22%3A%22NDVI_Color%22%7D",
    attribution=NAIP.attribution + ", vegetation index",
)

# A pixel of the NDVI_Color picture is canopy when it is clearly green (G > R + GREEN_MARGIN) and dark
# in red (R < CANOPY_MAX_RED); a majority over a SMOOTH x SMOOTH box then removes speckle. Tuned by eye on
# the Public Garden lagoon cell. Known weaknesses: very light sunlit foliage (willows) reads as lawn, so
# big canopies come out with holes; deep shade on lawn can read as canopy.
GREEN_MARGIN = 20
CANOPY_MAX_RED = 62
SMOOTH = 5

# Which canopy blobs become proposals or crown measurements (areas in square metres).
OWNER_REACH = 1.5  # a model tree belongs to a blob when its trunk is inside it or this close (m)
PROPOSE_AREA = (2.0, 60.0)
PROPOSE_FILL = 0.55  # blob area over bounding-box area: a round crown fills about 0.785
PROPOSE_ASPECT = 1.6
MEASURE_AREA = (3.0, 400.0)
MEASURE_FILL = 0.5
# A proposed crown stands apart: in the ring from HALO[0] to HALO[1] m beyond its edge, at most this share of
# pixels may be canopy. Otherwise it is a dark fragment inside a bigger, lighter crown (or a woodland edge).
PROPOSE_HALO = 0.2
HALO = (0.5, 2.5)
TREE_MARGIN = 25.0  # trees this far outside the cell are still considered (a blob there could be theirs)

YELLOW, ORANGE = "#ffe600", "#ff8c00"  # inventory trees, other trees
CYAN, WHITE, MAGENTA = "#00e5ff", "#ffffff", "#ff3df2"
BLUE, RED, GREY = "#3d7bff", "#ff4040", "#9a9a9a"
DARK = "#101010"
MONUMENT_KINDS = {"plaque", "stele", "obelisk", "column", "statue", "sculpture", "monument", "fountain"}

Bounds = tuple[float, float, float, float]


# --- pixel mapping ----------------------------------------------------------------------------

def to_pixel(x: float, y: float, bounds: Bounds, pixels: int) -> tuple[float, float]:
    """(column, row) of a world point; north is up, so row 0 is the cell's northern edge."""
    x0, y0, x1, y1 = bounds
    return (x - x0) / (x1 - x0) * pixels, (y1 - y) / (y1 - y0) * pixels


def to_world(col: float, row: float, bounds: Bounds, pixels: int) -> tuple[float, float]:
    """Exact inverse of `to_pixel`."""
    x0, y0, x1, y1 = bounds
    return x0 + col / pixels * (x1 - x0), y1 - row / pixels * (y1 - y0)


# --- cells ------------------------------------------------------------------------------------

def cell_name(i: int, j: int) -> str:
    return f"c_{i}_{j}"


def cell_bounds(i: int, j: int, cell: float) -> Bounds:
    return (i * cell, j * cell, (i + 1) * cell, (j + 1) * cell)


def cells_for(geometry, cell: float) -> list[tuple[int, int]]:
    """Grid cells (aligned to multiples of `cell` metres) whose square intersects the geometry."""
    x0, y0, x1, y1 = geometry.bounds
    found = []
    for i in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
        for j in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
            if geometry.intersects(box(*cell_bounds(i, j, cell))):
                found.append((i, j))
    return found


def park_geometry(scene: Scene, name: str):
    """The union of the named green and bed areas, or exit listing the names there are."""
    areas = [*scene.areas["green"], *scene.areas["bed"]]
    wanted = [a.polygon for a in areas if (a.tags.get("name") or "").casefold() == name.casefold()]
    if not wanted:
        names = sorted({a.tags["name"] for a in areas if a.tags.get("name")})
        raise SystemExit(f"no park named '{name}'; named areas available:\n  " + "\n  ".join(names))
    from shapely.ops import unary_union
    return unary_union(wanted)


def square_around(lat: float, lon: float, size: float):
    x, y = (float(v) for v in world.to_local(lon, lat))
    return box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)


# --- the photograph ---------------------------------------------------------------------------

def _need_pil():
    try:
        import PIL.Image  # noqa: F401
        import PIL.ImageDraw  # noqa: F401
    except ImportError:
        raise SystemExit("the audit command needs Pillow; install with: pip install -e './pipeline[audit]'") from None


def photo_url(bounds: Bounds, pixels: int, source: PhotoSource = SOURCE) -> str:
    # World metres are the CRS metres less the origin; the service wants the CRS values.
    x0, y0, x1, y1 = bounds
    bbox = ",".join(f"{v:.3f}" for v in (x0 + world.ORIGIN_E, y0 + world.ORIGIN_N, x1 + world.ORIGIN_E, y1 + world.ORIGIN_N))
    return source.url.format(bbox=bbox, size=f"{pixels},{pixels}")


def download(url: str) -> bytes:
    """The bytes at a URL, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(RETRY_PAUSE)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as reply:
                data = reply.read()
            if data[:2] != b"\xff\xd8":  # the server answers errors with JSON or text, not a JPEG
                raise ValueError(f"not a JPEG: {data[:200]!r}")
            return data
        except (OSError, http.client.HTTPException, ValueError) as error:
            print(f"  failed: {error}")
            last_error = error
    raise RuntimeError(f"the photograph download failed {ATTEMPTS} times; the last error was: {last_error}")


def fetch_photo(bounds: Bounds, pixels: int, cache: Path, force: bool = False, source: PhotoSource = SOURCE):
    """The photograph of a cell, from the cache file when there is one."""
    _need_pil()
    from PIL import Image
    if force or not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(download(photo_url(bounds, pixels, source)))
    return Image.open(cache).convert("RGB")


# --- the canopy: mask, blobs, proposals --------------------------------------------------------

def _box_mean(values: np.ndarray, size: int) -> np.ndarray:
    """Mean over a size x size window centred on each pixel (edges repeat), via a summed-area table."""
    k = size // 2
    table = np.pad(values, k + 1, mode="edge").cumsum(0).cumsum(1)
    n, m = values.shape
    total = table[size:size + n, size:size + m] - table[:n, size:size + m] - table[size:size + n, :m] + table[:n, :m]
    return total / (size * size)


def canopy_mask(picture) -> np.ndarray:
    """Boolean canopy mask of an NDVI_Color picture (PIL image or RGB array)."""
    rgb = np.asarray(picture.convert("RGB") if hasattr(picture, "convert") else picture).astype(np.int16)
    red, green = rgb[..., 0], rgb[..., 1]
    raw = (green > red + GREEN_MARGIN) & (red < CANOPY_MAX_RED)
    return _box_mean(raw.astype(np.float32), SMOOTH) > 0.5


@dataclass
class Blob:
    id: int
    area_px: int
    col: float  # centroid, pixel coordinates (pixel centres at +0.5)
    row: float
    c0: int  # bounding box, inclusive
    r0: int
    c1: int
    r1: int

    @property
    def box_px(self) -> int:
        return (self.c1 - self.c0 + 1) * (self.r1 - self.r0 + 1)

    @property
    def fill(self) -> float:
        return self.area_px / self.box_px

    @property
    def aspect(self) -> float:
        w, h = self.c1 - self.c0 + 1, self.r1 - self.r0 + 1
        return max(w, h) / min(w, h)

    def touches_edge(self, shape: tuple[int, int]) -> bool:
        return self.c0 == 0 or self.r0 == 0 or self.c1 == shape[1] - 1 or self.r1 == shape[0] - 1


def label_blobs(mask: np.ndarray) -> tuple[np.ndarray, list[Blob]]:
    """Label 8-connected blobs: (label image, 0 = background; blobs, ids from 1).

    Works on horizontal runs of pixels joined by a union-find, which is far faster in plain Python
    than filling pixel by pixel.
    """
    height, width = mask.shape
    padded = np.zeros((height, width + 2), np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)  # same rows in the same order; exclusive end columns
    rows, starts, ends = rows.tolist(), starts.tolist(), ends.tolist()
    parent = list(range(len(rows)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first = np.searchsorted(rows, np.arange(height + 1)).tolist()
    for r in range(1, height):
        i, j = first[r - 1], first[r]
        i_end, j_end = first[r], first[r + 1]
        while i < i_end and j < j_end:
            if starts[j] <= ends[i] and starts[i] <= ends[j]:  # touching diagonally counts
                parent[find(i)] = find(j)
            if ends[i] <= ends[j]:
                i += 1
            else:
                j += 1

    labels = np.zeros((height, width), np.int32)
    ids: dict[int, int] = {}
    sums: list[list] = []
    for k, (r, s, e) in enumerate(zip(rows, starts, ends)):
        root = find(k)
        blob = ids.get(root)
        if blob is None:
            blob = ids[root] = len(ids) + 1
            sums.append([0, 0.0, 0.0, s, r, e - 1, r])
        entry = sums[blob - 1]
        n = e - s
        entry[0] += n
        entry[1] += n * (s + e) / 2  # sum of pixel-centre columns
        entry[2] += n * (r + 0.5)
        entry[3], entry[4], entry[5], entry[6] = min(entry[3], s), min(entry[4], r), max(entry[5], e - 1), max(entry[6], r)
        labels[r, s:e] = blob
    blobs = [Blob(n + 1, int(a), cs / a, rs / a, int(c0), int(r0), int(c1), int(r1)) for n, (a, cs, rs, c0, r0, c1, r1) in enumerate(sums)]
    return labels, blobs


@dataclass
class Canopy:
    """What the canopy picture says about one cell. Positions are pixels and world metres."""

    mask: np.ndarray
    share: float
    proposed: list[dict]  # label, px, pos, crown_m, area_m2 (north to south, then west to east)
    measured: dict[str, dict]  # tree id -> {"seen_crown": m, "seen_px": [col, row]}


def _crown(area_m2: float) -> float:
    return 2 * math.sqrt(area_m2 / math.pi)


def _halo_share(mask: np.ndarray, blob: Blob, radius_px: float, mpp: float) -> float:
    """Share of canopy pixels in the ring around a blob (measured from the radius of an equal-area disc)."""
    lo, hi = radius_px + HALO[0] / mpp, radius_px + HALO[1] / mpp
    r0, r1 = max(0, math.floor(blob.row - hi)), min(mask.shape[0], math.ceil(blob.row + hi) + 1)
    c0, c1 = max(0, math.floor(blob.col - hi)), min(mask.shape[1], math.ceil(blob.col + hi) + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    d2 = (xx + 0.5 - blob.col) ** 2 + (yy + 0.5 - blob.row) ** 2
    ring = (d2 >= lo * lo) & (d2 <= hi * hi)
    return float(mask[r0:r1, c0:c1][ring].mean()) if ring.any() else 0.0


def analyse_canopy(mask: np.ndarray, bounds: Bounds, trees: list, blockers: list = ()) -> Canopy:
    """Propose crowns the data lacks and measure crowns it has, from a canopy mask.

    `trees` is a list of (id, x, y, crown_width) in world metres (include some beyond the cell's edge),
    `blockers` shapely geometries (buildings, roads) in which a proposed tree is not believable.
    Blobs touching the picture's edge are skipped, not extended: their true size is unknown.
    """
    pixels = mask.shape[0]
    mpp = (bounds[2] - bounds[0]) / pixels
    labels, blobs = label_blobs(mask)
    by_id = {b.id: b for b in blobs}

    # Which blobs does each tree belong to: those under its trunk or within OWNER_REACH of it.
    reach = OWNER_REACH / mpp
    owned: dict[int, list[str]] = {}
    spots = {}
    for ident, x, y, _ in trees:
        col, row = to_pixel(x, y, bounds, pixels)
        spots[ident] = (col, row)
        r0, r1 = max(0, math.floor(row - reach)), min(pixels, math.ceil(row + reach) + 1)
        c0, c1 = max(0, math.floor(col - reach)), min(pixels, math.ceil(col + reach) + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        near = (xx + 0.5 - col) ** 2 + (yy + 0.5 - row) ** 2 <= reach**2
        for blob in np.unique(labels[r0:r1, c0:c1][near]):
            if blob:
                owned.setdefault(int(blob), []).append(ident)

    blocked = prep(unary_union(list(blockers))) if blockers else None
    proposals, measured = [], {}
    for blob in blobs:
        if blob.touches_edge(mask.shape):
            continue
        area = blob.area_px * mpp * mpp
        x, y = to_world(blob.col, blob.row, bounds, pixels)
        mine = owned.get(blob.id, [])
        if len(mine) == 1 and MEASURE_AREA[0] <= area <= MEASURE_AREA[1] and blob.fill >= MEASURE_FILL:
            measured[mine[0]] = {"seen_crown": round(_crown(area), 2), "seen_px": [round(blob.col), round(blob.row)]}
        if mine or not (PROPOSE_AREA[0] <= area <= PROPOSE_AREA[1]) or blob.fill < PROPOSE_FILL or blob.aspect > PROPOSE_ASPECT:
            continue
        crown = _crown(area)
        clear = max(2.5, crown / 2 + OWNER_REACH)
        # Too near a modelled tree, or inside its modelled crown: probably that tree's own foliage.
        if any(math.hypot(tx - x, ty - y) < max(clear, tc / 2) for _, tx, ty, tc in trees):
            continue
        if blocked is not None and blocked.intersects(ShapelyPoint(x, y)):
            continue
        if _halo_share(mask, blob, math.sqrt(area / math.pi) / mpp, mpp) > PROPOSE_HALO:
            continue
        proposals.append({
            "px": [round(blob.col), round(blob.row)], "pos": [round(x, 2), round(y, 2)], "crown_m": round(crown, 2), "area_m2": round(area, 1),
        })
    proposals.sort(key=lambda p: (-p["pos"][1], p["pos"][0]))
    for n, p in enumerate(proposals, 1):
        lon, lat = (float(v) for v in world.to_lonlat(*p["pos"]))
        p.update(label=f"C{n}", lat=round(lat, 7), lon=round(lon, 7))
    proposals = [{k: p[k] for k in ("label", "px", "pos", "lat", "lon", "crown_m", "area_m2")} for p in proposals]
    return Canopy(mask, round(float(mask.mean()), 3), proposals, measured)


def analyse_cell(scene: Scene, bounds: Bounds, ndvi) -> Canopy:
    """The canopy of a cell from its NDVI picture and the scene (trees, buildings and roads in and around it)."""
    pixels = ndvi.size[0]
    reach = box(*bounds).buffer(TREE_MARGIN)
    trees_near = [
        (t.id, t.x, t.y, trees.describe(t.tags, zlib.crc32(t.id.encode())).crown_width)
        for t in scene.trees if reach.intersects(ShapelyPoint(t.x, t.y))
    ]
    cell = box(*bounds)
    blockers = [b.polygon for b in scene.buildings if b.polygon.intersects(cell)]
    blockers += [w.line.buffer(w.width / 2, quad_segs=3) for w in scene.ways if w.surface == "road" and w.line.intersects(cell)]
    blockers += [a.polygon for a in scene.areas["road"] if a.polygon.intersects(cell)]
    return analyse_canopy(canopy_mask(ndvi), bounds, trees_near, blockers)


def canopy_picture(photo, mask: np.ndarray):
    """The photograph with the canopy mask's outline in magenta (a debugging aid)."""
    from PIL import Image
    inner = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    pixels = np.asarray(photo.convert("RGB")).copy()
    pixels[mask & ~inner] = tuple(int(MAGENTA[i:i + 2], 16) for i in (1, 3, 5))
    return Image.fromarray(pixels)


# --- drawing ----------------------------------------------------------------------------------

def _parts(geometry):
    return getattr(geometry, "geoms", [geometry])


def _outline(draw, geometry, clip, bounds, pixels, colour, width) -> None:
    """The boundary of a polygon (or the line itself), clipped to the cell, as thin lines."""
    if geometry.is_empty:
        return
    edges = geometry if geometry.geom_type in ("LineString", "MultiLineString") else geometry.boundary
    for part in _parts(edges.intersection(clip)):
        if part.geom_type != "LineString" or len(part.coords) < 2:
            continue
        draw.line([to_pixel(x, y, bounds, pixels) for x, y, *_ in part.coords], fill=colour, width=width)


def _label(draw, xy, text, colour, font) -> None:
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, fill=DARK, font=font)
    draw.text((x, y), text, fill=colour, font=font)


def _ring(draw, x, y, r, colour, width) -> None:
    draw.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=width)


def render_cell(scene: Scene, decks: list, bounds: Bounds, pixels: int, photo, ndvi=None, canopy: Canopy | None = None):
    """Draw the model on a copy of `photo`; returns (overlay image, features dict).

    `decks` is a list of (way, raised line) pairs, the footbridges that are raised.
    With `ndvi` (the vegetation-index picture of the same cell), or a ready `canopy` analysis, the
    canopy's proposed trees and measured crowns are drawn last, in white, and listed in the features.
    """
    _need_pil()
    if canopy is None and ndvi is not None:
        canopy = analyse_cell(scene, bounds, ndvi)
    from PIL import ImageDraw, ImageFont
    image = photo.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(11, pixels // 70))
    mpp = (bounds[2] - bounds[0]) / pixels
    cell = box(*bounds)
    px = lambda x, y: to_pixel(x, y, bounds, pixels)  # noqa: E731
    here = lambda geometry: geometry.intersects(cell)  # noqa: E731
    outline = lambda geometry, colour, width: _outline(draw, geometry, cell, bounds, pixels, colour, width)  # noqa: E731
    strip = lambda way: way.line.buffer(way.width / 2, quad_segs=3)  # noqa: E731
    features: dict = {
        "cell": None, "bounds": list(bounds), "pixels": pixels, "metres_per_pixel": round(mpp, 4),
        "centre": None, "source": SOURCE.attribution, "trees": [], "paths": [], "beds": [], "bridges": [], "objects": {},
    }

    for area in scene.areas["water"]:
        if here(area.polygon):
            outline(area.polygon, BLUE, 1)
    roads = [w for w in scene.ways if w.surface == "road" and here(w.line)]
    for way in roads:
        outline(strip(way), WHITE, 1)
    for area in scene.areas["road"]:
        if here(area.polygon):
            outline(area.polygon, WHITE, 1)

    for way in (w for w in scene.ways if w.surface == "path" and here(w.line)):
        outline(strip(way), CYAN, 1)
        features["paths"].append({"id": way.id, "name": way.name, "kind": way.kind, "surface": way.tags.get("surface"), "width": round(way.width, 1)})
    for area in scene.areas["path"]:
        if here(area.polygon):
            outline(area.polygon, CYAN, 1)
            features["paths"].append({"id": area.id, "name": area.tags.get("name"), "kind": area.tags.get("highway") or "area", "surface": area.tags.get("surface"), "width": None})

    for bed in scene.areas["bed"]:
        if here(bed.polygon):
            outline(bed.polygon, MAGENTA, 2)
            features["beds"].append({"id": bed.id, "area": round(bed.polygon.intersection(cell).area, 1)})

    for building in scene.buildings:
        if here(building.polygon):
            outline(building.polygon, RED, 1)

    labels = []
    for way, line in (d for d in decks if here(d[1])):
        deck = line.buffer(way.width / 2, quad_segs=3)
        outline(deck, WHITE, 2)
        name = f"B{len(features['bridges']) + 1}"
        spot = line.interpolate(0.5, normalized=True)
        labels.append((px(spot.x, spot.y), name, WHITE))
        features["bridges"].append({"label": name, "id": way.id, "name": way.name, "width": round(way.width, 1), "length": round(line.length, 1)})

    for thing in scene.objects:
        if not (bounds[0] <= thing.x < bounds[2] and bounds[1] <= thing.y < bounds[3]):
            continue
        features["objects"][thing.kind] = features["objects"].get(thing.kind, 0) + 1
        cx, cy = px(thing.x, thing.y)
        if thing.kind == "bench":
            draw.rectangle([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=RED)
        elif thing.kind == "lamp":
            draw.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=WHITE)
        elif thing.kind in MONUMENT_KINDS:
            draw.polygon([(cx, cy - 3.5), (cx + 3.5, cy), (cx, cy + 3.5), (cx - 3.5, cy)], fill=MAGENTA)
        else:
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=GREY)

    # A tree belongs to the cell its trunk stands in and is numbered north to south, then west to east.
    # A crown reaching in from a neighbouring cell is drawn (so it is not mistaken for a missing tree) but not listed.
    shapes = {t.id: trees.describe(t.tags, zlib.crc32(t.id.encode())) for t in scene.trees}
    reach = prep(cell)
    inside = sorted(
        (t for t in scene.trees if bounds[0] <= t.x < bounds[2] and bounds[1] <= t.y < bounds[3]), key=lambda t: (-t.y, t.x)
    )
    listed = {t.id for t in inside}
    for tree in scene.trees:
        if tree.id not in listed and reach.intersects(box(tree.x, tree.y, tree.x, tree.y).buffer(shapes[tree.id].crown_width / 2)):
            cx, cy = px(tree.x, tree.y)
            _ring(draw, cx, cy, shapes[tree.id].crown_width / 2 / mpp, YELLOW if tree.id.startswith("bprd:") else ORANGE, 2)
    for n, tree in enumerate(inside, 1):
        shape = shapes[tree.id]
        colour = YELLOW if tree.id.startswith("bprd:") else ORANGE
        cx, cy = px(tree.x, tree.y)
        _ring(draw, cx, cy, shape.crown_width / 2 / mpp, colour, 2)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=colour)
        labels.append(((cx + 4, cy - 12), str(n), colour))
        girth = trees._number(tree.tags.get("circumference"))
        entry = {
            "n": n, "id": tree.id, "species": shape.name or tree.tags.get("species"), "form": shape.form,
            "height": round(shape.height, 1), "crown_width": round(shape.crown_width, 2),
            "trunk_cm": round(girth / math.pi * 100) if girth else None, "measured": shape.measured,
            "px": [round(cx), round(cy)], "pos": [round(tree.x, 2), round(tree.y, 2)],
        }
        if canopy is not None and tree.id in canopy.measured:
            entry.update(canopy.measured[tree.id])
        features["trees"].append(entry)
    for xy, text, colour in labels:
        _label(draw, xy, text, colour, font)
    if canopy is not None:
        features["proposed"] = canopy.proposed
        features["canopy_share"] = canopy.share
        draw_canopy_marks(draw, canopy, mpp, font)
    return image, features


def draw_canopy_marks(draw, canopy: Canopy, mpp: float, font) -> None:
    """White circles: a thin one at each measured crown, a heavier labelled one at each proposed tree."""
    for seen in canopy.measured.values():
        col, row = seen["seen_px"]
        _ring(draw, col, row, seen["seen_crown"] / 2 / mpp, WHITE, 1)
    for p in canopy.proposed:
        col, row = p["px"]
        r = p["crown_m"] / 2 / mpp
        _ring(draw, col, row, r, WHITE, 2)
        _label(draw, (col + 0.7 * r + 2, row - 0.7 * r - 13), p["label"], WHITE, font)


# --- running ----------------------------------------------------------------------------------

def run(
    area: str, scene: Scene, geometry, decks: list, cache_root: Path, out_root: Path, cell: float, pixels: int,
    force: bool = False, canopy: bool = True,
) -> int:
    """Write photograph, overlay and features for every cell touching `geometry`; returns the cell count.

    With `canopy` (the default) the vegetation-index picture is fetched too, and each cell also gets
    `canopy.jpg` and proposed trees and measured crowns in the overlay and features.
    """
    _need_pil()
    out = out_root / area
    index = {
        "source": SOURCE.attribution, "cell": cell, "pixels": pixels, "metres_per_pixel": round(cell / pixels, 4), "cells": [],
    }
    for i, j in sorted(cells_for(geometry, cell)):
        name, bounds = cell_name(i, j), cell_bounds(i, j, cell)
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        photo = fetch_photo(bounds, pixels, cache_root / "aerial" / SOURCE.name / f"{name}_{cell:g}m_{pixels}px.jpg", force)
        found = None
        if canopy:
            ndvi = fetch_photo(bounds, pixels, cache_root / "aerial" / NDVI.name / f"{name}_{cell:g}m_{pixels}px.jpg", force, NDVI)
            found = analyse_cell(scene, bounds, ndvi)
            canopy_picture(photo, found.mask).save(folder / "canopy.jpg", quality=92)
        overlay, features = render_cell(scene, decks, bounds, pixels, photo, canopy=found)
        lon, lat = (float(v) for v in world.to_lonlat((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2))
        features.update(cell=name, centre={"lat": round(lat, 6), "lon": round(lon, 6)})
        photo.save(folder / "aerial.jpg", quality=92)
        overlay.save(folder / "overlay.jpg", quality=92)
        (folder / "features.json").write_text(json.dumps(features, indent=1), encoding="utf-8")
        index["cells"].append({"name": name, "bounds": list(bounds), "centre": features["centre"]})
        extra = ""
        if canopy:
            seen = sum("seen_crown" in t for t in features["trees"])
            extra = f", {len(features['proposed'])} proposed, {seen} crowns measured, canopy {features['canopy_share']:.0%}"
        print(f"{name}: {len(features['trees'])} trees{extra}, {len(features['paths'])} paths, {len(features['beds'])} beds -> {folder}")
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(index['cells'])} cells written to {out}")
    return len(index["cells"])

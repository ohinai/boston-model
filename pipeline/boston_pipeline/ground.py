"""Lay ground-level photographs of a park beside pictures of the model from the same spots.

Photographs come from Wikimedia Commons (geotagged, recent, openly licensed). For each one the viewer
is asked (see `capture.py`) for pictures of the model from the same position, looking the way the
camera faced, or all round when that is unknown. A `sheet.jpg` puts them together so a person or an
assistant can compare them.

Networking (`find_photos`, `download_photos`) is kept apart from parsing, selection and drawing so
those can be tested without a network.
"""

from __future__ import annotations

import html
import http.client
import json
import math
import re
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep

from . import audit, world

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = audit.USER_AGENT
ATTEMPTS = audit.ATTEMPTS
RETRY_PAUSE = audit.RETRY_PAUSE
POLITE_PAUSE = 0.3  # seconds between API requests
DOWNLOAD_PAUSE = 1.0  # between image downloads; the image servers answer 429 to faster clients
BACKOFF = 30  # seconds to wait after a refused download before the whole sequence of tries is repeated
ROUNDS = 2

GRID_SPACING = 150.0  # centres of the geosearch queries, metres
SEARCH_RADIUS = 120  # metres; wider than half the diagonal (106 m) so the grid leaves no gaps
THUMB_WIDTH = 1280  # a standard thumbnail size; other widths are generated on demand and the servers throttle them (429)
MIN_WIDTH = 800
MIN_YEAR = 2015
SQUARE = 35.0  # at most one photograph per square of this side (m)
COARSE = 100.0  # the limit is spread by taking turns over squares of this side (m)
DEFAULT_LIMIT = 40
SPREAD = 35.0  # degrees either side of a known camera direction
ALL_ROUND = (0, 60, 120, 180, 240, 300)

SHEET_WIDTH = 1536
PHOTO_WIDTH, PHOTO_MAX_HEIGHT = 1024, 768
TILE_WIDTH = 512
CAPTION_HEIGHT, LABEL_HEIGHT, GAP = 36, 26, 8
COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass
class Photo:
    pageid: int
    title: str
    page_url: str
    thumb_url: str
    mime: str
    width: int
    height: int
    author: str
    licence: str
    licence_class: str  # "data" (public domain, CC0, CC BY) or "look" (CC BY-SA and other open licences)
    date: str
    year: int | None
    stamp: int  # yyyymmdd where known (month and day 0 otherwise); to find the newest
    description: str
    lat: float
    lon: float
    x: float
    y: float
    direction: float | None  # degrees clockwise from north that the camera faced, if recorded


# --- parsing ----------------------------------------------------------------------------------

def strip_html(text) -> str:
    """Plain text of an HTML fragment: tags removed, entities decoded, whitespace collapsed."""
    if text is None:
        return ""
    text = re.sub(r"<[^>]*>", " ", str(text))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_year(text) -> int | None:
    found = re.search(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)", strip_html(text))
    return int(found.group(1)) if found else None


def parse_stamp(text) -> int:
    clean = strip_html(text)
    iso = re.search(r"(?<!\d)((?:1[5-9]|20)\d\d)-(\d\d)-(\d\d)", clean)
    if iso:
        return int(iso.group(1)) * 10000 + int(iso.group(2)) * 100 + int(iso.group(3))
    year = parse_year(clean)
    return year * 10000 if year else 0


def licence_class(name) -> str | None:
    """"data", "look" or None (no licence, or not free) for a Commons licence short name."""
    text = strip_html(name).casefold()
    if not text:
        return None
    if re.search(r"non-?commercial|\bnc\b|-nc\b|\bnd\b|-nd\b|no derivs|fair use|all rights reserved|copyrighted", text):
        return None
    if re.search(r"\bsa\b|-sa\b|share ?alike", text):
        return "look"
    if re.search(r"public domain|\bcc0\b|cc-zero|\bpd\b|^pd-", text):
        return "data"
    if re.search(r"\bcc[- ]by\b", text):
        return "data"
    return "look"


def parse_direction(value) -> float | None:
    """Degrees from an EXIF value such as 123, 123.5, "123/1" or "1230/10"; None if absent or unreadable."""
    if value is None or value == "":
        return None
    try:
        text = str(value).strip()
        if "/" in text:
            top, bottom = text.split("/", 1)
            number = float(top) / float(bottom)
        else:
            number = float(text)
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(number):
        return None
    return number % 360


def _meta_value(meta: dict, key: str) -> str:
    entry = meta.get(key)
    return strip_html(entry.get("value") if isinstance(entry, dict) else entry)


def parse_record(page: dict) -> Photo | None:
    """One Commons page (with imageinfo and coordinates) as a Photo, or None when it lacks the essentials."""
    infos = page.get("imageinfo") or []
    coords = page.get("coordinates") or []
    if not infos or not coords:
        return None
    info = infos[0]
    coord = coords[0]
    if "lat" not in coord or "lon" not in coord:
        return None
    lat, lon = float(coord["lat"]), float(coord["lon"])
    ext = info.get("extmetadata") or {}
    exif = {m.get("name"): m.get("value") for m in info.get("metadata") or [] if isinstance(m, dict)}
    title = _meta_value(ext, "ObjectName") or re.sub(r"^File:", "", str(page.get("title", "")))
    date = _meta_value(ext, "DateTimeOriginal")
    x, y = (float(v) for v in world.to_local(lon, lat))
    licence = _meta_value(ext, "LicenseShortName")
    description = _meta_value(ext, "ImageDescription")
    if len(description) > 300:
        description = description[:297].rstrip() + "..."
    return Photo(
        pageid=int(page["pageid"]), title=title, page_url=info.get("descriptionurl") or "",
        thumb_url=info.get("thumburl") or info.get("url") or "",
        mime=str(info.get("mime") or ""), width=int(info.get("width") or 0), height=int(info.get("height") or 0),
        author=_meta_value(ext, "Artist") or "unknown", licence=licence, licence_class=licence_class(licence) or "",
        date=date, year=parse_year(date), stamp=parse_stamp(date), description=description,
        lat=lat, lon=lon, x=x, y=y, direction=parse_direction(exif.get("GPSImgDirection")),
    )


# --- selection --------------------------------------------------------------------------------

def is_eligible(photo: Photo, geometry, min_year: int = MIN_YEAR) -> bool:
    return bool(
        photo.mime == "image/jpeg"
        and photo.thumb_url
        and photo.width >= MIN_WIDTH
        and photo.year is not None and photo.year >= min_year
        and photo.licence_class in ("data", "look")
        and geometry.covers(ShapelyPoint(photo.x, photo.y))
    )


def eligible(pages: dict, geometry, min_year: int = MIN_YEAR) -> list[Photo]:
    """Photographs from raw Commons pages that pass every filter, in page id order."""
    inside = prep(geometry)
    found = []
    for pageid in sorted(pages, key=int):
        page = pages[pageid]
        photo = parse_record(page)
        if photo is not None and is_eligible(photo, inside, min_year):
            found.append(photo)
    return found


def _quality(photo: Photo) -> tuple:
    return (photo.direction is None, -photo.stamp, photo.pageid)


def thin(photos: list[Photo], limit: int = DEFAULT_LIMIT, square: float = SQUARE, coarse: float = COARSE) -> list[Photo]:
    """At most one photograph per `square` metres (one with a camera direction, then the newest), then at
    most `limit` of them, taking turns over coarser squares so the whole park is covered. Deterministic."""
    best: dict[tuple[int, int], Photo] = {}
    for photo in sorted(photos, key=_quality):
        best.setdefault((math.floor(photo.x / square), math.floor(photo.y / square)), photo)
    chosen = sorted(best.values(), key=_quality)
    if len(chosen) > limit:
        cells: dict[tuple[int, int], list[Photo]] = {}
        for photo in chosen:
            cells.setdefault((math.floor(photo.x / coarse), math.floor(photo.y / coarse)), []).append(photo)
        order = sorted(cells)
        picked: list[Photo] = []
        turn = 0
        while len(picked) < limit:
            for key in order:
                if turn < len(cells[key]) and len(picked) < limit:
                    picked.append(cells[key][turn])
            turn += 1
        chosen = picked
    return sorted(chosen, key=lambda p: p.pageid)


def view_headings(direction: float | None) -> list[int]:
    """Compass headings (whole degrees, 0..359) to render for a photograph."""
    if direction is None:
        return list(ALL_ROUND)
    return [round(direction + offset) % 360 for offset in (-SPREAD, 0, SPREAD)]


def view_id(pageid: int, heading: int) -> str:
    return f"{pageid}_{heading:03d}"


def compass_word(heading: float) -> str:
    return COMPASS[round(heading / 45) % 8]


# --- Wikimedia Commons ------------------------------------------------------------------------

def search_centres(geometry, spacing: float = GRID_SPACING, radius: float = SEARCH_RADIUS) -> list[tuple[float, float]]:
    """(lat, lon) of query centres on a grid over the geometry, only where a circle of `radius` reaches it."""
    x0, y0, x1, y1 = geometry.bounds
    centres = []
    nx, ny = max(1, math.ceil((x1 - x0) / spacing)), max(1, math.ceil((y1 - y0) / spacing))
    for i in range(nx):
        for j in range(ny):
            x, y = x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny
            if geometry.distance(ShapelyPoint(x, y)) <= radius:
                lon, lat = (float(v) for v in world.to_lonlat(x, y))
                centres.append((lat, lon))
    return centres


def get_json(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(RETRY_PAUSE)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as reply:
                data = json.loads(reply.read().decode("utf-8"))
            if "error" in data:
                raise ValueError(f"API error: {data['error']}")
            return data
        except (OSError, http.client.HTTPException, ValueError) as error:
            print(f"  failed: {error}")
            last_error = error
    raise RuntimeError(f"the Commons query failed {ATTEMPTS} times; the last error was: {last_error}")


def merge_page(pages: dict, page: dict) -> None:
    mine = pages.setdefault(str(page["pageid"]), {"pageid": page["pageid"]})
    for key, value in page.items():
        if key in ("imageinfo", "coordinates") and mine.get(key) and not value:
            continue
        mine[key] = value


def query_centre(lat: float, lon: float, pages: dict) -> int:
    """Add the geotagged files around a point to `pages` (by page id); returns how many came back."""
    params = {
        "action": "query", "format": "json", "formatversion": "2", "generator": "geosearch",
        "ggscoord": f"{lat:.6f}|{lon:.6f}", "ggsradius": SEARCH_RADIUS, "ggslimit": 500, "ggsnamespace": 6,
        "prop": "imageinfo|coordinates", "iiprop": "url|size|mime|extmetadata|metadata", "iiurlwidth": THUMB_WIDTH,
        "iimetadataversion": "latest", "colimit": "max", "iilimit": 1,
    }
    seen: set[int] = set()
    extra: dict = {}
    while True:
        data = get_json({**params, **extra})
        for page in data.get("query", {}).get("pages", []):
            merge_page(pages, page)
            seen.add(page["pageid"])
        if "continue" not in data:
            return len(seen)
        extra = dict(data["continue"])
        time.sleep(POLITE_PAUSE)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_") or "park"


def find_pages(geometry, park: str, cache_dir: Path, force: bool = False) -> dict:
    """Every geotagged Commons file near the park, by page id, from the cache file when there is one."""
    cache = cache_dir / f"query_{slug(park)}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))["pages"]
    pages: dict = {}
    centres = search_centres(geometry)
    print(f"searching Wikimedia Commons around {len(centres)} points ...")
    for n, (lat, lon) in enumerate(centres, 1):
        count = query_centre(lat, lon, pages)
        print(f"  {n}/{len(centres)}: {count} files ({len(pages)} distinct so far)")
        time.sleep(POLITE_PAUSE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"park": park, "centres": centres, "pages": pages}), encoding="utf-8")
    return pages


def download_photos(photos: list[Photo], cache_dir: Path, force: bool = False) -> list[Photo]:
    """Fetch each photograph's 1024 px rendition into the cache; returns those that arrived."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for photo in photos:
        target = cache_dir / f"{photo.pageid}.jpg"
        if force or not target.exists():
            data = None
            for round_ in range(ROUNDS):
                if round_:
                    time.sleep(BACKOFF)
                try:
                    data = audit.download(photo.thumb_url)
                    break
                except RuntimeError as error:
                    last = error
            if data is None:
                print(f"  skipping {photo.pageid} ({photo.title}): {last}")
                continue
            target.write_bytes(data)
            time.sleep(DOWNLOAD_PAUSE)
        kept.append(photo)
    return kept


# --- the sheet --------------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an old Pillow without a scalable default font
        return ImageFont.load_default()


def caption_text(photo: Photo) -> str:
    return f"{photo.title} — {photo.author} — {photo.licence} — {photo.date or 'date unknown'}"


def _fit(draw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def render_sheet(photo: Photo, picture, models: list[tuple[int, object]]):
    """The sheet image: caption, the photograph, then the model's pictures (heading, image or None) in rows of three."""
    from PIL import Image, ImageDraw

    pic = picture.convert("RGB")
    scale = min(PHOTO_WIDTH / pic.width, PHOTO_MAX_HEIGHT / pic.height)
    pic = pic.resize((max(1, round(pic.width * scale)), max(1, round(pic.height * scale))), Image.LANCZOS)
    arrived = [(h, im) for h, im in models if im is not None]
    tile_h = round(arrived[0][1].height * TILE_WIDTH / arrived[0][1].width) if arrived else TILE_WIDTH * 3 // 4
    rows = math.ceil(len(models) / 3)
    note_h = 0 if arrived else 60
    height = CAPTION_HEIGHT + GAP + pic.height + GAP + (rows * (LABEL_HEIGHT + tile_h + GAP) if arrived else note_h)
    sheet = Image.new("RGB", (SHEET_WIDTH, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 6), _fit(draw, caption_text(photo), _font(20), SHEET_WIDTH - 20), fill="black", font=_font(20))
    top = CAPTION_HEIGHT + GAP
    sheet.paste(pic, ((SHEET_WIDTH - pic.width) // 2, top))
    top += pic.height + GAP
    if not arrived:
        draw.text((10, top + 10), "model pictures not captured yet: run again without --no-capture", fill="#555555", font=_font(20))
        return sheet
    label_font = _font(20)
    for n, (heading, image) in enumerate(models):
        left, row_top = (n % 3) * TILE_WIDTH, top + (n // 3) * (LABEL_HEIGHT + tile_h + GAP)
        draw.text((left + 8, row_top + 2), f"model, facing {heading}° {compass_word(heading)}", fill="black", font=label_font)
        if image is None:
            draw.rectangle([left + 2, row_top + LABEL_HEIGHT, left + TILE_WIDTH - 3, row_top + LABEL_HEIGHT + tile_h - 1], outline="#999999")
            draw.text((left + 12, row_top + LABEL_HEIGHT + 12), "not captured", fill="#777777", font=label_font)
            continue
        tile = image.convert("RGB").resize((TILE_WIDTH - 4, tile_h), Image.LANCZOS)
        sheet.paste(tile, (left + 2, row_top + LABEL_HEIGHT))
    return sheet


# --- running ----------------------------------------------------------------------------------

def _info(photo: Photo, views: list[dict]) -> dict:
    return {
        "pageid": photo.pageid, "title": photo.title, "page_url": photo.page_url, "author": photo.author,
        "licence": photo.licence, "licence_class": photo.licence_class, "date": photo.date, "description": photo.description,
        "lat": photo.lat, "lon": photo.lon, "x": round(photo.x, 2), "y": round(photo.y, 2),
        "direction": None if photo.direction is None else round(photo.direction, 1), "views": views,
    }


def _write_photo_folder(photo: Photo, folder: Path, headings: list[int]) -> int:
    """(Re)write info.json and sheet.jpg from what is in the folder; returns how many model pictures it has."""
    from PIL import Image

    models, views = [], []
    for heading in headings:
        file = folder / f"model_{heading:03d}.jpg"
        image = None
        if file.exists():
            with Image.open(file) as opened:
                image = opened.convert("RGB")
        models.append((heading, image))
        views.append({"heading": heading, "id": view_id(photo.pageid, heading), "file": file.name, "captured": image is not None})
    with Image.open(folder / "photo.jpg") as opened:
        render_sheet(photo, opened, models).save(folder / "sheet.jpg", quality=92)
    (folder / "info.json").write_text(json.dumps(_info(photo, views), indent=1, ensure_ascii=False), encoding="utf-8")
    return sum(image is not None for _, image in models)


def run(
    area: str, park: str, geometry, cache_root: Path, out_root: Path, site_dir: Path, limit: int = DEFAULT_LIMIT,
    force: bool = False, do_capture: bool = True,
) -> dict:
    """Collect photographs, capture the model's views, write sheets and index.json; returns the index."""
    audit._need_pil()
    cache_dir = cache_root / "commons"
    out = out_root / area
    pages = find_pages(geometry, park, cache_dir, force)
    with_geotag = [p for p in pages.values() if p.get("coordinates")]
    candidates = eligible(pages, geometry)
    chosen = download_photos(thin(candidates, limit), cache_dir, force)

    views, folders = [], {}
    for photo in chosen:
        folder = out / str(photo.pageid)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_dir / f"{photo.pageid}.jpg", folder / "photo.jpg")
        headings = view_headings(photo.direction)
        folders[photo.pageid] = (folder, headings)
        for heading in headings:
            if force or not (folder / f"model_{heading:03d}.jpg").exists():
                views.append({"id": view_id(photo.pageid, heading), "lat": round(photo.lat, 7), "lon": round(photo.lon, 7), "heading": heading, "pitch": 0})

    capture_result = None
    if do_capture and views:
        from . import capture

        scratch = out / "_capture"
        shutil.rmtree(scratch, ignore_errors=True)
        capture_result = capture.capture(views, scratch, site_dir=site_dir)
        for view in views:
            got = scratch / f"{view['id']}.jpg"
            if got.exists():
                pageid = int(view["id"].split("_")[0])
                shutil.copyfile(got, folders[pageid][0] / f"model_{view['heading']:03d}.jpg")
        shutil.rmtree(scratch, ignore_errors=True)
        if capture_result.get("error"):
            print(f"  the viewer reported an error: {capture_result['error']}")
        for failure in capture_result.get("failed", []):
            print(f"  failed: {failure.get('id')}: {failure.get('reason')}")

    sheets = []
    for photo in chosen:
        folder, headings = folders[photo.pageid]
        count = _write_photo_folder(photo, folder, headings)
        sheets.append({
            "id": photo.pageid, "title": photo.title, "lat": photo.lat, "lon": photo.lon, "x": round(photo.x, 2), "y": round(photo.y, 2),
            "direction": None if photo.direction is None else round(photo.direction, 1), "licence_class": photo.licence_class,
            "model_pictures": count, "views": len(headings), "folder": str(folder), "sheet": str(folder / "sheet.jpg"),
        })
    classes = {c: sum(p.licence_class == c for p in chosen) for c in ("data", "look")}
    index = {
        "park": park, "found": len(with_geotag), "eligible": len(candidates), "kept": len(chosen),
        "with_direction": sum(p.direction is not None for p in chosen), "licence_classes": classes,
        "capture": None if capture_result is None else {k: capture_result.get(k) for k in ("captured", "failed", "error", "completed")},
        "sheets": sheets,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    pictures = sum(s["model_pictures"] for s in sheets)
    print(
        f"{park}: {index['found']} geotagged files found, {index['eligible']} eligible, {index['kept']} kept "
        f"({index['with_direction']} with a camera direction); licences: {classes['data']} data, {classes['look']} look\n"
        f"{len(sheets)} sheets written to {out} ({pictures} model pictures)"
    )
    return index

"""Turn raw OSM elements into typed features in world coordinates.

This is where tagging conventions are interpreted (what counts as a road, how
tall an untagged building is). Geometry generation lives in tiles.py.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from . import world

LEVEL_HEIGHT = 3.4  # metres per storey when only building:levels is known

DEFAULT_HEIGHTS = {
    "house": 9.0, "detached": 9.0, "semidetached_house": 9.0, "terrace": 11.0,
    "residential": 12.0, "apartments": 15.0, "dormitory": 15.0, "hotel": 25.0,
    "commercial": 16.0, "office": 20.0, "retail": 8.0, "industrial": 9.0,
    "warehouse": 9.0, "church": 18.0, "cathedral": 25.0, "chapel": 10.0,
    "school": 13.0, "university": 16.0, "college": 16.0, "hospital": 22.0,
    "public": 15.0, "civic": 15.0, "government": 18.0, "train_station": 12.0,
    "parking": 14.0, "garage": 3.0, "garages": 3.0, "shed": 2.8, "hut": 2.8,
    "kiosk": 3.0, "roof": 4.5, "service": 3.5, "toilets": 3.2,
}
FALLBACK_HEIGHT = 12.0

# Carriageway width in metres by highway class, used when no width or lanes tag exists.
ROAD_WIDTHS = {
    "motorway": 14.0, "trunk": 12.0, "primary": 11.0, "secondary": 9.5,
    "tertiary": 8.5, "unclassified": 7.0, "residential": 6.5, "living_street": 5.5,
    "service": 4.0, "motorway_link": 6.5, "trunk_link": 6.5, "primary_link": 6.0,
    "secondary_link": 6.0, "tertiary_link": 6.0, "busway": 7.0, "road": 6.5, "track": 3.0,
}
PATH_WIDTHS = {
    "pedestrian": 6.0, "footway": 2.2, "path": 1.8, "steps": 2.4,
    "cycleway": 2.2, "bridleway": 2.0, "platform": 3.0,
}
RAIL_WIDTH = 3.2
PIER_WIDTH = 4.0

GREEN_LEISURE = {"park", "garden", "pitch", "playground", "common", "dog_park", "golf_course"}
GREEN_LANDUSE = {"grass", "forest", "meadow", "cemetery", "recreation_ground", "village_green", "flowerbed"}
GREEN_NATURAL = {"wood", "scrub", "grassland"}

POI_AMENITIES = {
    "restaurant", "cafe", "bar", "pub", "fast_food", "bank", "pharmacy", "library",
    "theatre", "cinema", "place_of_worship", "school", "university", "college",
    "post_office", "police", "fire_station", "hospital", "townhall", "courthouse",
    "marketplace", "ferry_terminal", "ice_cream", "food_court", "clinic", "arts_centre",
}
POI_TOURISM = {"museum", "attraction", "hotel", "artwork", "gallery", "viewpoint", "aquarium"}

BRICK = ["#8a4b3c", "#93543f", "#7d4436", "#9a5b47", "#a3644c", "#86503f", "#74402f"]
STONE = ["#b8b0a2", "#a9a296", "#c2b9a8", "#9d978c", "#cfc6b4"]
CONCRETE = ["#a7a7a3", "#b5b4ae", "#96969a", "#bcbab2"]
GLASS = ["#6f8796", "#5f7888", "#7b93a1", "#8097a8", "#62737f"]
ROOFS = ["#5a5a5c", "#6b6a68", "#4f5052", "#65625e"]
MATERIAL_PALETTES = {
    "brick": BRICK, "stone": STONE, "sandstone": STONE, "limestone": STONE, "granite": STONE,
    "concrete": CONCRETE, "plaster": STONE, "glass": GLASS, "metal": CONCRETE, "steel": GLASS,
}
NAMED_COLOURS = {
    "white": "#f2f0ea", "black": "#2b2b2d", "grey": "#8f8f8f", "gray": "#8f8f8f",
    "red": "#9c3b30", "brown": "#7a5236", "beige": "#d6c7a1", "yellow": "#d9bf5b",
    "tan": "#c4a77d", "green": "#557a55", "blue": "#52718f", "orange": "#c97a3d",
    "cream": "#e8dcc0", "silver": "#b9bcc0", "maroon": "#6e2b2b", "pink": "#d9a5a5",
}


@dataclass
class Building:
    id: str
    polygon: Polygon
    height: float
    min_height: float
    height_source: str  # "height", "levels", "default" or "override"
    wall_colour: str
    roof_colour: str
    tags: dict
    part: bool = False


@dataclass
class Way:
    id: str
    line: LineString
    kind: str
    surface: str  # "road", "path" or "rail"
    width: float
    name: str | None


@dataclass
class Point:
    id: str
    x: float
    y: float
    name: str | None = None
    kind: str | None = None


@dataclass
class Scene:
    buildings: list[Building] = field(default_factory=list)
    ways: list[Way] = field(default_factory=list)
    areas: dict[str, list[Polygon]] = field(
        default_factory=lambda: {"road": [], "path": [], "water": [], "green": []}
    )
    coastlines: list[LineString] = field(default_factory=list)
    trees: list[Point] = field(default_factory=list)
    signals: list[Point] = field(default_factory=list)
    pois: list[Point] = field(default_factory=list)


def parse_length(value) -> float | None:
    """Parse an OSM length ("12", "12.5 m", "40 ft", "12'6\\"") into metres."""
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", ".")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\")?", text)
    if match:
        return float(match.group(1)) * 0.3048 + float(match.group(2) or 0) * 0.0254
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|metres?|meters?|ft|feet)?", text)
    if not match:
        return None
    metres = float(match.group(1))
    return metres * 0.3048 if match.group(2) in ("ft", "feet") else metres


def parse_number(value) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_colour(value) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", text):
        return text
    if re.fullmatch(r"#[0-9a-f]{3}", text):
        return "#" + "".join(c * 2 for c in text[1:])
    return NAMED_COLOURS.get(text)


def building_height(tags: dict) -> tuple[float, float, str]:
    """(height, min_height, source) in metres for a building or building part."""
    kind = tags.get("building") or tags.get("building:part") or "yes"
    height = parse_length(tags.get("height") or tags.get("building:height"))
    source = "height"
    if height is None or height <= 0:
        levels = parse_number(tags.get("building:levels"))
        if levels and levels > 0:
            height = levels * LEVEL_HEIGHT + 1.0
            source = "levels"
        else:
            height = DEFAULT_HEIGHTS.get(kind, FALLBACK_HEIGHT)
            source = "default"
    base = parse_length(tags.get("min_height"))
    if base is None:
        min_level = parse_number(tags.get("building:min_level"))
        base = min_level * LEVEL_HEIGHT if min_level else 0.0
    if kind == "roof":
        base = max(base, height - 0.4)
    if base >= height:
        base = max(0.0, height - 0.5)
    return height, base, source


def _pick(palette: list[str], seed: int) -> str:
    return palette[seed % len(palette)]


def building_colours(ident: str, tags: dict, height: float) -> tuple[str, str]:
    """Wall and roof colours: tagged where possible, otherwise a stable guess."""
    seed = zlib.crc32(ident.encode())
    wall = parse_colour(tags.get("building:colour"))
    if wall is None:
        palette = MATERIAL_PALETTES.get(tags.get("building:material", ""))
        if palette is None:
            roll = seed % 100
            if height > 70:
                palette = GLASS if roll < 70 else CONCRETE
            elif height > 25:
                palette = STONE if roll < 45 else CONCRETE if roll < 75 else BRICK
            else:
                palette = BRICK if roll < 65 else STONE if roll < 90 else CONCRETE
        wall = _pick(palette, seed >> 8)
    roof = parse_colour(tags.get("roof:colour")) or _pick(ROOFS, seed >> 16)
    return wall, roof


def _coords(geometry: list[dict]) -> np.ndarray:
    points = [p for p in geometry if p]
    lon = np.fromiter((p["lon"] for p in points), float, len(points))
    lat = np.fromiter((p["lat"] for p in points), float, len(points))
    x, y = world.to_local(lon, lat)
    return np.column_stack([x, y])


def polygons_of(geom) -> list[Polygon]:
    """Flatten any geometry into its valid, non-empty polygons."""
    if geom is None or geom.is_empty:
        return []
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for part in getattr(geom, "geoms", []) for g in polygons_of(part)]


def _way_polygons(coords: np.ndarray) -> list[Polygon]:
    if len(coords) < 4 or not np.allclose(coords[0], coords[-1]):
        return []
    return polygons_of(Polygon(coords))


def _relation_polygons(element: dict) -> list[Polygon]:
    outers, inners = [], []
    for member in element.get("members", []):
        if member.get("type") != "way" or not member.get("geometry"):
            continue
        coords = _coords(member["geometry"])
        if len(coords) >= 2:
            (inners if member.get("role") == "inner" else outers).append(LineString(coords))
    if not outers:
        return []
    shell = unary_union(list(polygonize(unary_union(outers))))
    if inners:
        shell = shell.difference(unary_union(list(polygonize(unary_union(inners)))))
    return polygons_of(shell)


def _is_underground(tags: dict) -> bool:
    if tags.get("tunnel", "no") != "no" or tags.get("location") == "underground":
        return True
    if tags.get("indoor", "no") != "no":
        return True
    layer = parse_number(tags.get("layer"))
    return layer is not None and layer < 0


def _parking_sides(tags: dict, kind: str) -> int:
    """How many sides of the carriageway have a parking lane."""
    if tags.get("parking:both", "no") != "no":
        return 2
    sides = sum(tags.get(f"parking:{side}", "no") != "no" for side in ("left", "right"))
    tagged = any(key.startswith("parking:") for key in tags)
    if not tagged and kind in ("residential", "unclassified", "tertiary", "living_street"):
        return 1  # the usual case on Boston side streets
    return sides


def _road_width(tags: dict, kind: str) -> float:
    """Curb-to-curb width in metres."""
    # Widths from the 2007 MassGIS import are right-of-way widths (building line
    # to building line), not carriageways, so they are ignored here.
    imported = "massgis:way_id" in tags or tags.get("source", "").startswith("massgis")
    width = None if imported else parse_length(tags.get("width"))
    if width and 1.5 <= width <= 40:
        return width
    lanes = parse_number(tags.get("lanes"))
    if lanes and 1 <= lanes <= 10:
        return lanes * 3.2 + _parking_sides(tags, kind) * 2.2 + 0.6
    if kind == "service" and tags.get("service") in ("alley", "driveway", "parking_aisle"):
        return 3.5
    return ROAD_WIDTHS.get(kind, 6.5)


def _poi_kind(tags: dict) -> str | None:
    if "shop" in tags:
        return f"shop={tags['shop']}"
    if tags.get("amenity") in POI_AMENITIES:
        return f"amenity={tags['amenity']}"
    if tags.get("tourism") in POI_TOURISM:
        return f"tourism={tags['tourism']}"
    if tags.get("railway") in ("station", "subway_entrance", "tram_stop", "halt"):
        return f"railway={tags['railway']}"
    if "historic" in tags:
        return f"historic={tags['historic']}"
    return None


def _add_building(scene: Scene, ident: str, tags: dict, polygons: list[Polygon], overrides: dict) -> None:
    if tags.get("building") == "no" or tags.get("location") == "underground":
        return
    override = overrides.get(ident, {})
    if override.get("hide"):
        return
    height, base, source = building_height(tags)
    if "height" in override:
        height, source = float(override["height"]), "override"
        base = min(base, height - 0.5)
    wall, roof = building_colours(ident, tags, height)
    wall = parse_colour(override.get("colour")) or wall
    roof = parse_colour(override.get("roof_colour")) or roof
    for polygon in polygons:
        if polygon.area >= 2.0:
            scene.buildings.append(
                Building(ident, polygon, height, base, source, wall, roof, tags, part="building:part" in tags)
            )


def _add_way(scene: Scene, ident: str, tags: dict, coords: np.ndarray) -> None:
    """Classify one OSM way. A way has exactly one role; the first match wins."""
    closed_polygons = _way_polygons(coords)
    if "highway" in tags:
        kind = tags["highway"]
        if _is_underground(tags) or kind in ("proposed", "construction", "corridor", "elevator", "razed"):
            return
        surface = "path" if kind in PATH_WIDTHS else "road"
        if tags.get("area") == "yes" and closed_polygons:
            scene.areas[surface].extend(closed_polygons)
            return
        width = PATH_WIDTHS[kind] if surface == "path" else _road_width(tags, kind)
        if surface == "path":
            width = parse_length(tags.get("width")) or width
        if len(coords) >= 2:
            scene.ways.append(Way(ident, LineString(coords), kind, surface, width, tags.get("name")))
    elif "railway" in tags:
        if _is_underground(tags):
            return
        if tags["railway"] == "platform":
            if closed_polygons:
                scene.areas["path"].extend(closed_polygons)
            elif len(coords) >= 2:
                scene.ways.append(Way(ident, LineString(coords), "platform", "path", 3.0, tags.get("name")))
        elif len(coords) >= 2:
            scene.ways.append(Way(ident, LineString(coords), tags["railway"], "rail", RAIL_WIDTH, tags.get("name")))
    elif tags.get("natural") == "coastline":
        if len(coords) >= 2:
            scene.coastlines.append(LineString(coords))
    elif tags.get("man_made") in ("pier", "breakwater", "groyne"):
        if closed_polygons:
            scene.areas["path"].extend(closed_polygons)
        elif len(coords) >= 2:
            width = parse_length(tags.get("width")) or PIER_WIDTH
            scene.areas["path"].extend(polygons_of(LineString(coords).buffer(width / 2, cap_style="flat")))
    else:
        _add_area(scene, tags, closed_polygons)


def _add_area(scene: Scene, tags: dict, polygons: list[Polygon]) -> None:
    if not polygons or _is_underground(tags):
        return
    if tags.get("natural") == "water" or tags.get("waterway") in ("dock", "riverbank"):
        scene.areas["water"].extend(polygons)
    elif (
        tags.get("leisure") in GREEN_LEISURE
        or tags.get("landuse") in GREEN_LANDUSE
        or tags.get("natural") in GREEN_NATURAL
    ):
        scene.areas["green"].extend(polygons)
    elif tags.get("man_made") == "pier":
        scene.areas["path"].extend(polygons)


def classify(elements: list[dict], overrides: dict | None = None) -> Scene:
    """Interpret raw Overpass elements. `overrides` maps OSM ids to building fixes."""
    overrides = overrides or {}
    scene = Scene()
    for element in elements:
        tags = element.get("tags") or {}
        kind = element["type"]
        ident = f"{kind[0]}{element['id']}"
        if kind == "node":
            x, y = (float(v) for v in world.to_local(element["lon"], element["lat"]))
            if tags.get("natural") == "tree":
                scene.trees.append(Point(ident, x, y))
            elif tags.get("highway") == "traffic_signals":
                scene.signals.append(Point(ident, x, y))
            elif tags.get("name") and _poi_kind(tags):
                scene.pois.append(Point(ident, x, y, tags["name"], _poi_kind(tags)))
        elif kind == "way":
            if not element.get("geometry"):
                continue
            coords = _coords(element["geometry"])
            if "building" in tags or "building:part" in tags:
                _add_building(scene, ident, tags, _way_polygons(coords), overrides)
            else:
                _add_way(scene, ident, tags, coords)
        elif kind == "relation" and tags.get("type") == "multipolygon":
            polygons = _relation_polygons(element)
            if "building" in tags or "building:part" in tags:
                _add_building(scene, ident, tags, polygons, overrides)
            else:
                _add_area(scene, tags, polygons)
    _drop_outlines_with_parts(scene)
    return scene


def _drop_outlines_with_parts(scene: Scene) -> None:
    """Simple 3D Buildings rule: an outline that contains parts is not drawn."""
    parts = [b for b in scene.buildings if b.part]
    if not parts:
        return
    tree = STRtree([p.polygon.representative_point() for p in parts])
    kept = []
    for building in scene.buildings:
        if not building.part and len(tree.query(building.polygon, predicate="contains")):
            continue
        kept.append(building)
    scene.buildings = kept

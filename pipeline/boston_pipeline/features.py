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

from . import extras, world

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
# The `height:source` of a `height` that came from the Town of Brookline's building footprints (brookline.py), not from
# OpenStreetMap: such a building's height_source is "town".
HEIGHT_FROM_TOWN = "Town of Brookline building footprints"
# The same for a `height` from the City of Boston's building survey, the roof-break parts (cityheights.py): "city".
HEIGHT_FROM_CITY = "City of Boston building roof breaks (2010 survey)"
# A building tagged `amenity=shelter` (a bus shelter, a sun or weather shelter, a picnic gazebo; 15 in the pilot, 7 to 85 m²)
# with no height is a small structure, not a 12 m block.
SHELTER_HEIGHT = 3.0

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

# What a path is made of, from its `surface` tag. Each cover is drawn in its own
# colour; anything not listed is "paved", a neutral light grey.
PATH_COVERS = {
    "asphalt": "asphalt", "concrete": "concrete", "concrete:plates": "concrete", "concrete:lanes": "concrete",
    "paving_stones": "paving_stones", "brick": "brick", "bricks": "brick", "sett": "sett",
    "cobblestone": "sett", "unhewn_cobblestone": "sett", "wood": "wood", "metal": "metal",
    "gravel": "gravel", "fine_gravel": "gravel", "compacted": "gravel", "pebblestone": "gravel",
    "dirt": "dirt", "ground": "dirt", "earth": "dirt", "unpaved": "dirt", "woodchips": "dirt", "sand": "sand",
}
# Water that is a pond or lagoon, by its `water` tag, and water that is open and large. Water
# with no `water` tag (or "yes") is a pond when it is smaller than SMALL_POND square metres.
POND_WATER = {"pond", "lagoon", "basin", "reservoir", "pool", "fountain", "stream_pool", "wastewater"}
OPEN_WATER = {"river", "canal", "lake", "harbour", "bay", "oxbow", "stream", "ditch", "tidal_channel"}
SMALL_POND = 20_000.0
# A tree that stands on the lawn of a named park or garden of at least MULCH_MIN_PARK square
# metres is drawn with a ring of mulch round its foot, as Boston's maintained parks do (the
# Public Garden and the Common above all). Not a tag in the data: the tree's own
# `tree_surround=none` (set through a file in fixes/) switches it off for that tree.
MULCH_PARKS = {"park", "garden", "common"}
MULCH_MIN_PARK = 20_000.0
# Sports and play surfaces that are not grass, by `surface` tag.
HARD_COURTS = {"asphalt", "concrete", "acrylic", "tartan", "hard", "paved", "rubber"}
BARE_GROUND = {"dirt", "clay", "sand", "earth", "ground", "gravel", "fine_gravel", "woodchips"}
# A playground whose `surface` is one of these is paving (6 of the 121 in the `boston` area are "paved"); drawn grey.
PLAYGROUND_PAVING = {"asphalt", "concrete", "paved", "paving_stones", "tiles"}
# Games played on a hard court: a pitch for one of these and no `surface` is drawn as a hard court (see green_cover).
HARD_COURT_SPORTS = {"tennis", "basketball", "pickleball", "futsal", "street_hockey", "roller_skating", "skateboard", "racquet", "handball", "netball"}

# A footbridge with no stated height rises this share of its length at mid-span,
# within these limits (metres): enough to read as a bridge, gentle enough to walk.
BRIDGE_RISE = 0.025
BRIDGE_RISE_LIMITS = (0.3, 0.8)

# A bench with no `direction` is turned to look at the nearest path, if one is
# within BENCH_REACH metres (see facing.py). Farther than that the guess is no
# better than chance, so the bench is left facing north. A bench with water within
# BENCH_WATERSIDE metres on the side away from that path looks at the water.
BENCH_REACH = 20.0
BENCH_WATERSIDE = 10.0

# A street lamp is drawn as a park lamp (a globe on a post) unless it is on a bracket arm: a `lamp_mount` of bent_mast
# or any arm, or a stated height above this many metres, taller than the park lamps (4 to 5 m; the pilot's tall ones are 7).
LAMP_ARM_MIN_HEIGHT = 5.5

# Barriers along a line, by their `barrier` tag: (kind, default height, default
# width) in metres. A `height` or `width` tag wins when it is within BARRIER_LIMITS.
# A `barrier=gate` way is the opening in a fence or wall. It is drawn open, so nobody is
# walled in: a stone pier at each end and the iron leaves swung back (see barriers.py).
# Nodes (bollard, block) are objects.
BARRIERS = {
    "fence": ("fence", 1.2, 0.06),  # a railing or paling, mostly 1 to 1.5 m in the parks
    "chain": ("chain", 0.6, 0.05),  # low posts joined by a hanging chain, along paths
    "wall": ("wall", 1.0, 0.4),  # a low masonry wall
    "city_wall": ("wall", 1.0, 0.4),
    "retaining_wall": ("retaining_wall", 1.0, 0.4),
    "hedge": ("hedge", 1.2, 0.8),
    "guard_rail": ("guard_rail", 0.7, 0.1),  # a steel beam on posts
    "kerb": ("kerb", 0.15, 0.3),  # a stone edge, not a full-height curb
    "bollard": ("bollards", 0.9, 0.0),  # a row of bollards; no width of its own
    "gate": ("gate", 1.5, 0.06),  # the way is the opening; its height is the leaves' (the parks' railings are 1.5 m)
}
BARRIER_LIMITS = {"height": (0.05, 6.0), "width": (0.01, 3.0)}
# `gate=*` values that are not a swing gate in a fence: a turnstile or ticket gate stands indoors, in a station.
NOT_SWING_GATES = {"ticket_barrier", "turnstile", "kissing", "full-height_turnstile", "lift"}
# A fence whose `fence_type` says it is wire is drawn as a plain post-and-rail fence even when
# its `material` is metal: railings (barriers.py draws those for metal, iron and steel) are bars.
WIRE_FENCE_TYPES = {"chain_link", "wire_mesh", "wire", "mesh", "net", "netting"}
# A stone edge round water. OpenStreetMap has no agreed tag for the masonry edge
# of a pond, so this repository uses `edge=kerb` on the water area (set through
# a file in fixes/), with `edge:width` and `edge:height` in metres.
EDGE_WIDTH, EDGE_HEIGHT = 0.4, 0.15
EDGE_MATERIAL = "granite"

COMPASS = {
    name: index * 22.5
    for index, name in enumerate(
        ["n", "nne", "ne", "ene", "e", "ese", "se", "sse", "s", "ssw", "sw", "wsw", "w", "wnw", "nw", "nnw"]
    )
}

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

# Roofs (see roofs.py). `roof:shape` values that are drawn as a roof, and the shape roofs.py builds for each; `flat`,
# and any value not listed here, keeps the flat slab. A value with several (`hipped;pyramidal`) takes its first one
# that is listed. `pitched` is OpenStreetMap's word for a roof of two slopes with no more said; `cone` is drawn as a
# pyramid over the footprint (a conical roof on a round tower has a many-sided footprint, so it comes out smooth).
ROOF_SHAPES = {
    "gabled": "gable", "gable": "gable", "pitched": "gable", "hipped": "hip", "hip": "hip",
    "half-hipped": "half_hip", "half_hipped": "half_hip", "pyramidal": "pyramid", "cone": "pyramid",
    "skillion": "skillion", "gambrel": "gambrel", "mansard": "mansard",
    # Mappers use quadruple_saltbox (four slopes round a raised middle) for the mansard roofs of rowhouses.
    "quadruple_saltbox": "mansard", "round": "barrel", "dome": "dome",
}
# `building:levels` counts storeys below the roof; a roof's `roof:levels` are the storeys inside it, about this tall each.
ROOF_LEVEL_HEIGHT = 2.5
# A building known only by `building:levels` is given its levels x LEVEL_HEIGHT plus this much for a parapet or a
# flat roof's thickness (see building_height). A pitched roof stands on the walls without it.
ROOF_ALLOWANCE = 1.0
# What a roof is covered in, by `roof:material`; the colour is used when there is no `roof:colour`.
ROOF_MATERIAL_COLOURS = {
    "slate": "#4a535c", "copper": "#5d9a85", "roof_tiles": "#a0523a", "tile": "#a0523a", "tiles": "#a0523a",
    "clay_tiles": "#a0523a", "metal": "#8b9095", "steel": "#8b9095", "zinc": "#8b9095", "tin": "#8b9095",
    "aluminium": "#9a9fa4", "metal_sheet": "#8b9095", "asphalt": "#3e3f42", "shingle": "#3e3f42",
    "asphalt_shingle": "#3e3f42", "tar_paper": "#3e3f42", "bitumen": "#3e3f42", "roofing_felt": "#3e3f42",
    "concrete": "#8f8f8b", "gravel": "#7d7a74", "stone": "#9a948a", "glass": "#86a9ba", "gold": "#d3a22b",
    "grass": "#6b8f4a", "green_roof": "#6b8f4a", "wood": "#6e5742", "wood_shingle": "#6e5742", "thatch": "#8a7550",
    "plastic": "#7b7e80", "fibre_cement": "#7b7e80", "eternit": "#7b7e80",
}
# A pitched roof with no colour or material is slate on a church, a public building or a mansard, and on brick
# generally, and dark asphalt shingle on a house.
SLATE_ROOFS = ["#46505a", "#3f4a54", "#525c66", "#4a525c"]
SHINGLE_ROOFS = ["#414143", "#4a4849", "#3b3c3f", "#53504d"]
SLATE_BUILDINGS = {
    "church", "cathedral", "chapel", "synagogue", "mosque", "temple", "civic", "public", "government", "school",
    "university", "college", "hospital", "train_station",
}
SHINGLE_BUILDINGS = {"house", "detached", "semidetached_house", "terrace", "residential", "apartments", "dormitory", "garage", "garages", "shed", "hut"}
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
class RoofTags:
    """What the `roof:*` tags say about a roof of a shape we draw (see roof_tags)."""

    shape: str  # a value of ROOF_SHAPES
    height: float | None = None  # `roof:height`: the roof's own height, metres
    levels: float | None = None  # `roof:levels`: storeys inside the roof
    direction: float | None = None  # `roof:direction`: degrees clockwise from north the roof slopes down toward
    orientation: str | None = None  # `roof:orientation`: "along" or "across" the footprint's long side (the ridge)


@dataclass
class Building:
    id: str
    polygon: Polygon
    height: float
    min_height: float
    height_source: str  # "height", "levels", "default", "override", "town" (the Town of Brookline's footprints) or "city" (the City of Boston's survey)
    wall_colour: str
    roof_colour: str
    tags: dict
    part: bool = False
    # For a part: the building it belongs to, whose own outline is not drawn.
    part_of: str | None = None
    part_of_name: str | None = None
    roof: RoofTags | None = None  # set where the tags describe a pitched roof; None is a flat slab


@dataclass
class Way:
    id: str
    line: LineString
    kind: str
    surface: str  # "road", "path" or "rail"
    width: float
    name: str | None
    tags: dict = field(default_factory=dict)
    cover: str = "paved"  # for paths: what the surface is made of (see PATH_COVERS)
    bridge: bool = False


@dataclass
class Area:
    id: str
    polygon: Polygon
    tags: dict = field(default_factory=dict)
    cover: str = ""  # which kind of path or green surface this is; picks its colour


@dataclass
class Barrier:
    """A fence, wall, hedge, chain, kerb, row of bollards or gate along a line (a gate's line is its opening)."""

    id: str
    line: LineString
    kind: str  # one of the kinds in BARRIERS (the first item of each entry)
    height: float  # metres
    width: float  # metres across; 0 for a row of bollards
    colour: str | None = None
    material: str | None = None
    tags: dict = field(default_factory=dict)


@dataclass
class Object:
    """Something standing at a point: a bench, a lamp post, a statue, a fountain."""

    id: str
    x: float
    y: float
    kind: str  # one of furniture.KINDS
    name: str | None = None
    direction: float | None = None  # degrees clockwise from north that it faces; None where the data does not say
    height: float | None = None  # metres, where the data states it
    size: float | None = None  # length of a bench or width of a fountain basin or of an animal statue, where stated; capacity of a bicycle rack
    colour: str | None = None
    material: str | None = None
    # bench "backless"; lamp "arm"/"post"; fountain "nozzle"; statue "equestrian"/"animal"/"ground"; signal "post"/"mast"/
    # "pedestrian"; post_box "ups"; bicycle_parking the `bicycle_parking` tag; bus_stop "shelter"/"shelter+bench" (see streetfurniture.py)
    variant: str | None = None
    tags: dict = field(default_factory=dict)


@dataclass
class Point:
    id: str
    x: float
    y: float
    name: str | None = None
    kind: str | None = None
    tags: dict = field(default_factory=dict)


@dataclass
class Scene:
    buildings: list[Building] = field(default_factory=list)
    ways: list[Way] = field(default_factory=list)
    areas: dict[str, list[Area]] = field(
        default_factory=lambda: {"road": [], "path": [], "water": [], "bed": [], "green": []}
    )
    objects: list[Object] = field(default_factory=list)
    barriers: list[Barrier] = field(default_factory=list)
    coastlines: list[LineString] = field(default_factory=list)
    trees: list[Point] = field(default_factory=list)
    signals: list[Point] = field(default_factory=list)
    roadnodes: list[Point] = field(default_factory=list)  # `highway=crossing` and `highway=stop` nodes: paint on the road (markings.py)
    pois: list[Point] = field(default_factory=list)
    outlines: list[Building] = field(default_factory=list)  # building outlines not drawn because they contain parts (stadium.py reads a stadium's)


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
            height = levels * LEVEL_HEIGHT + ROOF_ALLOWANCE
            source = "levels"
        else:
            height = SHELTER_HEIGHT if tags.get("amenity") == "shelter" else DEFAULT_HEIGHTS.get(kind, FALLBACK_HEIGHT)
            source = "default"
    if source == "height" and tags.get("height:source") == HEIGHT_FROM_TOWN:
        source = "town"
    elif source == "height" and tags.get("height:source") == HEIGHT_FROM_CITY:
        source = "city"
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


def roof_tags(tags: dict) -> RoofTags | None:
    """The roof of a building, when its `roof:shape` is one we draw as a pitched roof (ROOF_SHAPES); else None.

    A `building=roof` is not a building with a roof but a roof alone (pavilions.py draws it on posts, or it is a slab).
    """
    if tags.get("building") == "roof":
        return None
    shape = next((ROOF_SHAPES[w.strip().lower()] for w in str(tags.get("roof:shape", "")).split(";") if w.strip().lower() in ROOF_SHAPES), None)
    if shape is None:
        return None
    height, levels = parse_length(tags.get("roof:height")), parse_number(tags.get("roof:levels"))
    orientation = str(tags.get("roof:orientation", "")).strip().lower()
    return RoofTags(
        shape, height if height and height > 0 else None, levels if levels and levels > 0 else None,
        parse_direction(tags.get("roof:direction")), orientation if orientation in ("along", "across") else None,
    )


def _default_roof_colour(tags: dict, wall: str, seed: int) -> str:
    """A roof with no `roof:colour` or `roof:material`. A flat roof is one of the dark greys in ROOFS; a pitched roof
    is slate or asphalt shingle (see SLATE_BUILDINGS)."""
    pitched = roof_tags(tags)
    if pitched is None:
        return _pick(ROOFS, seed >> 16)
    kind = tags.get("building") or tags.get("building:part") or "yes"
    red, green = (int(wall[i : i + 2], 16) for i in (1, 3))
    brick = red > 1.4 * green
    slate = kind in SLATE_BUILDINGS or tags.get("amenity") == "place_of_worship" or pitched.shape == "mansard"
    return _pick(SLATE_ROOFS if slate or (brick and kind not in SHINGLE_BUILDINGS) else SHINGLE_ROOFS, seed >> 16)


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
    material = ROOF_MATERIAL_COLOURS.get(str(tags.get("roof:material", "")).split(";")[0].strip().lower())
    roof = parse_colour(tags.get("roof:colour")) or material or _default_roof_colour(tags, wall, seed)
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


def path_cover(tags: dict) -> str:
    return PATH_COVERS.get(tags.get("surface", ""), "wood" if tags.get("man_made") == "pier" else "paved")


def green_cover(tags: dict) -> tuple[str, str]:
    """(surface, cover) for a park or planting area: which layer it joins and how it looks."""
    leisure, landuse, natural = tags.get("leisure"), tags.get("landuse"), tags.get("natural")
    surface = tags.get("surface", "")
    if landuse == "flowerbed" or (leisure == "garden" and tags.get("garden:type") == "flowerbed"):
        return "bed", "flowers"
    if leisure == "playground" and surface in PLAYGROUND_PAVING:
        return "green", "playground_hard"  # a paved play area is grey paving, not the dark green of a court
    if leisure == "playground" and surface == "sand":
        return "green", "playground_sand"
    if leisure in ("pitch", "playground", "dog_park"):
        # A pitch for a game that is played on a hard court, with no surface stated, is a hard court: a mapped tennis or basketball
        # court is almost never grass (USGS NAIP, 7 July 2023: painted green or red hard courts all over Boston).
        if leisure == "pitch" and not surface and (sports := {s.strip() for s in str(tags.get("sport", "")).split(";")}) and sports <= HARD_COURT_SPORTS:
            return "green", "court"
        if surface == "artificial_turf":
            return "green", "turf"
        if surface in HARD_COURTS:
            return "green", "court"
        if surface in BARE_GROUND:
            return "green", "dirt"
        if leisure == "playground" and surface != "grass":
            return "green", "playground"  # a soft play surface unless the data says otherwise
        return "green", "pitch" if leisure == "pitch" else "grass"
    if landuse in ("grass", "meadow") or natural == "grassland":
        return "green", "meadow" if landuse == "meadow" else "grass"
    if natural == "wood" or landuse == "forest":
        return "green", "wood"
    if natural == "scrub":
        return "green", "scrub"
    if leisure == "garden":
        return "green", "garden"
    # Parks, commons, cemeteries and the like: the lawn everything else sits on.
    return "green", "lawn"


def water_cover(tags: dict, area: float) -> str:
    """How a body of water looks: "pond" for a pond, lagoon, basin, pool or any small water of no stated kind
    (dark olive: it reflects the trees round it), "open" for rivers, canals, docks, the harbour and the sea."""
    kind = tags.get("water", "")
    if kind in POND_WATER:
        return "pond"
    if kind in OPEN_WATER or tags.get("waterway") in ("dock", "riverbank"):
        return "open"
    return "pond" if area < SMALL_POND else "open"


def bridge_rise(length: float) -> float:
    """How far a footbridge of this length arches above its ends, in metres."""
    low, high = BRIDGE_RISE_LIMITS
    return min(high, max(low, length * BRIDGE_RISE))


def parse_direction(value) -> float | None:
    """An OSM direction ("165", "NE", "10;190") as degrees clockwise from north."""
    if value is None:
        return None
    first = str(value).split(";")[0].strip().lower()
    if first in COMPASS:
        return COMPASS[first]
    number = parse_number(first)
    return number % 360 if number is not None else None


# How a memorial or artwork is drawn, by its `memorial` or `artwork_type` tag.
MEMORIAL_FORMS = {
    "statue": "statue", "bust": "statue", "plaque": "plaque", "blue_plaque": "plaque",
    "obelisk": "obelisk", "column": "column", "stele": "stele", "stone": "stele", "tablet": "stele",
    "war_memorial": "monument", "lantern": "lantern",
}


def flies_us_flag(tags: dict) -> bool:
    """Whether a flagpole carries the national flag of the United States: Wikidata's Q42537 in `flag:wikidata`, or
    the name in `flag:name`, or `country=US` on a `flag:type` pole. Other flags are not drawn (a pole stays bare)."""
    return (
        tags.get("flag:wikidata") == "Q42537"
        or str(tags.get("flag:name", "")).split(";")[0].strip().lower() == "united states"
        or (str(tags.get("country", "")).strip().lower() == "us" and "flag:type" in tags)
    )


def _object(ident: str, tags: dict, x: float, y: float) -> Object | None:
    """Interpret a point as park or street furniture, or as a monument."""
    if _is_underground(tags) or tags.get("location") == "indoor":
        return None
    kind = variant = size = None
    height = parse_length(tags.get("height"))
    colour = parse_colour(tags.get("colour"))
    if tags.get("amenity") == "fountain":
        kind, variant = "fountain", "nozzle" if tags.get("fountain") in ("nozzle", "jet_water") else None  # both throw jets
        size = parse_length(tags.get("diameter") or tags.get("width"))
    elif tags.get("historic") in ("monument", "memorial") or tags.get("tourism") == "artwork":
        form = tags.get("memorial") or tags.get("artwork_type") or ""
        if tags.get("historic") == "monument":
            # A monument that is a statue (a rider on a tall pedestal, say) is drawn as one.
            kind = MEMORIAL_FORMS.get(form, "monument")
        else:
            kind = MEMORIAL_FORMS.get(form, "sculpture" if "tourism" in tags or form else "stele")
        if kind == "plaque" and height is None and tags.get("support") in ("wall", "fence", "building"):
            return None  # fixed to something else, with nothing of its own to draw
        # What kind of figure: `statue=equestrian` (a rider on a horse; a standard OpenStreetMap tag), `animal=*` (a
        # duck, a turtle: drawn low and sized by `length`), `support=ground` (no pedestal). Else a person on a pedestal.
        if tags.get("statue") == "equestrian" and kind != "plaque":
            kind, variant = "statue", "equestrian"
        elif tags.get("animal") and kind in ("statue", "sculpture"):
            kind, variant, size = "statue", "animal", parse_length(tags.get("length"))
        elif tags.get("support") == "ground" and kind == "statue":
            variant = "ground"
        if kind == "column" and flies_us_flag(tags):
            variant = "flag"  # a memorial column that is a flagpole carrying the national flag (a `man_made=flagpole` too)
    elif tags.get("amenity") == "bench":
        kind, variant = "bench", "backless" if tags.get("backrest") == "no" else None
        seats = parse_number(tags.get("seats"))
        size = parse_length(tags.get("length")) or (seats * 0.6 if seats else None)
        height = None  # a bench's height tag is its seat height, which the shape already has
    elif tags.get("highway") == "street_lamp":
        mount = tags.get("lamp_mount", "")
        if mount in ("ground", "wall") or tags.get("support") in ("wall", "ceiling", "ground"):
            return None
        if height is not None and height < 1.5:
            return None  # a light set into the ground
        # A lamp on a bracket arm: a bent mast, or any lamp too tall to be a park lamp (a globe on a post).
        arm = "bent" in mount or "arm" in mount or (height is not None and height > LAMP_ARM_MIN_HEIGHT)
        kind, variant = "lamp", "arm" if arm else "post"
    # Street furniture beside a carriageway (drawn by streetfurniture.py, which also works out which way each faces).
    elif tags.get("highway") == "stop":
        kind = "stop_sign"
    elif tags.get("amenity") == "post_box":
        kind = "post_box"
        owner = f"{tags.get('operator', '')} {tags.get('brand', '')}".lower()
        variant = "ups" if "ups" in owner.split() or "united parcel" in owner else None
        if tags.get("colour") == "blue":
            colour = None  # "blue" is the Postal Service's own blue, the default; the NAMED_COLOURS blue is paler
    elif tags.get("amenity") == "bicycle_parking":
        kind, variant = "bicycle_parking", tags.get("bicycle_parking")
        size = parse_number(tags.get("capacity"))
    elif tags.get("highway") == "bus_stop":
        kind = "bus_stop"
        if tags.get("shelter") == "yes":  # a bench is only drawn inside a shelter
            variant = "shelter+bench" if tags.get("bench") == "yes" else "shelter"
    elif tags.get("amenity") == "waste_basket":
        kind = "waste_basket"
    elif tags.get("amenity") == "drinking_water":
        kind = "drinking_water"
    elif tags.get("man_made") == "flagpole":
        kind = "flagpole"
    elif tags.get("barrier") in ("bollard", "block"):
        kind = tags["barrier"]
    if kind is None:
        found = extras.interpret(tags)  # sports.py, playgrounds.py, stadium.py: kinds of their own (see extras.py)
        if found is None:
            return None
        kind, variant, size = found
    material = (tags.get("material") or "").split(";")[0] or None
    return Object(
        ident, x, y, kind, tags.get("name"), parse_direction(tags.get("direction")),
        height if height and height > 0 else None, size, colour, material, variant, tags,
    )


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


def _add_building(scene: Scene, ident: str, tags: dict, polygons: list[Polygon], patched: set) -> None:
    if tags.get("building") == "no" or tags.get("location") == "underground":
        return
    height, base, source = building_height(tags)
    if {"height": "height", "levels": "building:levels"}.get(source) in patched:
        source = "override"  # the figure comes from a fix in this repository
    wall, roof = building_colours(ident, tags, height)
    for polygon in polygons:
        if polygon.area >= 2.0:
            scene.buildings.append(
                Building(ident, polygon, height, base, source, wall, roof, tags, part="building:part" in tags, roof=roof_tags(tags))
            )


def _sized(tags: dict, key: str, default: float, prefix: str = "") -> float:
    """A length from a tag when it is a sensible figure, else the default."""
    low, high = BARRIER_LIMITS[key]
    value = parse_length(tags.get(prefix + key))
    return value if value is not None and low <= value <= high else default


def _add_barrier(scene: Scene, ident: str, tags: dict, coords: np.ndarray) -> None:
    """A way tagged `barrier` becomes a Barrier along its line; a closed way is its outline, not an area."""
    entry = BARRIERS.get(tags.get("barrier", ""))
    if entry is None or len(coords) < 2 or _is_underground(tags):
        return
    kind, height, width = entry
    if kind == "gate" and (tags.get("gate") or "").strip().lower() in NOT_SWING_GATES:
        return
    material = ((tags.get("material") or tags.get("wall") or "").split(";")[0].strip().lower()) or None
    if kind == "fence" and (tags.get("fence_type") or "").strip().lower() in WIRE_FENCE_TYPES:
        material = "wire"
    scene.barriers.append(
        Barrier(
            ident, LineString(coords), kind, _sized(tags, "height", height), _sized(tags, "width", width) if width else 0.0,
            parse_colour(tags.get("colour")), material, tags,
        )
    )


def _add_edge(scene: Scene, ident: str, tags: dict, polygons: list[Polygon]) -> None:
    """A stone kerb along the outer boundary of each water polygon (not round islands), on the land side."""
    width = _sized(tags, "width", EDGE_WIDTH, "edge:")
    height = _sized(tags, "height", EDGE_HEIGHT, "edge:")
    for polygon in polygons:
        # Offset by half the width so the strip lies on land, from the shoreline outward.
        for ring in polygons_of(Polygon(polygon.exterior).buffer(width / 2, quad_segs=2)):
            scene.barriers.append(
                Barrier(ident, LineString(ring.exterior.coords), "kerb", height, width, None, EDGE_MATERIAL, tags)
            )


def _add_way(scene: Scene, ident: str, tags: dict, coords: np.ndarray) -> None:
    """Classify one OSM way. A way has exactly one role; the first match wins."""
    closed_polygons = _way_polygons(coords)
    if "barrier" in tags and "highway" not in tags and "railway" not in tags:
        _add_barrier(scene, ident, tags, coords)

    def areas(polygons: list[Polygon]) -> list[Area]:
        return [Area(ident, polygon, tags, path_cover(tags)) for polygon in polygons]

    if "highway" in tags:
        kind = tags["highway"]
        if _is_underground(tags) or kind in ("proposed", "construction", "corridor", "elevator", "razed"):
            return
        surface = "path" if kind in PATH_WIDTHS else "road"
        if tags.get("area") == "yes" and closed_polygons:
            scene.areas[surface].extend(areas(closed_polygons))
            return
        width = PATH_WIDTHS[kind] if surface == "path" else _road_width(tags, kind)
        if surface == "path":
            width = parse_length(tags.get("width")) or width
        if len(coords) >= 2:
            cover = path_cover(tags) if surface == "path" else ""
            bridge = tags.get("bridge", "no") != "no"
            scene.ways.append(Way(ident, LineString(coords), kind, surface, width, tags.get("name"), tags, cover, bridge))
    elif "railway" in tags:
        if _is_underground(tags):
            return
        if tags["railway"] == "platform":
            if closed_polygons:
                scene.areas["path"].extend(areas(closed_polygons))
            elif len(coords) >= 2:
                scene.ways.append(
                    Way(ident, LineString(coords), "platform", "path", 3.0, tags.get("name"), tags, path_cover(tags))
                )
        elif len(coords) >= 2:
            scene.ways.append(Way(ident, LineString(coords), tags["railway"], "rail", RAIL_WIDTH, tags.get("name"), tags))
    elif tags.get("natural") == "coastline":
        if len(coords) >= 2:
            scene.coastlines.append(LineString(coords))
    elif tags.get("man_made") in ("pier", "breakwater", "groyne"):
        if closed_polygons:
            scene.areas["path"].extend(areas(closed_polygons))
        elif len(coords) >= 2:
            width = parse_length(tags.get("width")) or PIER_WIDTH
            scene.areas["path"].extend(areas(polygons_of(LineString(coords).buffer(width / 2, cap_style="flat"))))
    else:
        _add_area(scene, ident, tags, closed_polygons)


def _add_area(scene: Scene, ident: str, tags: dict, polygons: list[Polygon]) -> None:
    if not polygons or _is_underground(tags):
        return
    cover = ""
    if tags.get("natural") == "water" or tags.get("waterway") in ("dock", "riverbank"):
        surface = "water"
        if tags.get("edge") == "kerb":
            _add_edge(scene, ident, tags, polygons)
    elif (
        tags.get("leisure") in GREEN_LEISURE
        or tags.get("landuse") in GREEN_LANDUSE
        or tags.get("natural") in GREEN_NATURAL
    ):
        surface, cover = green_cover(tags)
    elif tags.get("man_made") == "pier":
        surface, cover = "path", path_cover(tags)
    else:
        return
    scene.areas[surface].extend(
        Area(ident, polygon, tags, water_cover(tags, polygon.area) if surface == "water" else cover) for polygon in polygons
    )


def element_id(element: dict) -> str:
    """The id used everywhere for a source element: n, w or r plus the OpenStreetMap
    id, or the name a fix in this repository gave the feature it added."""
    return element.get("ident") or f"{element['type'][0]}{element['id']}"


def classify(elements: list[dict]) -> Scene:
    """Interpret source elements: OpenStreetMap data with this repository's fixes applied."""
    scene = Scene()
    for element in elements:
        tags = element.get("tags") or {}
        kind = element["type"]
        ident = element_id(element)
        patched = set(element.get("patched", ()))
        if kind == "node":
            x, y = (float(v) for v in world.to_local(element["lon"], element["lat"]))
            if tags.get("highway") in ("crossing", "stop"):  # kept as well as, not instead of, whatever else the node is
                scene.roadnodes.append(Point(ident, x, y, kind=tags["highway"], tags=tags))
            if tags.get("natural") == "tree":
                scene.trees.append(Point(ident, x, y, tags=tags))
            elif tags.get("highway") == "traffic_signals":
                scene.signals.append(Point(ident, x, y, tags=tags))
            else:
                # A statue is both something to draw and a named place to be near.
                thing = _object(ident, tags, x, y)
                if thing:
                    scene.objects.append(thing)
                if tags.get("name") and _poi_kind(tags):
                    scene.pois.append(Point(ident, x, y, tags["name"], _poi_kind(tags), tags))
        elif kind == "way":
            if not element.get("geometry"):
                continue
            coords = _coords(element["geometry"])
            if "building" in tags or "building:part" in tags:
                _add_building(scene, ident, tags, _way_polygons(coords), patched)
            else:
                _add_way(scene, ident, tags, coords)
        elif kind == "relation" and tags.get("type") == "multipolygon":
            polygons = _relation_polygons(element)
            if "building" in tags or "building:part" in tags:
                _add_building(scene, ident, tags, polygons, patched)
            else:
                _add_area(scene, ident, tags, polygons)
    _drop_outlines_with_parts(scene)
    from . import stadium  # stands, field walls and stadium parts (stadium.py); imported here because it imports this module

    stadium.augment(scene)
    return scene


def _drop_outlines_with_parts(scene: Scene) -> None:
    """Simple 3D Buildings rule: an outline that contains parts is not drawn."""
    parts = [b for b in scene.buildings if b.part]
    if not parts:
        return
    tree = STRtree([p.polygon.representative_point() for p in parts])
    kept = []
    for building in scene.buildings:
        if not building.part:
            inside = tree.query(building.polygon, predicate="contains")
            if len(inside):
                # The outline goes, but its parts remember whose they are.
                scene.outlines.append(building)
                for index in inside:
                    parts[index].part_of = building.id
                    parts[index].part_of_name = building.tags.get("name")
                continue
        kept.append(building)
    scene.buildings = kept

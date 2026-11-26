"""Assemble features into per-tile GLBs, semantic sidecars and a 3D Tiles tileset.

Each tile is built on its own from spatial-index lookups, so build time grows
with the number of tiles rather than the size of the city.
"""

from __future__ import annotations

import json
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import rotate
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon, box
from shapely.ops import polygonize, substring, unary_union
from shapely.strtree import STRtree

from . import world
from .barriers import add_barrier, add_gate, gate_plan, gate_swing_side, join_gates
from .bridges import SUSPENSION, add_bridge, bridge_style, deck_height
from .facing import bench_facings
from .features import MULCH_MIN_PARK, MULCH_PARKS, Building, Scene, bridge_rise, parse_colour, polygons_of
from .furniture import add_object
from .gltf import Material, write_glb
from .groundlook import add_bed_planting, add_lawn, add_mulch, bed_style, mulch_radius
from .markings import add_asphalt, add_paint, plan as plan_markings
from .mesh import MeshBuilder, add_flat, add_skirt, add_walls, colour_bytes
from .pavilions import add_dock_edge, add_pavilion, painted_pier, roof_shape
from .roofs import add_roof, plan_roof
from .streetfurniture import street_objects
from .trees import add_tree, describe

ATTRIBUTION = "© OpenStreetMap contributors (ODbL 1.0)"

ROAD_Z = -0.12  # carriageways sit a curb's height below the pavement
WATER_Z = -0.6
SIDEWALK_MIN = 1.0  # carriageways never come closer than this to a building

BED_HEIGHT = 0.35  # planting in a bed stands this far above the soil
BED_EDGE = 0.25  # bare soil left between a bed's outline and its planting

# Layers drawn in white take their colour from each vertex.
MATERIALS = {
    "ground": Material("#a09d96", 0.95),
    "green": Material("#ffffff", 1.0),
    "beds": Material("#ffffff", 1.0),
    "water": Material("#ffffff", 0.15),
    "roads": Material("#ffffff", 0.92),  # worn asphalt, coloured per vertex (see markings.py)
    "markings": Material("#ffffff", 0.75),  # paint on the roads: crosswalks, centre and lane lines, stop lines
    "paths": Material("#ffffff", 0.9),
    "rail": Material("#6b6259", 1.0),
    "curbs": Material("#8d8c88", 0.8),
    "building_walls": Material("#ffffff", 0.85),
    "building_roofs": Material("#ffffff", 0.9),
    "trees": Material("#ffffff", 1.0),
    "furniture": Material("#ffffff", 0.7),
    "monuments": Material("#ffffff", 0.8),
    "figures": Material("#ffffff", 0.8),  # hand-made statues: drawn smooth-shaded (see assets.py)
    "bridges": Material("#ffffff", 0.85),
    "barriers": Material("#ffffff", 0.8),
}
# The colour of each kind of surface within a layer (see path_cover and
# green_cover in features.py). Where two kinds of path overlap, the one listed
# first is drawn.
#
# How light each is relative to grass was checked against aerial photographs
# (USGS NAIP, July 2023): sunlit asphalt paths and carriageways are about one
# and a half times as bright as lawn, and concrete nearer twice. Weathered
# asphalt is a mid grey, not black. In the viewer, though, an asphalt path drawn
# at the aerial's ratio comes out a washed-out lavender next to the lawn, which
# in photographs taken from the ground it is not (there asphalt is about as light
# as the lawn beside it, and warmer); the colour here is the compromise.
COVERS = {
    "paths": {
        "wood": "#8b7f6b", "brick": "#a8705a", "sett": "#94908a", "metal": "#7f8386",
        "paving_stones": "#b3ada3", "concrete": "#bdb9b0", "gravel": "#b3a688", "sand": "#cdbb93",
        "dirt": "#9c8566", "asphalt": "#928e84", "paved": "#b9b4a8",
    },
    "green": {
        "grass": "#6d8c4c", "lawn": "#6d8c4c", "pitch": "#74964e", "turf": "#4f8a4a", "court": "#3f6f5c",
        "playground": "#b79f7b", "dirt": "#9c8566", "garden": "#62824a", "scrub": "#5f7a48",
        "wood": "#55703f", "meadow": "#7f9455",
    },
    # Ponds and lagoons are dark olive, not blue: in the NAIP aerial the Public Garden
    # lagoon is (44, 57, 49) on a 0 to 255 scale, and in photographs it is a dark green-grey
    # that reflects the trees. The last entry is the default (rivers, the harbour, the sea).
    "water": {"pond": "#3a4d37", "open": "#2e5a6e"},
}
# Greens drawn with patches of lighter and darker, yellower and bluer (see groundlook.py).
# Courts, artificial turf, playground surfaces and bare ground are not.
PATCHY_GREEN = {"grass": 1.0, "lawn": 1.0, "pitch": 0.6, "garden": 0.8, "scrub": 0.8, "wood": 0.8, "meadow": 1.2}
BED_SOIL = "#5a4837"  # mulch: the same as groundlook.MULCH
BRIDGE_STRUCTURE = "#7d8a8c"
BRIDGE_RAILS = "#3a4648"
# Something mapped in the water right beside a raised footbridge (typically a
# lamp) stands on a stone pier that rises this far above the deck.
PIER_REACH = 2.5  # how far from the deck's edge still counts as beside it
PIER_SIZE = 1.3
PIER_ABOVE_DECK = 1.1
PIER_STONE = "#a9a59b"
PIER_LAMP = 2.8  # a lamp on a pier is a lantern on a short post, not a full lamp post
# Objects drawn in the `monuments` layer; every other kind goes in `furniture`.
MONUMENT_KINDS = {"plaque", "stele", "obelisk", "column", "statue", "sculpture", "monument", "fountain", "lantern"}
MIN_AREA = 0.05  # square metres; smaller slivers are dropped
# A fence, wall or hedge is opened where a path crosses it, so nobody is walled in
# by one mapped straight over a gateway, and cut where it lies in a carriageway.
# This is the margin left each side of such a gap; pieces shorter than the
# minimum are dropped.
BARRIER_GAP = 0.4
BARRIER_INSIDE = 0.3  # a barrier this far inside a paved area is cut; one along its edge is kept
PATH_CLEAR = 0.5  # no barrier is drawn this close to a path's centre line
BARRIER_MIN_PIECE = 0.5


class PolygonIndex:
    """A bag of polygons with fast lookup by box. Each may carry a label (its cover, or its feature)."""

    def __init__(self, polygons: list[Polygon], labels: list | None = None):
        self.polygons = polygons
        self.labels = labels if labels is not None else [""] * len(polygons)
        self.tree = STRtree(polygons) if polygons else None

    def near(self, area) -> list[Polygon]:
        return [polygon for polygon, _ in self.labelled(area)]

    def labelled(self, area) -> list[tuple[Polygon, object]]:
        if self.tree is None:
            return []
        return [(self.polygons[i], self.labels[i]) for i in sorted(self.tree.query(area, predicate="intersects"))]

    def clip(self, area: Polygon):
        """Union of everything inside `area`, cut to its outline."""
        return unary_union([p.intersection(area) for p in self.near(area)])


def sea_polygons(coastlines: list, extent: Polygon) -> list[Polygon]:
    """Sea areas inside `extent`, from OSM coastline ways (land lies to their left)."""
    clipped = [line.intersection(extent) for line in coastlines]
    clipped = [g for g in clipped if not g.is_empty]
    if not clipped:
        return []
    pieces = list(polygonize(unary_union([extent.boundary, *clipped])))
    if len(pieces) < 2:
        return []
    tree = STRtree(pieces)
    votes = np.zeros(len(pieces))
    for line in coastlines:
        coords = np.asarray(line.coords)[:, :2]
        a, b = coords[:-1], coords[1:]
        delta = b - a
        length = np.hypot(delta[:, 0], delta[:, 1])
        keep = length > 0.5
        if not keep.any():
            continue
        middle, delta, length = ((a + b) / 2)[keep], delta[keep], length[keep]
        left = np.column_stack([-delta[:, 1], delta[:, 0]]) / length[:, None] * 0.3
        for probes, vote in ((middle + left, -1.0), (middle - left, 1.0)):
            hits = tree.query(shapely.points(probes), predicate="within")
            np.add.at(votes, hits[1], vote * length[hits[0]])
    return [piece for piece, vote in zip(pieces, votes) if vote > 0]


@dataclass
class Context:
    layers: dict[str, PolygonIndex]
    footprints: PolygonIndex
    buildings: dict[tuple[int, int], list[Building]]
    trees: dict[tuple[int, int], list]
    objects: dict[tuple[int, int], list]
    decks: dict[tuple[int, int], list]  # (way, raised line) of each footbridge, by the tile of its middle
    named: list[tuple]  # (area, kind, whole size in square metres) for every named park-like area
    scene: Scene
    ways_tree: STRtree
    barriers: dict[tuple[int, int], list] = field(default_factory=dict)  # (barrier, its line cut to the tile) by tile
    path_areas: PolygonIndex | None = None  # paths drawn as areas (docks, piers, plazas)
    deck_strips: PolygonIndex | None = None  # the footprint of every raised footbridge deck
    facing: dict[str, float] = field(default_factory=dict)  # direction of each bench that states none, by id
    towers: dict[str, list] = field(default_factory=dict)  # pylons of each suspension footbridge, by way id (see _towers)
    docks: dict[tuple[int, int], list] = field(default_factory=dict)  # painted pier areas, by the tile of a point inside each
    markings: object = None  # every piece of road paint in the district (see markings.plan)


ROAD_BRIDGE_REACH = 6.0  # a footway this close to a carriageway is part of a road bridge
BRIDGE_LANDING = 0.3  # a raised deck starts and ends this far back from the water's edge


def _over_water(line, water: PolygonIndex):
    """The stretch of a bridge way that is raised: from just before the water to
    just after it. What is left at each end lies on the bank as ordinary path, so
    paths that run along the bank and cross the bridge way are not blocked."""
    wet = line.intersection(unary_union(water.near(line)))
    reach = [line.project(ShapelyPoint(x, y)) for part in getattr(wet, "geoms", [wet]) for x, y in part.coords]
    start = max(0.0, min(reach) - BRIDGE_LANDING)
    end = min(line.length, max(reach) + BRIDGE_LANDING)
    return substring(line, start, end), [substring(line, 0.0, start), substring(line, end, line.length)]


def _free_standing_over_water(line, water: PolygonIndex, roads: PolygonIndex) -> bool:
    """Whether a bridge way is a footbridge of its own over water, not the
    sidewalk of a road bridge. Only the middle counts: the ends touch the banks."""
    middle = substring(line, 0.15, 0.85, normalized=True)
    if not any(middle.intersects(polygon) for polygon in water.near(middle)):
        return False
    reach = middle.buffer(ROAD_BRIDGE_REACH)
    beside = unary_union([polygon.intersection(reach) for polygon in roads.near(reach)])
    return beside.is_empty or middle.intersection(beside.buffer(ROAD_BRIDGE_REACH)).length < 0.3 * middle.length


def prepare(scene: Scene, tiles: list[tuple[int, int]]) -> Context:
    surfaces: dict[str, list[Polygon]] = {name: [a.polygon for a in areas] for name, areas in scene.areas.items()}
    labels: dict[str, list] = {name: [a.cover for a in areas] for name, areas in scene.areas.items()}
    labels["bed"] = list(scene.areas["bed"])  # a bed is drawn on its own, so it keeps its identity
    surfaces["rail"], labels["rail"] = [], []

    x0 = min(t[0] for t in tiles) * world.TILE_SIZE
    y0 = min(t[1] for t in tiles) * world.TILE_SIZE
    x1 = (max(t[0] for t in tiles) + 1) * world.TILE_SIZE
    y1 = (max(t[1] for t in tiles) + 1) * world.TILE_SIZE
    sea = sea_polygons(scene.coastlines, box(x0, y0, x1, y1))
    surfaces["water"].extend(sea)
    labels["water"].extend(["open"] * len(sea))
    water = PolygonIndex(surfaces["water"])

    def lay(way, line=None) -> None:
        strip = polygons_of((way.line if line is None else line).buffer(way.width / 2, quad_segs=3))
        surfaces[way.surface].extend(strip)
        labels[way.surface].extend([way.cover] * len(strip))

    candidates = []
    for way in scene.ways:
        footbridge = way.bridge and way.surface == "path" and way.kind != "steps" and way.line.length >= 4
        if footbridge and way.tags.get("footway") not in ("sidewalk", "crossing"):
            candidates.append(way)
        else:
            lay(way)
    # A footbridge of its own over water becomes a raised deck and leaves the water
    # beneath it alone. Bridges over roads and rail, and the sidewalks of road
    # bridges, stay on the ground until the model has levels.
    roads = PolygonIndex(surfaces["road"])
    decks = defaultdict(list)
    for way in candidates:
        if _free_standing_over_water(way.line, water, roads):
            raised, banks = _over_water(way.line, water)
            middle = raised.interpolate(0.5, normalized=True)
            decks[world.tile_of(middle.x, middle.y)].append((way, raised))
            for bank in banks:
                if bank.geom_type == "LineString" and bank.length > 0.1:
                    lay(way, bank)
        else:
            lay(way)

    buildings = defaultdict(list)
    for building in scene.buildings:
        point = building.polygon.representative_point()
        buildings[world.tile_of(point.x, point.y)].append(building)
    trees = defaultdict(list)
    for tree in scene.trees:
        trees[world.tile_of(tree.x, tree.y)].append(tree)
    objects = defaultdict(list)
    for thing in street_objects(scene):  # the scene's objects, with signals made and street furniture placed (streetfurniture.py)
        objects[world.tile_of(thing.x, thing.y)].append(thing)

    barriers = defaultdict(list)
    for barrier in join_gates(scene.barriers):
        if barrier.kind == "gate":  # a gate is drawn whole, in the tile of its middle
            middle = barrier.line.interpolate(0.5, normalized=True)
            if (tile := world.tile_of(middle.x, middle.y)) in tiles:
                barriers[tile].append((barrier, barrier.line))
            continue
        for tile, piece in _split_by_tile(barrier.line, tiles):
            barriers[tile].append((barrier, piece))
    strips = [polygon for decks in decks.values() for way, line in decks for polygon in polygons_of(line.buffer(way.width / 2, quad_segs=2))]

    sizes: dict[str, float] = defaultdict(float)
    named = [a for a in (*scene.areas["green"], *scene.areas["bed"]) if a.tags.get("name")]
    for area in named:
        sizes[area.id] += area.polygon.area
    kind = lambda tags: tags.get("leisure") or tags.get("landuse") or tags.get("natural") or "area"  # noqa: E731

    return Context(
        layers={name: PolygonIndex(polys, labels[name]) for name, polys in surfaces.items()},
        footprints=PolygonIndex([b.polygon for b in scene.buildings if b.min_height < 1.0]),
        buildings=buildings,
        trees=trees,
        objects=objects,
        decks=decks,
        named=[(area, kind(area.tags), sizes[area.id]) for area in named],
        scene=scene,
        ways_tree=STRtree([w.line for w in scene.ways]),
        barriers=barriers,
        path_areas=PolygonIndex([a.polygon for a in scene.areas["path"]]),
        deck_strips=PolygonIndex(strips),
        facing=bench_facings(scene.objects, scene.ways, [a.polygon for a in scene.areas["path"]], surfaces["water"]),
        towers=_towers(decks, water, scene.objects),
        docks=_docks(scene.areas["path"]),
        markings=plan_markings(scene, roads),
    )


def _lines(geom) -> list:
    """The line strings in any geometry."""
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    return [line for part in getattr(geom, "geoms", []) for line in _lines(part)]


def _split_by_tile(line, tiles: list[tuple[int, int]]) -> list[tuple[tuple[int, int], object]]:
    """The part of a line inside each of the given tiles that it crosses."""
    wanted = set(tiles)
    x0, y0, x1, y1 = line.bounds
    size = world.TILE_SIZE
    out = []
    for tx in range(math.floor(x0 / size), math.floor(x1 / size) + 1):
        for ty in range(math.floor(y0 / size), math.floor(y1 / size) + 1):
            if (tx, ty) not in wanted:
                continue
            piece = line.intersection(box(*world.tile_bounds(tx, ty)))
            if _lines(piece):
                out.append(((tx, ty), piece))
    return out


def _openings(ctx: Context, tile_box: Polygon) -> tuple:
    """(what every barrier in this tile is cut by, the inside of paved areas).

    Carriageways, buildings and the decks of raised footbridges cut a barrier outright. A paved
    area (a plaza, a dock) cuts one only where it runs through the area's inside, not along its edge.
    """
    reach = tile_box.buffer(BARRIER_GAP + 1)

    def union(index, buffer: float):
        return unary_union([polygon.buffer(buffer, quad_segs=2) for polygon in index.near(reach)])

    cut = unary_union([
        union(ctx.layers["road"], BARRIER_GAP), union(ctx.deck_strips, BARRIER_GAP),
        unary_union(ctx.footprints.near(reach)),
    ])
    return cut, union(ctx.path_areas, -BARRIER_INSIDE)


def _path_crossings(ctx: Context, line):
    """Gaps where paths cross a barrier, each as wide as the path plus a margin.

    Only a crossing opens a fence. A path that merely runs beside one does not: fences
    usually stand right at a path's edge, and cutting everything near a path removed
    almost half of all fence length.
    """
    gaps = []
    for index in ctx.ways_tree.query(line, predicate="dwithin", distance=PATH_CLEAR):
        way = ctx.scene.ways[index]
        if way.surface != "path":
            continue
        # The middle of a path is always kept clear, even of a wall mapped along its centre line.
        gaps.append(way.line.buffer(PATH_CLEAR, quad_segs=2))
        crossing = way.line.intersection(line)
        if not crossing.is_empty:
            gaps.append(crossing.buffer(way.width / 2 + BARRIER_GAP, quad_segs=2))
    return unary_union(gaps)


GATE_REACH = 4.0  # round a gate, what stands in the way of its leaves decides which way they swing
GATE_MEETS = 0.3  # a fence or wall whose line comes this close to an end of a gate runs into it
GATE_NEIGHBOURS = ("fence", "wall", "retaining_wall", "hedge")


def _draw_gate(ctx: Context, builder: MeshBuilder, gate, origin: np.ndarray, feature: int) -> None:
    """One gate (see barriers.py). Its leaves swing to the side with less in the way (carriageways,
    paths, water, buildings), then to the side with more lawn; a pier that would stand inside a
    building is left out."""
    reach = gate.line.buffer(GATE_REACH, quad_segs=2)
    near = lambda index: unary_union(index.near(reach))  # noqa: E731
    blocked = unary_union([near(ctx.layers["road"]), near(ctx.layers["path"]), near(ctx.layers["water"]), near(ctx.footprints)])
    line = np.asarray(gate.line.coords)[:, :2]
    side = gate_swing_side(line, gate.height, blocked, near(ctx.layers["green"]))
    plan = gate_plan(line, gate.height)
    walls = unary_union(ctx.footprints.near(reach))
    piers = (True, True) if plan is None else tuple(
        not walls.contains(ShapelyPoint(plan.pier_centre(k))) for k in (0, 1)
    )
    # The piers are at least as tall as any fence or wall that runs into the gate.
    ends = unary_union([ShapelyPoint(line[0]), ShapelyPoint(line[-1])]).buffer(GATE_MEETS)
    fence = max((b.height for b in ctx.scene.barriers if b.kind in GATE_NEIGHBOURS and b.line.intersects(ends)), default=0.0)
    add_gate(builder, line - origin, gate.height, gate.colour, gate.material, feature, side, gate.tags, piers, fence)


def _beside_deck(decks: dict, water: PolygonIndex, thing) -> tuple | None:
    """(bridge, its raised line, distance along it, direction of travel there, side) of something that
    stands over water beside a raised footbridge, or None for things on the ground. The side is +1
    when the thing is left of the bridge going along its line and -1 when it is right."""
    point = ShapelyPoint(thing.x, thing.y)
    for deck_list in decks.values():
        for way, line in deck_list:
            if line.distance(point) > way.width / 2 + PIER_REACH:
                continue
            if not any(polygon.contains(point) for polygon in water.near(point)):
                return None
            along = line.project(point)
            a, b = line.interpolate(max(0.0, along - 1)), line.interpolate(min(line.length, along + 1))
            heading = math.atan2(b.y - a.y, b.x - a.x)
            left = (b.x - a.x) * (thing.y - a.y) - (b.y - a.y) * (thing.x - a.x)
            return way, line, along, heading, 1 if left >= 0 else -1
    return None


def _pier(ctx: Context, thing) -> tuple | None:
    """(bridge, top height, outline) of the pier under something that stands
    over water beside a raised footbridge, or None for things on the ground."""
    found = _beside_deck(ctx.decks, ctx.layers["water"], thing)
    if found is None:
        return None
    way, line, along, heading, _ = found
    top = deck_height(along, line.length, bridge_rise(line.length)) + PIER_ABOVE_DECK
    half = PIER_SIZE / 2
    square = box(thing.x - half, thing.y - half, thing.x + half, thing.y + half)
    return way, top, rotate(square, math.degrees(heading), origin=(thing.x, thing.y))


def _towers(decks: dict, water: PolygonIndex, objects: list) -> dict[str, list]:
    """The pylons of each suspension footbridge, as (distance along its raised line, side, lamp height):
    one for every lamp mapped in the water right beside the deck, where such a bridge has its lamp-topped
    pylons. The bridge draws the pylon and the globe lamp on it; a lamp is not drawn separately."""
    suspension = {key: [(w, l) for w, l in found if bridge_style(w.tags) == SUSPENSION] for key, found in decks.items()}
    suspension = {key: found for key, found in suspension.items() if found}
    towers: dict[str, list] = defaultdict(list)
    for thing in objects if suspension else ():
        found = _beside_deck(suspension, water, thing) if thing.kind == "lamp" else None
        if found:
            way, line, along, _, side = found
            towers[way.id].append((along, side, thing.height))
    return dict(towers)


def _docks(areas: list) -> dict[tuple[int, int], list]:
    """Painted pier areas (see pavilions.painted_pier), each in the tile of a point inside it."""
    found = defaultdict(list)
    for area in areas:
        if painted_pier(area.tags):
            point = area.polygon.representative_point()
            found[world.tile_of(point.x, point.y)].append(area)
    return dict(found)


def _pieces(geom) -> list[Polygon]:
    return [p for p in polygons_of(geom) if p.area >= MIN_AREA]


def _by_cover(geom, labelled: list[tuple[Polygon, str]], order: list[str]) -> list[tuple[Polygon, str]]:
    """Split a layer's ground by cover. Where covers overlap, the earlier in `order` wins."""
    present = [cover for cover in order if any(label == cover for _, label in labelled)]
    if len(present) <= 1:
        return [(p, present[0] if present else order[-1]) for p in _pieces(geom)]
    out, remaining = [], geom
    for cover in present[:-1]:
        part = unary_union([polygon for polygon, label in labelled if label == cover]).intersection(remaining)
        out += [(p, cover) for p in _pieces(part)]
        remaining = remaining.difference(part)
    return out + [(p, present[-1]) for p in _pieces(remaining)]


def _by_size(geom, labelled: list[tuple[Polygon, str]], default: str) -> list[tuple[Polygon, str]]:
    """Split a layer's ground by cover. Where areas overlap, the smaller wins:
    a tennis court inside a lawn inside a park shows as court, then lawn."""
    covers = {label for _, label in labelled}
    if len(covers) <= 1:
        return [(p, next(iter(covers), default)) for p in _pieces(geom)]
    out, remaining = [], geom
    for polygon, cover in sorted(labelled, key=lambda item: item[0].area):
        if remaining.is_empty:
            break
        out += [(p, cover) for p in _pieces(polygon.intersection(remaining))]
        remaining = remaining.difference(polygon)
    return out + [(p, default) for p in _pieces(remaining)]


def _partition(ctx: Context, tile_box: Polygon) -> dict[str, list[tuple[Polygon, str]]]:
    """Split the tile's ground into non-overlapping surfaces, highest priority first.

    Returns, per layer, each piece with its cover (empty for layers of one colour).
    """
    keep_clear = unary_union(ctx.footprints.near(tile_box.buffer(SIDEWALK_MIN + 1))).buffer(SIDEWALK_MIN)
    roads = ctx.layers["road"].clip(tile_box).difference(keep_clear)
    rail = ctx.layers["rail"].clip(tile_box).difference(roads)
    paths = ctx.layers["path"].clip(tile_box).difference(unary_union([roads, rail]))
    taken = unary_union([roads, rail, paths])
    water = ctx.layers["water"].clip(tile_box).difference(taken)
    taken = unary_union([taken, water])
    beds = ctx.layers["bed"].clip(tile_box).difference(taken)
    taken = unary_union([taken, beds])
    green = ctx.layers["green"].clip(tile_box).difference(taken)
    ground = tile_box.difference(unary_union([taken, green]))
    plain = {"roads": roads, "rail": rail, "beds": beds, "ground": ground}
    surfaces = {name: [(p, "") for p in _pieces(g)] for name, g in plain.items()}
    surfaces["water"] = _by_size(water, ctx.layers["water"].labelled(tile_box), "open")
    surfaces["paths"] = _by_cover(paths, ctx.layers["path"].labelled(tile_box), list(COVERS["paths"]))
    surfaces["green"] = _by_size(green, ctx.layers["green"].labelled(tile_box), "lawn")
    return surfaces


def _mulched_lawn(ctx: Context, surfaces: dict, tile_box: Polygon):
    """The lawn in this tile that lies in a large named park, where trees stand in mulch; None if there is none."""
    parks = [
        area.polygon
        for area, kind, size in ctx.named
        if kind in MULCH_PARKS and size >= MULCH_MIN_PARK and area.polygon.intersects(tile_box)
    ]
    lawn = [polygon for polygon, cover in surfaces["green"] if cover in ("lawn", "grass")]
    if not parks or not lawn:
        return None
    ground = unary_union(lawn).intersection(unary_union(parks))
    if ground.is_empty:
        return None
    shapely.prepare(ground)
    return ground


def _round(value: float) -> float:
    return round(float(value), 2)


def _named_areas(ctx: Context, tile_box: Polygon) -> list[dict]:
    """Named parks, gardens, pitches and playgrounds overlapping the tile, cut to it."""
    found: dict[str, dict] = {}
    for area, kind, size in ctx.named:
        if not area.polygon.intersects(tile_box):
            continue
        rings = [
            [[_round(x), _round(y)] for x, y in piece.exterior.simplify(0.5).coords]
            for piece in _pieces(area.polygon.intersection(tile_box))
        ]
        if rings:
            entry = found.setdefault(
                area.id, {"id": area.id, "name": area.tags["name"], "kind": kind, "area": round(size), "rings": []}
            )
            entry["rings"] += rings
    return list(found.values())


def _semantics(ctx: Context, tx: int, ty: int, tile_box: Polygon, objects: list[dict]) -> dict:
    """What is in the tile, as data: streets, places, objects and signals for clients to use."""
    x0, y0, x1, y1 = tile_box.bounds

    def inside(p) -> bool:
        return x0 <= p.x < x1 and y0 <= p.y < y1

    streets = []
    for index in ctx.ways_tree.query(tile_box, predicate="intersects"):
        way = ctx.scene.ways[index]
        if not way.name:
            continue
        clipped = way.line.intersection(tile_box)
        lines = [clipped] if clipped.geom_type == "LineString" else list(getattr(clipped, "geoms", []))
        for line in lines:
            if line.geom_type == "LineString" and line.length > 0.5:
                streets.append(
                    {
                        "id": way.id,
                        "name": way.name,
                        "kind": way.kind,
                        "surface": way.surface,
                        "width": _round(way.width),
                        "path": [[_round(x), _round(y)] for x, y in line.coords],
                    }
                )
    buildings = []
    for building in ctx.buildings.get((tx, ty), []):
        point = building.polygon.representative_point()
        entry = {
            "id": building.id,
            "pos": [_round(point.x), _round(point.y)],
            "height": _round(building.height),
            "height_source": building.height_source,
        }
        if building.tags.get("name") or building.part_of_name:
            entry["name"] = building.tags.get("name") or building.part_of_name
        if building.part_of:
            entry["part_of"] = building.part_of
        buildings.append(entry)
    return {
        "tile": [tx, ty],
        "streets": streets,
        "buildings": buildings,
        "objects": objects,
        "areas": _named_areas(ctx, tile_box),
        "pois": [
            {"id": p.id, "name": p.name, "kind": p.kind, "pos": [_round(p.x), _round(p.y)]}
            for p in ctx.scene.pois
            if inside(p)
        ],
        "signals": [[_round(p.x), _round(p.y)] for p in ctx.scene.signals if inside(p)],
    }


_colour_cache: dict[tuple[str, str], np.ndarray] = {}


def _cover_colour(layer: str, cover: str) -> np.ndarray:
    key = (layer, cover)
    if key not in _colour_cache:
        palette = COVERS[layer]
        _colour_cache[key] = colour_bytes(palette.get(cover) or next(reversed(palette.values())))
    return _colour_cache[key]


def build_tile(ctx: Context, tx: int, ty: int, out_dir: Path) -> dict:
    """Write one tile's .glb and .json; return its entry for the tileset."""
    x0, y0, x1, y1 = bounds = world.tile_bounds(tx, ty)
    tile_box = box(*bounds)
    origin = np.array([x0, y0])
    surfaces = _partition(ctx, tile_box)

    flat = {name: MeshBuilder(name, colours=name in COVERS or name == "roads") for name in ("ground", "green", "water", "roads", "paths", "rail")}
    curbs = MeshBuilder("curbs")
    heights = {"roads": ROAD_Z, "water": WATER_Z}
    for name, pieces in surfaces.items():
        if name == "beds":
            continue
        z = heights.get(name, 0.0)
        for polygon, cover in pieces:
            if name == "green" and cover in PATCHY_GREEN:
                add_lawn(flat[name], polygon, z, origin, COVERS[name][cover], PATCHY_GREEN[cover])
            elif name == "roads":
                add_asphalt(flat[name], polygon, z, origin)
            else:
                add_flat(flat[name], polygon, z, origin, _cover_colour(name, cover) if name in COVERS else None)
            if z < 0 and name != "water":
                add_skirt(curbs, polygon, z, 0.0, origin, bounds)
        if name == "water":
            # One wall round the water's whole outline, not round each kind of water.
            for polygon in _pieces(unary_union([polygon for polygon, _ in pieces])):
                add_skirt(curbs, polygon, z, 0.0, origin, bounds)

    # Paint on the carriageway, a few centimetres above it (see markings.py).
    paint = MeshBuilder("markings", colours=True)
    add_paint(paint, ctx.markings, tile_box, [polygon for polygon, _ in surfaces["roads"]], ROAD_Z, origin)

    # Planted beds: bare soil at the edge, and a low mass of planting in from it.
    beds = MeshBuilder("beds", colours=True)
    bed_ground = unary_union([polygon for polygon, _ in surfaces["beds"]])
    drawn = []
    for outline, bed in ctx.layers["bed"].labelled(tile_box) if surfaces["beds"] else []:
        patch = outline.intersection(bed_ground).difference(unary_union(drawn))
        if patch.is_empty:
            continue
        drawn.append(outline)
        side = colour_bytes(bed_style(bed.id)[0], 0.7)
        # The planting is set in from the whole bed's outline, not the tile's cut of it.
        planted = outline.buffer(-BED_EDGE).intersection(patch)
        for polygon in _pieces(planted):
            add_bed_planting(beds, polygon, BED_HEIGHT, origin, bed.id)
            add_walls(beds, polygon, 0.0, BED_HEIGHT, origin, side)
        # Soil under the whole bed, planting included: the ground must have no holes,
        # or a walker who arrives inside a bed falls through the world.
        for polygon in _pieces(patch):
            add_flat(beds, polygon, 0.0, origin, colour_bytes(BED_SOIL))

    # A ring of mulch round the foot of each tree on the lawn of a large park (see groundlook.py).
    lawn = _mulched_lawn(ctx, surfaces, tile_box)
    if lawn is not None:
        for tree in ctx.trees.get((tx, ty), []):
            if shapely.contains_xy(lawn, tree.x, tree.y) and tree.tags.get("tree_surround") != "none":
                add_mulch(beds, tree.x, tree.y, mulch_radius(tree.tags), origin, lawn)

    # Each building vertex carries the building's index in this tile's data
    # file, so a client can tell which building a ray or a click has hit.
    walls = MeshBuilder("building_walls", uvs=True, colours=True, features=True)
    roofs = MeshBuilder("building_roofs", colours=True, features=True)
    for index, building in enumerate(ctx.buildings.get((tx, ty), [])):
        if roof_shape(building.tags):  # an open roof on posts, not a box (see pavilions.py)
            add_pavilion(walls, roofs, building, origin, feature=index)
            continue
        wall = colour_bytes(building.wall_colour)
        roof = colour_bytes(building.roof_colour)
        # A gabled, hipped, ... roof from the roof tags, else None: a flat slab (see roofs.py).
        pitched = plan_roof(building, [p for p in ctx.footprints.near(building.polygon.buffer(1.0)) if p is not building.polygon]) if building.roof else None
        if pitched is None:
            add_walls(walls, building.polygon, building.min_height, building.height, origin, wall, feature=index, facade=True)
            add_flat(roofs, building.polygon, building.height, origin, roof, feature=index)
        else:
            if pitched.eave - building.min_height > 0.05:  # a spire starts where the part below it ends
                add_walls(walls, building.polygon, building.min_height, pitched.eave, origin, wall, feature=index, facade=True)
            add_roof(walls, roofs, building, pitched, origin, feature=index)
        if building.min_height > 0.5:
            underside = colour_bytes(building.wall_colour, 0.6)
            add_flat(roofs, building.polygon, building.min_height, origin, underside, up=False, feature=index)

    # Everything else that stands in the tile is listed in its data file, and
    # each vertex carries its entry's index, as for buildings.
    objects: list[dict] = []

    def register(ident: str, kind: str, x: float, y: float, name: str | None = None, **more) -> int:
        details = {key: value for key, value in {"name": name, **more}.items() if value is not None}
        objects.append({"id": ident, "kind": kind, **details, "pos": [_round(x), _round(y)]})
        return len(objects) - 1

    # Each tree's size and shape follow its species and trunk (see trees.py).
    trees = MeshBuilder("trees", colours=True, features=True)
    for tree in ctx.trees.get((tx, ty), []):
        seed = zlib.crc32(tree.id.encode())
        shape = describe(tree.tags, seed)
        index = register(tree.id, "tree", tree.x, tree.y, tree.tags.get("name"), species=shape.name, height=round(shape.height, 1))
        trunk = add_tree(trees, shape, tree.x - x0, tree.y - y0, seed, feature=index)
        if trunk > 0:  # the drawn trunk's radius at 1.3 m: the viewer makes it solid (web/src/trunks.ts)
            objects[index]["trunk"] = _round(trunk)

    bridges = MeshBuilder("bridges", colours=True, features=True)
    listed: dict[str, int] = {}

    def bridge_feature(way) -> int:
        if way.id not in listed:
            middle = way.line.interpolate(0.5, normalized=True)
            listed[way.id] = register(way.id, "bridge", middle.x, middle.y, way.name)
        return listed[way.id]

    for way, line in ctx.decks.get((tx, ty), []):
        add_bridge(
            bridges, np.asarray(line.coords)[:, :2] - origin, way.width, rise=bridge_rise(line.length),
            deck_colour=_cover_colour("paths", way.cover), structure_colour=colour_bytes(BRIDGE_STRUCTURE),
            rail_colour=colour_bytes(BRIDGE_RAILS), feature=bridge_feature(way),
            style=bridge_style(way.tags), towers=ctx.towers.get(way.id, ()), water_z=WATER_Z,
        )

    for area in ctx.docks.get((tx, ty), []):  # the painted sides of docks
        add_dock_edge(
            bridges, [polygon for polygon, _ in surfaces["water"]], area.polygon, parse_colour(area.tags.get("colour")),
            WATER_Z, origin, bounds,
            feature=register(area.id, "dock", *area.polygon.representative_point().coords[0], area.tags.get("name")),
        )

    furniture = MeshBuilder("furniture", colours=True, features=True)
    monuments = MeshBuilder("monuments", colours=True, features=True)
    figures = MeshBuilder("figures", colours=True, features=True)
    for thing in ctx.objects.get((tx, ty), []):
        base, height = 0.0, thing.height
        # What the data says, else what the thing would look at (benches), else north.
        direction = thing.direction if thing.direction is not None else ctx.facing.get(thing.id, 0.0)
        pier = _pier(ctx, thing)
        if pier and bridge_style(pier[0].tags) == SUSPENSION:
            # A suspension bridge draws its own pylons and globe lamps (see bridges.py).
            register(thing.id, thing.kind, thing.x, thing.y, thing.name)
            continue
        if pier:
            way, base, outline = pier
            stone = colour_bytes(PIER_STONE)
            add_walls(bridges, outline, WATER_Z, base, origin, stone, feature=bridge_feature(way))
            add_flat(bridges, outline, base, origin, stone, feature=bridge_feature(way))
            if thing.kind == "lamp":
                height = PIER_LAMP
        add_object(
            monuments if thing.kind in MONUMENT_KINDS else furniture, thing.kind, thing.x - x0, thing.y - y0,
            direction=direction, height=height, size=thing.size, colour=thing.colour,
            material=thing.material, variant=thing.variant, seed=zlib.crc32(thing.id.encode()),
            feature=register(thing.id, thing.kind, thing.x, thing.y, thing.name), base=base, figures=figures,
            tags=thing.tags,
        )

    # Fences, walls and the like: each is listed once in this tile however many pieces are left of it.
    barriers = MeshBuilder("barriers", colours=True, features=True)
    wanted = ctx.barriers.get((tx, ty), [])
    if wanted:
        cut, paved = _openings(ctx, tile_box)
        listed_barriers: dict[tuple[str, str], int] = {}
        for barrier, piece in wanted:
            if barrier.kind == "gate":
                middle = piece.interpolate(0.5, normalized=True)
                _draw_gate(ctx, barriers, barrier, origin, register(barrier.id, "gate", middle.x, middle.y, barrier.tags.get("name")))
                continue
            # A kerb may have a path running along it, so paths never open it.
            through = piece.intersection(paved)  # the stretch that crosses a paved area, plus a margin
            left = piece.difference(cut).difference(through.buffer(BARRIER_INSIDE + BARRIER_GAP, quad_segs=2))
            if barrier.kind != "kerb":
                left = left.difference(_path_crossings(ctx, piece))
            pieces = [p for p in _lines(left) if p.length >= BARRIER_MIN_PIECE]
            if not pieces:
                continue
            key = (barrier.id, barrier.kind)
            if key not in listed_barriers:
                middle = max(pieces, key=lambda p: p.length).interpolate(0.5, normalized=True)
                listed_barriers[key] = register(barrier.id, barrier.kind, middle.x, middle.y, barrier.tags.get("name"))
            for p in pieces:
                add_barrier(
                    barriers, barrier.kind, np.asarray(p.coords)[:, :2] - origin, barrier.height, barrier.width,
                    barrier.colour, barrier.material, feature=listed_barriers[key],
                )

    layers = [
        (builder.finish(), MATERIALS[builder.name])
        for builder in (*flat.values(), paint, beds, curbs, walls, roofs, trees, furniture, monuments, figures, bridges, barriers)
        if not builder.empty
    ]
    name = f"{tx}_{ty}"
    write_glb(out_dir / "tiles" / f"{name}.glb", layers, ATTRIBUTION, extras={"tile": [tx, ty]})
    semantics = _semantics(ctx, tx, ty, tile_box, objects)
    (out_dir / "tiles" / f"{name}.json").write_text(json.dumps(semantics, separators=(",", ":")), encoding="utf-8")

    # Bounding box in tile space (z up), from glTF-axis positions (x, up, -north).
    points = np.concatenate([mesh.positions for mesh, _ in layers])
    low = np.array([points[:, 0].min(), -points[:, 2].max(), points[:, 1].min()], dtype=float)
    high = np.array([points[:, 0].max(), -points[:, 2].min(), points[:, 1].max()], dtype=float)
    centre, half = (low + high) / 2, np.maximum((high - low) / 2, 0.01)
    return {
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x0, y0, 0, 1],
        "boundingVolume": {"box": [*centre.round(3).tolist(), half[0].round(3), 0, 0, 0, half[1].round(3), 0, 0, 0, half[2].round(3)]},
        "geometricError": 0,
        "content": {"uri": f"tiles/{name}.glb"},
        "extras": {
            "tile": [tx, ty],
            "semantics": f"tiles/{name}.json",
            "triangles": int(sum(len(mesh.indices) for mesh, _ in layers) // 3),
        },
    }


def write_tileset(out_dir: Path, entries: list[dict], area: dict, source_timestamp: str | None) -> None:
    lows, highs = [], []
    for entry in entries:
        centre = np.array(entry["boundingVolume"]["box"][:3]) + np.array(entry["transform"][12:15])
        half = np.array(entry["boundingVolume"]["box"])[[3, 7, 11]]
        lows.append(centre - half)
        highs.append(centre + half)
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    centre, half = (low + high) / 2, (high - low) / 2
    places = []
    for place in area.get("places", []):
        x, y = (float(v) for v in world.to_local(place["lon"], place["lat"]))
        places.append({**place, "pos": [_round(x), _round(y)]})
    tileset = {
        "asset": {"version": "1.1", "tilesetVersion": source_timestamp or "unknown"},
        "geometricError": 2000,
        "root": {
            "transform": world.ecef_transform(),
            "boundingVolume": {"box": [*centre.round(3).tolist(), half[0].round(3), 0, 0, 0, half[1].round(3), 0, 0, 0, half[2].round(3)]},
            "geometricError": 1000,
            "refine": "ADD",
            "children": entries,
        },
        "extras": {
            "name": area.get("name"),
            "attribution": ATTRIBUTION,
            "source_timestamp": source_timestamp,
            "world": {
                "crs": world.WORLD_CRS,
                "proj4": world.WORLD_PROJ4,
                "origin": [world.ORIGIN_E, world.ORIGIN_N],
                "tile_size": world.TILE_SIZE,
                "axes": "x east, y north, z up, metres; tile .glb files are y-up per glTF",
            },
            "places": places,
        },
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")


def build(scene: Scene, tiles: list[tuple[int, int]], out_dir: Path, area: dict, source_timestamp: str | None) -> dict:
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
    ctx = prepare(scene, tiles)
    entries = []
    for number, (tx, ty) in enumerate(tiles, 1):
        entries.append(build_tile(ctx, tx, ty, out_dir))
        print(f"\r  tile {number}/{len(tiles)}", end="", flush=True)
    print()
    write_tileset(out_dir, entries, area, source_timestamp)
    sources = defaultdict(int)
    for building in scene.buildings:
        sources[building.height_source] += 1
    return {
        "tiles": len(entries),
        "triangles": sum(e["extras"]["triangles"] for e in entries),
        "buildings": len(scene.buildings),
        "height_sources": dict(sources),
        "trees": len(scene.trees),
        "objects": len(scene.objects),
        "bridges": sum(len(decks) for decks in ctx.decks.values()),
        "barriers": len(scene.barriers),
        "streets": len(scene.ways),
        "megabytes": round(sum(f.stat().st_size for f in (out_dir / "tiles").glob("*.glb")) / 1e6, 1),
    }

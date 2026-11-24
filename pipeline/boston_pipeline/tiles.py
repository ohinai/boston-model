"""Assemble features into per-tile GLBs, semantic sidecars and a 3D Tiles tileset.

Each tile is built on its own from spatial-index lookups, so build time grows
with the number of tiles rather than the size of the city.
"""

from __future__ import annotations

import json
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import rotate
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon, box
from shapely.ops import polygonize, substring, unary_union
from shapely.strtree import STRtree

from . import world
from .bridges import add_bridge, deck_height
from .features import Building, Scene, bridge_rise, polygons_of
from .furniture import add_object
from .gltf import Material, write_glb
from .mesh import MeshBuilder, add_flat, add_skirt, add_walls, colour_bytes
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
    "water": Material("#2e5a6e", 0.15),
    "roads": Material("#3d3f42", 0.9),
    "paths": Material("#ffffff", 0.9),
    "rail": Material("#6b6259", 1.0),
    "curbs": Material("#8d8c88", 0.8),
    "building_walls": Material("#ffffff", 0.85),
    "building_roofs": Material("#ffffff", 0.9),
    "trees": Material("#ffffff", 1.0),
    "furniture": Material("#ffffff", 0.7),
    "monuments": Material("#ffffff", 0.8),
    "bridges": Material("#ffffff", 0.85),
}
# The colour of each kind of surface within a layer (see path_cover and
# green_cover in features.py). Where two kinds of path overlap, the one listed
# first is drawn.
COVERS = {
    "paths": {
        "wood": "#8b7355", "brick": "#96604c", "sett": "#85817b", "metal": "#7f8386",
        "paving_stones": "#aaa49b", "concrete": "#bdb9b0", "gravel": "#b3a688", "sand": "#cdbb93",
        "dirt": "#9c8566", "asphalt": "#6e6f70", "paved": "#b9b4a8",
    },
    "green": {
        "grass": "#6d8c4c", "lawn": "#6d8c4c", "pitch": "#74964e", "turf": "#4f8a4a", "court": "#3f6f5c",
        "playground": "#b79f7b", "dirt": "#9c8566", "garden": "#62824a", "scrub": "#5f7a48",
        "wood": "#55703f", "meadow": "#7f9455",
    },
}
BED_SOIL = "#5a4a3a"
# Planting colours, picked per bed: mostly foliage, some in flower.
BED_PLANTING = ["#5f7f3f", "#6d8a45", "#587a4a", "#66803c", "#8c5a6e", "#a8873a", "#9a4a4a", "#7a6a9a"]
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
MONUMENT_KINDS = {"plaque", "stele", "obelisk", "column", "statue", "sculpture", "monument", "fountain"}
MIN_AREA = 0.05  # square metres; smaller slivers are dropped


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
    labels["water"].extend([""] * len(sea))
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
    for thing in scene.objects:
        objects[world.tile_of(thing.x, thing.y)].append(thing)

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
    )


def _pier(ctx: Context, thing) -> tuple | None:
    """(bridge, top height, outline) of the pier under something that stands
    over water beside a raised footbridge, or None for things on the ground."""
    point = ShapelyPoint(thing.x, thing.y)
    for decks in ctx.decks.values():
        for way, line in decks:
            if line.distance(point) > way.width / 2 + PIER_REACH:
                continue
            if not any(polygon.contains(point) for polygon in ctx.layers["water"].near(point)):
                return None
            length = line.length
            along = line.project(point)
            top = deck_height(along, length, bridge_rise(length)) + PIER_ABOVE_DECK
            a, b = line.interpolate(max(0.0, along - 1)), line.interpolate(min(length, along + 1))
            half = PIER_SIZE / 2
            square = box(thing.x - half, thing.y - half, thing.x + half, thing.y + half)
            return way, top, rotate(square, math.degrees(math.atan2(b.y - a.y, b.x - a.x)), origin=(thing.x, thing.y))
    return None


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
    plain = {"roads": roads, "rail": rail, "water": water, "beds": beds, "ground": ground}
    surfaces = {name: [(p, "") for p in _pieces(g)] for name, g in plain.items()}
    surfaces["paths"] = _by_cover(paths, ctx.layers["path"].labelled(tile_box), list(COVERS["paths"]))
    surfaces["green"] = _by_size(green, ctx.layers["green"].labelled(tile_box), "lawn")
    return surfaces


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

    flat = {name: MeshBuilder(name, colours=name in COVERS) for name in ("ground", "green", "water", "roads", "paths", "rail")}
    curbs = MeshBuilder("curbs")
    heights = {"roads": ROAD_Z, "water": WATER_Z}
    for name, pieces in surfaces.items():
        if name == "beds":
            continue
        z = heights.get(name, 0.0)
        for polygon, cover in pieces:
            add_flat(flat[name], polygon, z, origin, _cover_colour(name, cover) if name in COVERS else None)
            if z < 0:
                add_skirt(curbs, polygon, z, 0.0, origin, bounds)

    # Planted beds: bare soil at the edge, and a low mass of planting in from it.
    beds = MeshBuilder("beds", colours=True)
    bed_ground = unary_union([polygon for polygon, _ in surfaces["beds"]])
    drawn = []
    for outline, bed in ctx.layers["bed"].labelled(tile_box) if surfaces["beds"] else []:
        patch = outline.intersection(bed_ground).difference(unary_union(drawn))
        if patch.is_empty:
            continue
        drawn.append(outline)
        seed = zlib.crc32(bed.id.encode())
        planting = colour_bytes(BED_PLANTING[seed % len(BED_PLANTING)])
        side = colour_bytes(BED_PLANTING[seed % len(BED_PLANTING)], 0.7)
        # The planting is set in from the whole bed's outline, not the tile's cut of it.
        planted = outline.buffer(-BED_EDGE).intersection(patch)
        for polygon in _pieces(planted):
            add_flat(beds, polygon, BED_HEIGHT, origin, planting)
            add_walls(beds, polygon, 0.0, BED_HEIGHT, origin, side)
        for polygon in _pieces(patch.difference(planted)):
            add_flat(beds, polygon, 0.0, origin, colour_bytes(BED_SOIL))

    # Each building vertex carries the building's index in this tile's data
    # file, so a client can tell which building a ray or a click has hit.
    walls = MeshBuilder("building_walls", uvs=True, colours=True, features=True)
    roofs = MeshBuilder("building_roofs", colours=True, features=True)
    for index, building in enumerate(ctx.buildings.get((tx, ty), [])):
        wall = colour_bytes(building.wall_colour)
        roof = colour_bytes(building.roof_colour)
        add_walls(walls, building.polygon, building.min_height, building.height, origin, wall, feature=index)
        add_flat(roofs, building.polygon, building.height, origin, roof, feature=index)
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
        add_tree(trees, shape, tree.x - x0, tree.y - y0, seed, feature=index)

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
        )

    furniture = MeshBuilder("furniture", colours=True, features=True)
    monuments = MeshBuilder("monuments", colours=True, features=True)
    for thing in ctx.objects.get((tx, ty), []):
        base, height = 0.0, thing.height
        pier = _pier(ctx, thing)
        if pier:
            way, base, outline = pier
            stone = colour_bytes(PIER_STONE)
            add_walls(bridges, outline, WATER_Z, base, origin, stone, feature=bridge_feature(way))
            add_flat(bridges, outline, base, origin, stone, feature=bridge_feature(way))
            if thing.kind == "lamp":
                height = PIER_LAMP
        add_object(
            monuments if thing.kind in MONUMENT_KINDS else furniture, thing.kind, thing.x - x0, thing.y - y0,
            direction=thing.direction, height=height, size=thing.size, colour=thing.colour,
            material=thing.material, variant=thing.variant, seed=zlib.crc32(thing.id.encode()),
            feature=register(thing.id, thing.kind, thing.x, thing.y, thing.name), base=base,
        )

    layers = [
        (builder.finish(), MATERIALS[builder.name])
        for builder in (*flat.values(), beds, curbs, walls, roofs, trees, furniture, monuments, bridges)
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
        "streets": len(scene.ways),
        "megabytes": round(sum(f.stat().st_size for f in (out_dir / "tiles").glob("*.glb")) / 1e6, 1),
    }

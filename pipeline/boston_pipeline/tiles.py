"""Assemble features into per-tile GLBs, semantic sidecars and a 3D Tiles tileset.

Each tile is built on its own from spatial-index lookups, so build time grows
with the number of tiles rather than the size of the city.
"""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from . import world
from .features import Building, Scene, polygons_of
from .gltf import Material, write_glb
from .mesh import MeshBuilder, add_flat, add_skirt, add_tree, add_walls, colour_bytes

ATTRIBUTION = "© OpenStreetMap contributors (ODbL 1.0)"

ROAD_Z = -0.12  # carriageways sit a curb's height below the pavement
WATER_Z = -0.6
SIDEWALK_MIN = 1.0  # carriageways never come closer than this to a building

MATERIALS = {
    "ground": Material("#a09d96", 0.95),
    "green": Material("#6d8c4c", 1.0),
    "water": Material("#2e5a6e", 0.15),
    "roads": Material("#3d3f42", 0.9),
    "paths": Material("#b9b4a8", 0.9),
    "rail": Material("#6b6259", 1.0),
    "curbs": Material("#8d8c88", 0.8),
    "building_walls": Material("#ffffff", 0.85),
    "building_roofs": Material("#ffffff", 0.9),
    "trees": Material("#ffffff", 1.0),
}
MIN_AREA = 0.05  # square metres; smaller slivers are dropped


class PolygonIndex:
    """A bag of polygons with fast lookup by box."""

    def __init__(self, polygons: list[Polygon]):
        self.polygons = polygons
        self.tree = STRtree(polygons) if polygons else None

    def near(self, area: Polygon) -> list[Polygon]:
        if self.tree is None:
            return []
        return [self.polygons[i] for i in self.tree.query(area, predicate="intersects")]

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
    scene: Scene
    ways_tree: STRtree


def prepare(scene: Scene, tiles: list[tuple[int, int]]) -> Context:
    surfaces: dict[str, list[Polygon]] = {name: list(polys) for name, polys in scene.areas.items()}
    surfaces["rail"] = []
    for way in scene.ways:
        surfaces[way.surface].extend(polygons_of(way.line.buffer(way.width / 2, quad_segs=3)))

    x0 = min(t[0] for t in tiles) * world.TILE_SIZE
    y0 = min(t[1] for t in tiles) * world.TILE_SIZE
    x1 = (max(t[0] for t in tiles) + 1) * world.TILE_SIZE
    y1 = (max(t[1] for t in tiles) + 1) * world.TILE_SIZE
    surfaces["water"].extend(sea_polygons(scene.coastlines, box(x0, y0, x1, y1)))

    buildings = defaultdict(list)
    for building in scene.buildings:
        point = building.polygon.representative_point()
        buildings[world.tile_of(point.x, point.y)].append(building)
    trees = defaultdict(list)
    for tree in scene.trees:
        trees[world.tile_of(tree.x, tree.y)].append(tree)

    return Context(
        layers={name: PolygonIndex(polys) for name, polys in surfaces.items()},
        footprints=PolygonIndex([b.polygon for b in scene.buildings if b.min_height < 1.0]),
        buildings=buildings,
        trees=trees,
        scene=scene,
        ways_tree=STRtree([w.line for w in scene.ways]),
    )


def _partition(ctx: Context, tile_box: Polygon) -> dict[str, list[Polygon]]:
    """Split the tile's ground into non-overlapping surfaces, highest priority first."""
    keep_clear = unary_union(ctx.footprints.near(tile_box.buffer(SIDEWALK_MIN + 1))).buffer(SIDEWALK_MIN)
    roads = ctx.layers["road"].clip(tile_box).difference(keep_clear)
    rail = ctx.layers["rail"].clip(tile_box).difference(roads)
    paths = ctx.layers["path"].clip(tile_box).difference(unary_union([roads, rail]))
    taken = unary_union([roads, rail, paths])
    water = ctx.layers["water"].clip(tile_box).difference(taken)
    taken = unary_union([taken, water])
    green = ctx.layers["green"].clip(tile_box).difference(taken)
    ground = tile_box.difference(unary_union([taken, green]))
    surfaces = {"roads": roads, "rail": rail, "paths": paths, "water": water, "green": green, "ground": ground}
    return {name: [p for p in polygons_of(g) if p.area >= MIN_AREA] for name, g in surfaces.items()}


def _round(value: float) -> float:
    return round(float(value), 2)


def _semantics(ctx: Context, tx: int, ty: int, tile_box: Polygon) -> dict:
    """What is in the tile, as data: streets, places and signals for clients to use."""
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
        if building.tags.get("name"):
            entry["name"] = building.tags["name"]
        buildings.append(entry)
    return {
        "tile": [tx, ty],
        "streets": streets,
        "buildings": buildings,
        "pois": [
            {"id": p.id, "name": p.name, "kind": p.kind, "pos": [_round(p.x), _round(p.y)]}
            for p in ctx.scene.pois
            if inside(p)
        ],
        "signals": [[_round(p.x), _round(p.y)] for p in ctx.scene.signals if inside(p)],
    }


def build_tile(ctx: Context, tx: int, ty: int, out_dir: Path) -> dict:
    """Write one tile's .glb and .json; return its entry for the tileset."""
    x0, y0, x1, y1 = bounds = world.tile_bounds(tx, ty)
    tile_box = box(*bounds)
    origin = np.array([x0, y0])
    surfaces = _partition(ctx, tile_box)

    flat = {name: MeshBuilder(name) for name in ("ground", "green", "water", "roads", "paths", "rail")}
    curbs = MeshBuilder("curbs")
    heights = {"roads": ROAD_Z, "water": WATER_Z}
    for name, polygons in surfaces.items():
        z = heights.get(name, 0.0)
        for polygon in polygons:
            add_flat(flat[name], polygon, z, origin)
            if z < 0:
                add_skirt(curbs, polygon, z, 0.0, origin, bounds)

    walls = MeshBuilder("building_walls", uvs=True, colours=True)
    roofs = MeshBuilder("building_roofs", colours=True)
    for building in ctx.buildings.get((tx, ty), []):
        wall = colour_bytes(building.wall_colour)
        roof = colour_bytes(building.roof_colour)
        add_walls(walls, building.polygon, building.min_height, building.height, origin, wall)
        add_flat(roofs, building.polygon, building.height, origin, roof)
        if building.min_height > 0.5:
            add_flat(roofs, building.polygon, building.min_height, origin, colour_bytes(building.wall_colour, 0.6), up=False)

    trees = MeshBuilder("trees", colours=True)
    for tree in ctx.trees.get((tx, ty), []):
        add_tree(trees, tree.x - x0, tree.y - y0, zlib.crc32(tree.id.encode()))

    layers = [
        (builder.finish(), MATERIALS[builder.name])
        for builder in (*flat.values(), curbs, walls, roofs, trees)
        if not builder.empty
    ]
    name = f"{tx}_{ty}"
    write_glb(out_dir / "tiles" / f"{name}.glb", layers, ATTRIBUTION, extras={"tile": [tx, ty]})
    semantics = _semantics(ctx, tx, ty, tile_box)
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
        "streets": len(scene.ways),
        "megabytes": round(sum(f.stat().st_size for f in (out_dir / "tiles").glob("*.glb")) / 1e6, 1),
    }

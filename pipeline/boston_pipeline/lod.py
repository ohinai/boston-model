"""Levels of detail: a much simpler model of every tile, drawn where the full tile is not loaded.

The viewer loads full tiles only near the walker, so beyond them there would be nothing: a hard cut-off,
when the camera, the sky and the fog reach for kilometres. For every base tile this module builds a *cell*:
buildings as extruded footprints, the flat surfaces (ground, lawn, water, carriageways) as a few large
polygons, and the largest trees as eight-triangle crowns. Nothing else: no paths, rail, bridges, furniture,
barriers, road paint or monuments. A cell is hidden by the viewer once its full tile is loaded and shown
again if that tile is dropped, so the two are never drawn together (web/src/lod.ts).

Cells are grouped into *super tiles* of SUPER x SUPER base tiles, one file each (`lod1/<I>_<J>.glb`, with
I = floor(tx / SUPER)), because a city has thousands of tiles and thousands of tiny files would not
scale. Inside a file each cell is its own glTF node, `cell_<tx>_<ty>`, positioned relative to the super
tile's south-west corner, with its two meshes (`lod_land`: buildings, trees and the ground, lawn and carriageway
polygons; `lod_water`) as its children. There is no data file, no feature ids and no object lists: a cell
cannot be picked, walked on or asked about. Every layer is vertex-coloured and flat-shaded, and every
surface is at height 0 (the model is flat; the full tiles' 12 cm road and 60 cm water steps are invisible
from where the cells are seen, and would only add things for the depth buffer to fight over).

What is matched to the full tiles, so the swap is not conspicuous:
  * Footprints are simplified (about 1.5 m at most, and less for a small building, keeping the area within a few
    per cent), then extruded to the building's height. A building with a pitched roof (roofs.py) is drawn flat at
    its eaves plus ROOF_SHARE of the roof's rise. The part of a building on a podium has no wall below `min_height`.
  * Walls are painted the colour the facade shader fades to at a distance (`far_wall_colour`, a copy of the
    far tint in web/src/facade.ts), roofs their own colour.
  * The flat surfaces are the full build's own non-overlapping partition (tiles._partition), merged by colour
    and simplified together as a polygonal coverage (shared edges simplified once), so nothing overlaps and nothing
    leaves a gap. Cells meet along the tile edges exactly where the full tiles do.
"""

from __future__ import annotations

import math
import shutil
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from . import world
from .features import MULCH_PARKS, Building, polygons_of
from .gltf import Material, write_glb_cells
from .markings import ASPHALT
from .mesh import FLOOR_HEIGHT, MAX_STOREYS, Mesh, MeshBuilder, add_walls, colour_bytes, srgb_to_linear, triangulate
from .pavilions import roof_shape
from .roofs import _plan as _roof_plan  # the eaves and the top of a pitched roof, without its triangles
from .tiles import ATTRIBUTION, COVERS, MATERIALS as TILE_MATERIALS, Context, _partition
from .trees import TreeShape, describe

LEVEL = 1
SUPER = 4  # base tiles along one side of a super tile: 1024 m
DIRECTORY = f"lod{LEVEL}"

# Buildings.
FOOTPRINT_TOLERANCE = (0.3, 1.5)  # metres: the simplification tolerance is 0.12 of the footprint's width, between these
FOOTPRINT_SHARE = 0.12
FOOTPRINT_AREA = 0.06  # a simplified footprint may differ from the original by this share of its area, else it is kept as it was
HOLE_MIN = 80.0  # square metres: a courtyard smaller than this is filled in
SHED_AREA, SHED_HEIGHT = 60.0, 6.0  # a building smaller than this and lower than this is left out
ROOF_SHARE = 0.4  # a pitched roof is drawn flat at its eaves plus this share of its rise
PAVILION_SLAB = 0.5  # an open roof on posts is a slab this thick
COVERED = 0.95  # a building whose footprint is this much under another one at least as tall adds nothing

# Flat surfaces.
SURFACE_TOLERANCE = 3.0  # metres, for the coverage simplification
TRIANGULATION_ERROR = 1e-4  # share of a polygon's area its triangles may be out by before they are cut again (covering_triangles)
MIN_SURFACE = 12.0  # square metres: a smaller piece of any surface but the ground is folded into the ground
PLAZA_RADIUS = 2.0  # a path is drawn only where it is at least twice this wide; the rest is ground (it is under a pixel from afar)
GROUND = TILE_MATERIALS["ground"].colour
ROAD = ASPHALT
# Lawn in a named park or garden, which in the full model is half hidden by trees the cell does not have: a little greener and
# darker. The largest trees are drawn as crowns (below), so this is only what the smaller ones, and their shade, do.
PARK_LAWN = "#5f7f44"
PARK_MIN = 5_000.0  # square metres: a named park, garden or common at least this big

# Trees.
TREE_MIN_CROWN = 6.0  # metres: a tree with a narrower crown is left out
CROWN_RADIUS = 1.12  # a four-sided crown's corners reach this far beyond the half-width, so it covers as much as a round one
CROWN_BELLY = 0.5  # where a broadleaf crown is widest, as a share of the way from its base to its top
CONE_FORMS = {"conifer", "spire", "clipped", "pine"}
CROWN_SHADE = {"broad": (0.92, 0.94, 0.73), "cone": (1.12, 1.10, 0.96)}  # the average of the full tree's shadow and sunlit colours (trees._palette)
CROWN_LIGHT = (1.12, 1.0, 0.7)  # top, belly and underside of a crown, as multiples of its colour

# A cell has two meshes, not one per kind of thing: the viewer draws a cell as one call per mesh, and a few hundred cells
# are on screen at once. Everything but water is matte (roughness between the tiles' 0.85 for walls and 1.0 for lawn).
MATERIALS = {
    "lod_land": Material("#ffffff", 0.95),  # buildings, trees, ground, lawn and carriageways
    "lod_water": Material("#ffffff", 0.15),
}
LAYER_ORDER = tuple(MATERIALS)


# --- colour ---------------------------------------------------------------------------------------------------


def _smoothstep(low: float, high: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - low) / (high - low)))
    return t * t * (3.0 - 2.0 * t)


_far_cache: dict[tuple[str, int], np.ndarray] = {}
# The sky a window reflects when looked at from far off, in linear light: facade.ts's `fSkyColour` a little above the horizon.
SKY_FAR = np.array([0.24, 0.29, 0.35])


def far_wall_colour(wall: str, floors: int) -> np.ndarray:
    """The linear RGBA bytes a wall of colour `wall` and `floors` storeys is drawn in once its windows are too small to see.

    This is the far tint in web/src/facade.ts (`farColour`, from `farWall` and `share`), worked out the same way:
    brick (reddish) is brick and mortar, a cool tall wall is glass, other tall walls are ribbon windows, the rest is
    stone, and each is darkened by the dark share of its windows. Keep the two in step.
    """
    floors = 0 if floors > MAX_STOREYS else floors  # the shader reads a wall of more than 62 storeys as of unknown height
    key = (wall, floors)
    if key not in _far_cache:
        base = np.array(srgb_to_linear(wall))
        brick = _smoothstep(1.4, 1.8, base[0] / max(base[1], 0.002))
        cool = _smoothstep(1.03, 1.22, base[2] / max(base[0], 0.002)) * (1.0 - brick)
        tint = np.ones(3)
        if cool * _smoothstep(6.5, 9.5, floors) > 0.5:  # a glass curtain wall
            share, far_wall, reflectance = 0.72, base * 0.42, 0.24
            tint = 0.3 + 0.7 * base / max(float(base.mean()), 1e-3)
        elif brick < 0.5 and floors > 7.5:  # ribbon windows between precast panels
            share, far_wall, reflectance = 0.50, base, 0.16
        else:
            share, reflectance = 0.21, 0.10
            if brick > 0.5:
                mortar = 0.35 * base * 1.3 + 0.65 * np.array([0.36, 0.34, 0.31])
                far_wall = 0.78 * base * 0.84 + 0.22 * mortar
            else:
                far_wall = base * 0.98
        dark = share * 0.85
        # The shader also adds the sky the windows reflect, as light the wall gives out; a surface lit as these are returns about
        # as much light as its colour, so it is added to the colour here.
        reflected = SKY_FAR * reflectance * dark * tint
        rgb = np.clip(far_wall * (1.0 - dark) + 0.03 * dark + reflected, 0.0, 1.0)
        _far_cache[key] = np.array([*np.round(rgb * 255), 255], dtype=np.uint8)
    return _far_cache[key]


def _hash_unit(seed: int, k: int) -> float:
    """A repeatable value in [0, 1) for (seed, k)."""
    x = (seed * 0x9E3779B1 + k * 0x85EBCA6B + 0x165667B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    return x / 2**32


# --- triangles ------------------------------------------------------------------------------------------------


def covering_triangles(polygon: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (n, 2), counter-clockwise triangles (m, 3)) that cover `polygon` exactly.

    Ear clipping (mesh.triangulate) is used unless it goes wrong. Simplified surfaces have slivers and vertices a
    hair apart (the coverage simplification leaves a polygon's rings touching and doubling back), and earcut then
    sometimes covers more than the polygon, so every result is checked against the polygon's area, and a polygon it
    fails on is cut by constrained Delaunay triangulation, which is slower but does not.
    """
    vertices, triangles = triangulate(polygon)
    if len(triangles) and abs(_area_of(vertices, triangles) - polygon.area) <= polygon.area * TRIANGULATION_ERROR + 1e-3:
        return vertices, triangles
    try:
        parts = shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
    except GEOSException:
        return vertices, triangles
    corners = np.array([np.asarray(part.exterior.coords)[:3, :2] for part in parts if not part.is_empty])
    if not len(corners):
        return vertices, triangles
    vertices = corners.reshape(-1, 2)
    triangles = np.arange(len(vertices)).reshape(-1, 3)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    clockwise = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
    triangles[clockwise] = triangles[clockwise][:, ::-1]
    return vertices, triangles


def _area_of(vertices: np.ndarray, triangles: np.ndarray) -> float:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum())


def add_flat_checked(builder: MeshBuilder, polygon: Polygon, z: float, origin, colour) -> None:
    """mesh.add_flat, with the triangulation checked (covering_triangles)."""
    vertices, triangles = covering_triangles(polygon)
    if len(triangles):
        builder.add(np.column_stack([vertices - origin, np.full(len(vertices), z)]), (0, 0, 1), triangles, colour=colour)


# --- buildings ------------------------------------------------------------------------------------------------


def top_of(building: Building) -> float:
    """The height a building is drawn to: its roof, or for a pitched roof the eaves plus ROOF_SHARE of the rise."""
    if building.roof is not None:
        try:
            plan = _roof_plan(building, building.roof, building.polygon)
        except (GEOSException, ValueError):
            plan = None
        if plan is not None:
            return plan.eave + ROOF_SHARE * (plan.top - plan.eave)
    return building.height


def simplified_footprint(polygon: Polygon) -> Polygon:
    """The footprint with its small jogs smoothed away and small courtyards filled in; the area stays within a few per cent."""
    width = math.sqrt(polygon.area)
    tolerance = min(FOOTPRINT_TOLERANCE[1], max(FOOTPRINT_TOLERANCE[0], FOOTPRINT_SHARE * width))
    result = polygon
    for scale in (1.0, 0.5, 0.25):
        try:
            candidate = polygon.simplify(tolerance * scale, preserve_topology=True)
        except GEOSException:
            break
        if (
            isinstance(candidate, Polygon) and not candidate.is_empty and candidate.is_valid
            and abs(candidate.area - polygon.area) <= FOOTPRINT_AREA * polygon.area
        ):
            result = candidate
            break
    holes = [ring for ring in result.interiors if Polygon(ring).area >= HOLE_MIN]
    return result if len(holes) == len(result.interiors) else Polygon(result.exterior, holes)


def _covered(polygons: list[Polygon], tops: list[float], bases: list[float]) -> set[int]:
    """Which buildings lie (almost) wholly under another that is at least as tall and starts no higher: they add nothing but
    a second, coincident roof for the depth buffer to fight over."""
    if len(polygons) < 2:
        return set()
    shapes = np.array(polygons, dtype=object)
    first, second = STRtree(shapes).query(shapes, predicate="intersects")
    keep = first != second
    first, second = first[keep], second[keep]
    if not len(first):
        return set()
    try:
        shared = shapely.area(shapely.intersection(shapes[first], shapes[second]))
    except GEOSException:  # a footprint with a self-intersection: ask each pair alone and skip the ones that cannot be answered
        shared = np.zeros(len(first))
        for k, (i, j) in enumerate(zip(first.tolist(), second.tolist())):
            try:
                shared[k] = shapes[i].intersection(shapes[j]).area
            except GEOSException:
                pass
    areas = shapely.area(shapes)
    hidden: set[int] = set()
    for i, j, overlap in zip(first.tolist(), second.tolist(), shared.tolist()):
        # `i` is hidden by `j` when j covers it and is at least as tall; of two equal ones the earlier stays.
        if overlap >= COVERED * areas[i] and tops[j] >= tops[i] - 0.25 and bases[j] <= bases[i] + 0.25:
            if (tops[j], areas[j], -j) > (tops[i], areas[i], -i):
                hidden.add(i)
    return hidden


def add_buildings(builder: MeshBuilder, buildings: list[Building], origin) -> None:
    """Every building of a tile as an extruded, simplified footprint with a flat roof. See the module docstring."""
    kept: list[tuple[Building, float, float]] = []  # (building, bottom, top)
    for building in buildings:
        top = top_of(building)
        bottom = building.min_height
        if roof_shape(building.tags):  # an open roof on posts: a slab, not a block
            bottom = max(bottom, top - PAVILION_SLAB)
        if top - bottom < 0.2 or (building.polygon.area < SHED_AREA and top < SHED_HEIGHT):
            continue
        kept.append((building, bottom, top))
    hidden = _covered([b.polygon for b, _, _ in kept], [t for _, _, t in kept], [z for _, z, _ in kept])
    for index, (building, bottom, top) in enumerate(kept):
        if index in hidden:
            continue
        footprint = simplified_footprint(building.polygon)
        floors = max(1, round((top - bottom) / FLOOR_HEIGHT))
        add_walls(builder, footprint, bottom, top, origin, far_wall_colour(building.wall_colour, floors))
        add_flat_checked(builder, footprint, top, origin, colour_bytes(building.roof_colour))


# --- trees ------------------------------------------------------------------------------------------------------


def add_crown(builder: MeshBuilder, shape: TreeShape, x: float, y: float, seed: int) -> None:
    """A tree as a four-sided crown of at most eight triangles, no trunk: a double pyramid, or a cone for an evergreen. (x, y)
    is relative to the cell's origin."""
    cone = shape.form in CONE_FORMS
    radius = shape.crown_width / 2 * CROWN_RADIUS
    top, base = shape.height, min(shape.crown_base, shape.height * 0.8)
    belly = base if cone else base + CROWN_BELLY * (top - base)
    spin = math.radians(90.0 * _hash_unit(seed, 1))
    angles = spin + np.arange(4) * (math.pi / 2)
    ring = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles), np.full(4, belly)])
    tint = np.array([1 + 0.06 * (2 * _hash_unit(seed, k) - 1) for k in (2, 3, 4)])
    colour = np.array(srgb_to_linear(shape.leaf)) * np.array(CROWN_SHADE["cone" if cone else "broad"]) * tint

    def paint(multiple: float) -> np.ndarray:
        return np.array([*np.round(np.clip(colour * multiple, 0, 1) * 255), 255], dtype=np.uint8)

    # Vertex 0 is the top, 1 to 4 the ring (counter-clockwise from above), 5 the underside of a broadleaf crown.
    positions = [[x, y, top], *ring]
    colours = [paint(CROWN_LIGHT[0]), *[paint(CROWN_LIGHT[1])] * 4]
    sides = [(1 + k, 1 + (k + 1) % 4, 0) for k in range(4)]  # counter-clockwise seen from outside
    if cone:
        faces = sides + [(1, 3, 2), (1, 4, 3)]  # the flat base, facing down
    else:
        positions.append([x, y, base])
        colours.append(paint(CROWN_LIGHT[2]))
        faces = sides + [(1 + (k + 1) % 4, 1 + k, 5) for k in range(4)]
    builder.add(np.array(positions), (0, 0, 1), np.array(faces), colour=np.array(colours))


def add_trees(builder: MeshBuilder, trees: list, origin) -> int:
    """The crowns of the trees of a tile whose crown is wide enough; returns how many."""
    count = 0
    for tree in trees:
        seed = zlib.crc32(tree.id.encode())
        shape = describe(tree.tags, seed)
        if shape.crown_width >= TREE_MIN_CROWN:
            add_crown(builder, shape, tree.x - origin[0], tree.y - origin[1], seed)
            count += 1
    return count


# --- flat surfaces ----------------------------------------------------------------------------------------------


def _surface_pieces(ctx: Context, tile_box: Polygon, surfaces: dict) -> dict[tuple[str, str], list[Polygon]]:
    """The tile's partition regrouped into the cell's layers, each a list of polygons by colour: (layer, colour) -> polygons."""
    keyed: dict[tuple[str, str], list[Polygon]] = defaultdict(list)
    ground = keyed[("lod_ground", GROUND)]
    ground += [polygon for polygon, _ in surfaces["ground"]]
    ground += [polygon for polygon, _ in surfaces["rail"]]
    for polygon, cover in surfaces["paths"]:
        # Only wide paved areas (plazas, wharves, big pedestrian streets) are worth their own colour from afar.
        wide = polygon.buffer(-PLAZA_RADIUS, quad_segs=2).buffer(PLAZA_RADIUS, quad_segs=2).intersection(polygon)
        if wide.is_empty:
            ground.append(polygon)
            continue
        keyed[("lod_ground", COVERS["paths"].get(cover) or COVERS["paths"]["paved"])] += polygons_of(wide)
        ground += polygons_of(polygon.difference(wide))
    for polygon, _ in surfaces["roads"]:
        keyed[("lod_roads", ROAD)].append(polygon)
    for polygon, cover in surfaces["water"]:
        keyed[("lod_water", COVERS["water"].get(cover) or COVERS["water"]["open"])].append(polygon)
    for polygon, _ in surfaces["beds"]:
        keyed[("lod_green", COVERS["green"]["garden"])].append(polygon)

    parks = [
        area.polygon for area, kind, size in ctx.named
        if kind in MULCH_PARKS and size >= PARK_MIN and area.polygon.intersects(tile_box)
    ]
    parkland = unary_union(parks) if parks else None
    if parkland is not None:
        shapely.prepare(parkland)
    for polygon, cover in surfaces["green"]:
        colour = COVERS["green"].get(cover) or COVERS["green"]["lawn"]
        if parkland is not None and cover in ("lawn", "grass") and polygon.intersects(parkland):
            inside = polygon.intersection(parkland)
            keyed[("lod_green", PARK_LAWN)] += polygons_of(inside)
            keyed[("lod_green", colour)] += polygons_of(polygon.difference(parkland))
        else:
            keyed[("lod_green", colour)].append(polygon)
    return keyed


def simplify_coverage(pieces: list[tuple[Polygon, tuple[str, str]]]) -> list[tuple[Polygon, tuple[str, str]]]:
    """Simplify polygons that tile a square together, as a coverage: an edge two of them share is simplified once and stays
    shared, so no gap opens between them and none overlaps another. The square's own outline is kept."""
    if len(pieces) < 2:
        return pieces
    try:
        out = shapely.coverage_simplify(np.array([p for p, _ in pieces], dtype=object), SURFACE_TOLERANCE, simplify_boundary=False)
    except GEOSException:  # an odd coverage: simplify each alone, which can leave slivers but never fails
        out = [p.simplify(SURFACE_TOLERANCE * 0.5, preserve_topology=True) for p, _ in pieces]
    return [(piece, key) for piece, (_, key) in zip(out, pieces) if piece is not None and not piece.is_empty]


def add_surfaces(land: MeshBuilder, water: MeshBuilder, ctx: Context, tile_box: Polygon, origin) -> None:
    """The cell's flat surfaces (ground, lawn, carriageways, and water into `water`), none overlapping another."""
    surfaces = _partition(ctx, tile_box)
    keyed = _surface_pieces(ctx, tile_box, surfaces)
    ground_key = ("lod_ground", GROUND)
    # One polygon list per colour; a small piece of anything but the ground is folded into the ground.
    pieces: list[tuple[Polygon, tuple[str, str]]] = []
    folded = list(keyed[ground_key])
    for key, polygons in keyed.items():
        if key == ground_key or not polygons:
            continue
        for polygon in polygons_of(unary_union(polygons)):
            if polygon.area < MIN_SURFACE:
                folded.append(polygon)
            else:
                pieces.append((polygon, key))
    if folded:
        pieces += [(polygon, ground_key) for polygon in polygons_of(unary_union(folded))]
    for polygon, (layer, colour) in simplify_coverage(pieces):
        for part in polygons_of(polygon):
            if part.area >= 0.5:
                add_flat_checked(water if layer == "lod_water" else land, part, 0.0, origin, colour_bytes(colour))


# --- cells and files --------------------------------------------------------------------------------------------


def build_cell(ctx: Context, tx: int, ty: int) -> list[tuple[Mesh, Material]]:
    """The layers of the simplified cell of base tile (tx, ty), as meshes in the tile's own frame; empty if there is nothing."""
    bounds = world.tile_bounds(tx, ty)
    origin = np.array(bounds[:2])
    builders = {name: MeshBuilder(name, colours=True) for name in LAYER_ORDER}
    add_surfaces(builders["lod_land"], builders["lod_water"], ctx, shapely.box(*bounds), origin)
    add_buildings(builders["lod_land"], ctx.buildings.get((tx, ty), []), origin)
    add_trees(builders["lod_land"], ctx.trees.get((tx, ty), []), origin)
    return [(builders[name].finish(), MATERIALS[name]) for name in LAYER_ORDER if not builders[name].empty]


def super_tile(tx: int, ty: int) -> tuple[int, int]:
    """The super tile a base tile belongs to."""
    return tx // SUPER, ty // SUPER


def build(ctx: Context, tiles: list[tuple[int, int]], out_dir: Path) -> dict:
    """Write the levels of detail of `tiles` into `<out_dir>/lod1/`. Returns the tileset's `extras["lod"]` and the totals."""
    started = time.perf_counter()
    target = out_dir / DIRECTORY
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    size = world.TILE_SIZE
    files: dict[tuple[int, int], list] = defaultdict(list)
    triangles = 0
    for number, (tx, ty) in enumerate(sorted(tiles), 1):
        layers = build_cell(ctx, tx, ty)
        print(f"\r  lod cell {number}/{len(tiles)}", end="", flush=True)
        if not layers:  # nothing in the cell: no node
            continue
        sx, sy = super_tile(tx, ty)
        translation = ((tx - SUPER * sx) * size, 0.0, -((ty - SUPER * sy) * size))  # glTF axes: x east, y up, z south
        files[(sx, sy)].append(((tx, ty), (f"cell_{tx}_{ty}", translation, layers)))
        triangles += sum(len(mesh.indices) // 3 for mesh, _ in layers)
    print()
    entries = []
    for (sx, sy), cells in sorted(files.items()):
        uri = f"{DIRECTORY}/{sx}_{sy}.glb"
        write_glb_cells(
            out_dir / uri, [cell for _, cell in cells], ATTRIBUTION, extras={"lod": LEVEL, "super": [sx, sy]}
        )
        entries.append({"uri": uri, "super": [sx, sy], "cells": [list(key) for key, _ in cells]})
    megabytes = sum(f.stat().st_size for f in target.glob("*.glb")) / 1e6
    seconds = time.perf_counter() - started
    print(f"  levels of detail: {len(entries)} files, {triangles:,} triangles, {megabytes:.1f} MB, {seconds:.0f} s")
    return {
        "extras": {"levels": [{"level": LEVEL, "super": SUPER, "tiles": entries}]},
        "files": len(entries),
        "cells": sum(len(e["cells"]) for e in entries),
        "triangles": triangles,
        "megabytes": round(megabytes, 1),
        "seconds": round(seconds, 1),
    }

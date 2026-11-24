"""Triangle mesh generation from 2D footprints.

Everything here works in z-up metres relative to a tile origin, with
counter-clockwise front faces. MeshBuilder.finish converts to glTF's y-up axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mapbox_earcut as earcut
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

BAY_WIDTH = 3.0  # metres per facade bay (one window column)
FLOOR_HEIGHT = 3.5  # metres per facade floor (one window row)


def srgb_to_linear(hex_colour: str) -> tuple[float, float, float]:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def colour_bytes(hex_colour: str, shade: float = 1.0) -> np.ndarray:
    """A hex sRGB colour as linear RGBA bytes, optionally darkened or lightened."""
    rgb = np.clip(np.array(srgb_to_linear(hex_colour)) * shade, 0, 1)
    return np.array([*np.round(rgb * 255), 255], dtype=np.uint8)


@dataclass
class Mesh:
    """Finished geometry in glTF axes (x east, y up, z south)."""

    name: str
    positions: np.ndarray  # float32 (n, 3)
    normals: np.ndarray  # float32 (n, 3)
    indices: np.ndarray  # uint32 (m,)
    uvs: np.ndarray | None = None  # float32 (n, 2)
    colours: np.ndarray | None = None  # uint8 (n, 4), linear RGBA


class MeshBuilder:
    def __init__(self, name: str, uvs: bool = False, colours: bool = False):
        self.name = name
        self._has_uvs, self._has_colours = uvs, colours
        self._positions: list[np.ndarray] = []
        self._normals: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []
        self._uvs: list[np.ndarray] = []
        self._colours: list[np.ndarray] = []
        self._count = 0

    @property
    def empty(self) -> bool:
        return self._count == 0

    def add(self, positions, normals, indices, uvs=None, colour=None) -> None:
        """Append vertices. `normals` may be one vector; `colour` one RGBA or per vertex."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        count = len(positions)
        if count == 0:
            return
        self._positions.append(positions)
        self._normals.append(np.broadcast_to(np.asarray(normals, dtype=np.float64), (count, 3)))
        self._indices.append(np.asarray(indices, dtype=np.int64).ravel() + self._count)
        if self._has_uvs:
            self._uvs.append(np.zeros((count, 2)) if uvs is None else np.asarray(uvs, dtype=np.float64))
        if self._has_colours:
            self._colours.append(np.broadcast_to(np.asarray(colour, dtype=np.uint8), (count, 4)))
        self._count += count

    def finish(self) -> Mesh:
        positions = np.concatenate(self._positions)
        normals = np.concatenate(self._normals)

        def to_gltf(v: np.ndarray) -> np.ndarray:
            return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype(np.float32)

        return Mesh(
            self.name,
            to_gltf(positions),
            to_gltf(normals),
            np.concatenate(self._indices).astype(np.uint32),
            np.concatenate(self._uvs).astype(np.float32) if self._has_uvs else None,
            np.concatenate(self._colours) if self._has_colours else None,
        )


def triangulate(polygon: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Ear-clip a polygon with holes. Returns (vertices (n, 2), ccw triangles (m, 3))."""
    rings = [np.asarray(polygon.exterior.coords)[:-1]]
    rings += [np.asarray(ring.coords)[:-1] for ring in polygon.interiors]
    vertices = np.concatenate(rings)[:, :2].astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    triangles = earcut.triangulate_float64(vertices, ends).reshape(-1, 3).astype(np.int64)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    triangles[cross < 0] = triangles[cross < 0][:, ::-1]
    return vertices, triangles[np.abs(cross) > 1e-9]


def add_flat(builder: MeshBuilder, polygon: Polygon, z: float, origin, colour=None, up: bool = True) -> None:
    """A horizontal face covering `polygon` at height z, facing up or down."""
    vertices, triangles = triangulate(polygon)
    if not len(triangles):
        return
    positions = np.column_stack([vertices - origin, np.full(len(vertices), z)])
    if not up:
        triangles = triangles[:, ::-1]
    builder.add(positions, (0, 0, 1 if up else -1), triangles, colour=colour)


def add_walls(builder: MeshBuilder, polygon: Polygon, z0: float, z1: float, origin, colour=None) -> None:
    """Outward-facing vertical walls around every ring of `polygon`.

    UVs count facade cells: u runs 0..bays along each wall and v runs 0..floors
    up it, so one UV unit is always one whole window bay by one whole storey.
    Walls too short for a bay get u = 0 throughout (no windows).
    """
    polygon = orient(polygon, 1.0)
    floors = max(1, round((z1 - z0) / FLOOR_HEIGHT))
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2] - origin
        a, b = coords[:-1], coords[1:]
        delta = b - a
        length = np.hypot(delta[:, 0], delta[:, 1])
        keep = length > 1e-3
        a, b, delta, length = a[keep], b[keep], delta[keep], length[keep]
        count = len(a)
        if not count:
            continue
        normal = np.column_stack([delta[:, 1], -delta[:, 0], np.zeros(count)]) / length[:, None]
        bays = np.where(length >= 1.5, np.maximum(1, np.round(length / BAY_WIDTH)), 0.0)
        low, high = np.full(count, z0), np.full(count, z1)
        positions = np.stack(
            [
                np.column_stack([a, low]),
                np.column_stack([b, low]),
                np.column_stack([b, high]),
                np.column_stack([a, high]),
            ],
            axis=1,
        )
        zeros = np.zeros(count)
        tops = np.full(count, float(floors))
        uvs = np.stack(
            [
                np.column_stack([zeros, zeros]),
                np.column_stack([bays, zeros]),
                np.column_stack([bays, tops]),
                np.column_stack([zeros, tops]),
            ],
            axis=1,
        )
        base = np.arange(count)[:, None] * 4
        indices = base + np.array([0, 1, 2, 0, 2, 3])
        builder.add(
            positions.reshape(-1, 3),
            np.repeat(normal, 4, axis=0),
            indices,
            uvs=uvs.reshape(-1, 2),
            colour=colour,
        )


def add_skirt(builder: MeshBuilder, polygon: Polygon, z0: float, z1: float, origin, tile_box) -> None:
    """Inward-facing walls from z0 up to z1 around a sunken area (curbs, banks).

    Edges lying on the tile boundary are skipped: the area continues next door.
    """
    polygon = orient(polygon, 1.0)
    x0, y0, x1, y1 = tile_box
    eps = 1e-3
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        a, b = coords[:-1], coords[1:]
        delta = b - a
        length = np.hypot(delta[:, 0], delta[:, 1])
        on_edge = np.zeros(len(a), dtype=bool)
        for axis, value in ((0, x0), (0, x1), (1, y0), (1, y1)):
            on_edge |= (np.abs(a[:, axis] - value) < eps) & (np.abs(b[:, axis] - value) < eps)
        keep = (length > 1e-3) & ~on_edge
        a, b, delta, length = a[keep] - origin, b[keep] - origin, delta[keep], length[keep]
        count = len(a)
        if not count:
            continue
        normal = np.column_stack([-delta[:, 1], delta[:, 0], np.zeros(count)]) / length[:, None]
        low, high = np.full(count, z0), np.full(count, z1)
        positions = np.stack(
            [
                np.column_stack([b, low]),
                np.column_stack([a, low]),
                np.column_stack([a, high]),
                np.column_stack([b, high]),
            ],
            axis=1,
        )
        indices = np.arange(count)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])
        builder.add(positions.reshape(-1, 3), np.repeat(normal, 4, axis=0), indices)


def _tree_template() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit low-poly tree: (positions, normals, is_canopy), flat shaded, unindexed."""
    triangles: list[tuple] = []
    sides, radius, height = 6, 0.03, 0.42
    ring = [(radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
    for i in range(sides):
        (ax, ay), (bx, by) = ring[i], ring[(i + 1) % sides]
        triangles += [((ax, ay, 0), (bx, by, 0), (bx, by, height)), ((ax, ay, 0), (bx, by, height), (ax, ay, height))]
    trunk_count = len(triangles)

    phi = (1 + 5**0.5) / 2
    corners = np.array(
        [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi),
         (0, -1, -phi), (0, 1, -phi), (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    )
    corners = corners / np.linalg.norm(corners[0]) * np.array([0.36, 0.36, 0.34]) + np.array([0, 0, 0.66])
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    triangles += [tuple(tuple(corners[i]) for i in face) for face in faces]

    positions = np.array(triangles, dtype=np.float64)  # (t, 3, 3)
    normals = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # Make every face point away from the tree's axis, whatever its winding was.
    centroid = positions.mean(axis=1)
    outward = centroid - np.array([0, 0, 1]) * np.where(np.arange(len(positions)) < trunk_count, centroid[:, 2], 0.66)[:, None]
    flip = np.einsum("ij,ij->i", normals, outward) < 0
    positions[flip] = positions[flip][:, ::-1]
    normals[flip] *= -1
    is_canopy = np.repeat(np.arange(len(positions)) >= trunk_count, 3)
    return positions.reshape(-1, 3), np.repeat(normals, 3, axis=0), is_canopy


_TREE = _tree_template()
_TRUNK = colour_bytes("#5b4636")
_CANOPIES = [colour_bytes(c) for c in ("#4f7a3a", "#5d8a45", "#456e36", "#6a9450", "#557f3e")]


def add_tree(builder: MeshBuilder, x: float, y: float, seed: int) -> None:
    """A stylised street tree at (x, y), 7 to 12 m tall depending on `seed`."""
    positions, normals, is_canopy = _TREE
    height = 7.0 + (seed % 50) / 10
    spread = height * (0.9 + ((seed >> 8) % 30) / 100)
    angle = (seed >> 16) % 360 * math.pi / 180
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    scaled = positions * np.array([spread, spread, height])
    colours = np.where(is_canopy[:, None], _CANOPIES[(seed >> 4) % len(_CANOPIES)], _TRUNK)
    builder.add(scaled @ rotation.T + np.array([x, y, 0]), normals @ rotation.T, np.arange(len(positions)), colour=colours)

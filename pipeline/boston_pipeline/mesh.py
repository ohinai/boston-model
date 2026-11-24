"""Triangle mesh generation from 2D footprints.

Everything here works in z-up metres relative to a tile origin, with
counter-clockwise front faces. MeshBuilder.finish converts to glTF's y-up axes.
"""

from __future__ import annotations

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
    features: np.ndarray | None = None  # float32 (n,), index of the feature each vertex belongs to


class MeshBuilder:
    def __init__(self, name: str, uvs: bool = False, colours: bool = False, features: bool = False):
        self.name = name
        self._has_uvs, self._has_colours, self._has_features = uvs, colours, features
        self._features: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._normals: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []
        self._uvs: list[np.ndarray] = []
        self._colours: list[np.ndarray] = []
        self._count = 0

    @property
    def empty(self) -> bool:
        return self._count == 0

    def add(self, positions, normals, indices, uvs=None, colour=None, feature: int = 0) -> None:
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
        if self._has_features:
            self._features.append(np.full(count, feature, dtype=np.float32))
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
            np.concatenate(self._features) if self._has_features else None,
        )


def weld(mesh: Mesh) -> Mesh:
    """Merge vertices that agree in position, colour, UV and feature, so each is
    stored once. Normals are ignored: tiles are written without them."""
    count = len(mesh.positions)
    parts = [np.ascontiguousarray(mesh.positions, dtype=np.float32).view(np.uint8).reshape(count, -1)]
    for extra, kind in ((mesh.uvs, np.float32), (mesh.colours, np.uint8), (mesh.features, np.float32)):
        if extra is not None:
            parts.append(np.ascontiguousarray(extra, dtype=kind).view(np.uint8).reshape(count, -1))
    rows = np.ascontiguousarray(np.hstack(parts))
    keys = rows.view([("", np.void, rows.shape[1])]).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # Keep vertices in the order they first appeared.
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    keep = first[order]

    def pick(array):
        return None if array is None else array[keep]

    return Mesh(
        mesh.name, mesh.positions[keep], mesh.normals[keep], rank[inverse.ravel()][mesh.indices].astype(np.uint32),
        pick(mesh.uvs), pick(mesh.colours), pick(mesh.features),
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


def add_flat(
    builder: MeshBuilder, polygon: Polygon, z: float, origin, colour=None, up: bool = True, feature: int = 0
) -> None:
    """A horizontal face covering `polygon` at height z, facing up or down."""
    vertices, triangles = triangulate(polygon)
    if not len(triangles):
        return
    positions = np.column_stack([vertices - origin, np.full(len(vertices), z)])
    if not up:
        triangles = triangles[:, ::-1]
    builder.add(positions, (0, 0, 1 if up else -1), triangles, colour=colour, feature=feature)


def add_walls(
    builder: MeshBuilder, polygon: Polygon, z0: float, z1: float, origin, colour=None, feature: int = 0
) -> None:
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
            feature=feature,
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

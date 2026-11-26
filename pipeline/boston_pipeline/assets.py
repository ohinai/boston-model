"""Hand-made assets: Blender models placed where the pipeline would otherwise draw a procedural shape.

An asset is `assets/<name>.glb` plus a sidecar `assets/<name>.toml` (see assets/README.md). The file is read with
numpy alone and flattened to what the tile builder wants: world axes (x east, y north, z up), metres, with a
linear RGBA byte colour on every vertex. The viewer keeps only vertex colours (see `materialFor` in
web/src/city.ts), so an asset carries no textures and no materials.

Conventions an asset must follow, all checked when it loads:

- Standard glTF axes (y up), 1 unit = 1 metre. (Blender's exporter does this by default.)
- Its origin is on the ground at the centre of its base, and it faces +y in Blender, which is north at heading 0.
- It stands on z = 0 and is as tall as its sidecar says; it stays within its triangle budget.
- Colours are linear, in the COLOR_0 attribute.

Placing one is `parts.add(*place(asset, x, y, direction, z0, scale))` for furniture's `_Parts`, or the same
arguments to `MeshBuilder.add`. Procedural shapes stay as the fallback: `load` returns None when an asset is
missing, and when the environment variable BOSTON_NO_ASSETS is set (for before/after comparisons).
"""

from __future__ import annotations

import json
import math
import os
import struct
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
DISABLE_VARIABLE = "BOSTON_NO_ASSETS"
REQUIRED = ("file", "height", "max_triangles", "author", "licence", "why")
GROUND_TOLERANCE = 0.05  # metres: how far the lowest vertex may be from z = 0
HEIGHT_TOLERANCE = 0.03  # of the declared height

_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class AssetError(ValueError):
    """An asset file or sidecar that breaks the contract. The message names the asset and what to change."""


@dataclass(frozen=True, eq=False)
class Asset:
    name: str
    positions: np.ndarray  # (n, 3) float64, world axes, metres, origin at the centre of the base on the ground
    normals: np.ndarray  # (n, 3) float64, unit
    indices: np.ndarray  # (m,) int64, counter-clockwise triangles
    colours: np.ndarray  # (n, 4) uint8, linear RGBA
    height: float  # metres, as declared in the sidecar and checked against the geometry
    max_triangles: int

    @property
    def triangles(self) -> int:
        return len(self.indices) // 3


# --- Reading a .glb ---------------------------------------------------------------------------------------------
def _chunks(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 20 or data[:4] != b"glTF":
        raise AssetError("not a binary glTF (.glb) file")
    version, _total = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise AssetError(f"glTF version {version}: only version 2 is read")
    document, binary, offset = None, b"", 12
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<I4s", data, offset)
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"JSON":
            document = json.loads(body)
        elif kind.startswith(b"BIN"):
            binary = body
        offset += 8 + length
    if document is None:
        raise AssetError("no JSON chunk")
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    spec = document["accessors"][index]
    if "sparse" in spec:
        raise AssetError("sparse accessors are not read; export without sparse morph targets")
    if "bufferView" not in spec:
        raise AssetError("an accessor without a buffer view is not read")
    view = document["bufferViews"][spec["bufferView"]]
    dtype = np.dtype(_DTYPES[spec["componentType"]])
    width = _WIDTHS[spec["type"]]
    count = spec["count"]
    start = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + dtype.itemsize * width, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * width), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(count, width)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF stores column-major
    matrix = np.eye(4)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, :3] = rotation * np.array(node.get("scale", (1.0, 1.0, 1.0)))  # T * R * S
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _colours(array: np.ndarray, component: int, normalised: bool) -> np.ndarray:
    """Vertex colours as linear RGBA bytes, whatever the file stores."""
    if component == 5126:
        values = np.clip(array.astype(np.float64), 0.0, 1.0) * 255.0
    elif component in (5121, 5123) and normalised:
        values = array.astype(np.float64) / np.iinfo(array.dtype).max * 255.0
    else:
        raise AssetError("COLOR_0 must be normalised unsigned bytes or shorts, or floats")
    out = np.round(values).astype(np.uint8)
    if out.shape[1] == 3:  # RGB: opaque
        out = np.column_stack([out, np.full(len(out), 255, dtype=np.uint8)])
    return out


def read_glb(path: Path) -> dict[str, np.ndarray]:
    """Every triangle of every mesh in the file's scene, in world axes (x east, y north, z up), with the nodes'
    transforms applied. Returns positions, normals, indices and colours (None if the file has none)."""
    document, binary = _chunks(Path(path).read_bytes())
    if document.get("textures") or document.get("images"):
        raise AssetError("has textures: the viewer keeps only vertex colours, so paint them into COLOR_0")
    scene = document["scenes"][document.get("scene", 0)]
    parts: list[tuple] = []

    def visit(index: int, parent: np.ndarray) -> None:
        node = document["nodes"][index]
        matrix = parent @ _node_matrix(node)
        if "mesh" in node:
            for primitive in document["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise AssetError("only triangle meshes are read")
                attributes = primitive["attributes"]
                positions = _accessor(document, binary, attributes["POSITION"]).astype(np.float64)
                if "indices" in primitive:
                    indices = _accessor(document, binary, primitive["indices"]).reshape(-1).astype(np.int64)
                else:
                    indices = np.arange(len(positions), dtype=np.int64)
                normals = (
                    _accessor(document, binary, attributes["NORMAL"]).astype(np.float64) if "NORMAL" in attributes else None
                )
                colours = None
                if "COLOR_0" in attributes:
                    spec = document["accessors"][attributes["COLOR_0"]]
                    colours = _colours(_accessor(document, binary, attributes["COLOR_0"]), spec["componentType"],
                                       bool(spec.get("normalized")))
                parts.append((positions, normals, indices, colours, matrix))
        for child in node.get("children", []):
            visit(child, matrix)

    for root in scene["nodes"]:
        visit(root, np.eye(4))
    if not parts:
        raise AssetError("no triangle meshes in the file")

    all_positions, all_normals, all_indices, all_colours, base = [], [], [], [], 0
    for positions, normals, indices, colours, matrix in parts:
        linear = matrix[:3, :3]
        world = positions @ linear.T + matrix[:3, 3]
        if normals is None:
            normals = np.tile([0.0, 1.0, 0.0], (len(positions), 1))
        else:
            normals = normals @ np.linalg.inv(linear).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        if np.linalg.det(linear) < 0:  # a mirroring node turns the triangles inside out
            indices = indices.reshape(-1, 3)[:, ::-1].reshape(-1)
        all_positions.append(world)
        all_normals.append(normals)
        all_indices.append(indices + base)
        all_colours.append(colours if colours is not None else np.full((len(positions), 4), 255, dtype=np.uint8))
        base += len(positions)

    def to_world(v: np.ndarray) -> np.ndarray:  # glTF (x, y up, z south) -> world (x east, y north, z up)
        return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])

    return {
        "positions": to_world(np.concatenate(all_positions)),
        "normals": to_world(np.concatenate(all_normals)),
        "indices": np.concatenate(all_indices),
        "colours": np.concatenate(all_colours),
        "has_colours": any(p[3] is not None for p in parts),
    }


# --- The library ------------------------------------------------------------------------------------------------
def available(directory: Path | None = None) -> list[str]:
    """Names of every asset with a sidecar in the library."""
    return sorted(p.stem for p in Path(directory or ASSETS_DIR).glob("*.toml"))


def load(name: str, directory: Path | None = None) -> Asset | None:
    """The named asset, or None when it is not in the library or assets are switched off (BOSTON_NO_ASSETS)."""
    if os.environ.get(DISABLE_VARIABLE):
        return None
    return _load(name, str(directory or ASSETS_DIR))


@lru_cache(maxsize=None)
def _load(name: str, directory: str) -> Asset | None:
    sidecar = Path(directory) / f"{name}.toml"
    if not sidecar.exists():
        return None
    try:
        meta = tomllib.loads(sidecar.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise AssetError(f"{sidecar}: not valid TOML: {error}") from error
    missing = [key for key in REQUIRED if key not in meta]
    if missing:
        raise AssetError(f"{sidecar}: missing {', '.join(missing)}. Required: {', '.join(REQUIRED)}")
    file = Path(directory) / meta["file"]
    if not file.exists():
        raise AssetError(f"{sidecar}: {meta['file']} does not exist")
    height, budget = float(meta["height"]), int(meta["max_triangles"])
    try:
        data = read_glb(file)
    except (AssetError, KeyError, ValueError, IndexError) as error:
        raise AssetError(f"{file.name}: {error}") from error
    if not data["has_colours"]:
        raise AssetError(f"{file.name}: no COLOR_0: paint every vertex (the viewer keeps only vertex colours)")
    positions = data["positions"]
    triangles = len(data["indices"]) // 3
    if triangles > budget:
        raise AssetError(f"{file.name}: {triangles} triangles, over its budget of {budget} (max_triangles in {sidecar.name})")
    low, top = float(positions[:, 2].min()), float(positions[:, 2].max())
    if abs(low) > GROUND_TOLERANCE:
        raise AssetError(f"{file.name}: its lowest point is {low:+.2f} m, not 0: put the origin on the ground at the base")
    if abs((top - low) - height) > HEIGHT_TOLERANCE * height:
        raise AssetError(f"{file.name}: it is {top - low:.2f} m tall but {sidecar.name} says {height:g} m")
    return Asset(name, positions, data["normals"], data["indices"], data["colours"], height, budget)


def place(asset: Asset, x: float, y: float, direction: float = 0.0, z0: float = 0.0, scale: float = 1.0):
    """The asset standing at (x, y) on height z0, facing `direction` (degrees clockwise from north), enlarged by
    `scale`. Returns (positions, normals, indices, colours): exactly what `_Parts.add` takes, and
    `MeshBuilder.add` with `colour=`."""
    t = math.radians(direction)
    c, s = math.cos(t), math.sin(t)
    rotation = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # local (u right, v ahead) -> world
    positions = (asset.positions * scale) @ rotation.T + np.array([x, y, z0])
    return positions, asset.normals @ rotation.T, asset.indices, asset.colours


def scale_to(asset: Asset, height: float) -> float:
    """The `scale` that makes the asset `height` metres tall."""
    return height / asset.height

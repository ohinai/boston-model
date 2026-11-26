"""A minimal binary glTF 2.0 (.glb) writer: one node, mesh and material per layer.

Meshes are written without normals. Every surface in the model is flat-shaded,
and glTF says a client must then shade each triangle by its own face normal.
That lets vertices be shared between faces, which roughly halves the files.

`write_glb` writes a tile: its layers are the scene's nodes. `write_glb_cells` writes a
file of several cells (see lod.py), each a node with the layers as its children.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import Mesh, srgb_to_linear, weld

FLOAT, UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass(frozen=True)
class Material:
    colour: str  # sRGB hex; multiplied by vertex colours where a mesh has them
    roughness: float = 0.9
    metallic: float = 0.0


class _Document:
    """The parts of a glTF file as they are written: one binary buffer, and the lists that describe it."""

    def __init__(self) -> None:
        self.binary = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []
        self.meshes: list[dict] = []
        self.materials: list[dict] = []
        self.nodes: list[dict] = []
        self.extensions: set[str] = set()
        self._shared: dict[tuple[str, Material], int] = {}

    def accessor(self, array: np.ndarray, component: int, kind: str, target: int, **extra) -> int:
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        data = np.ascontiguousarray(array).tobytes()
        self.views.append({"buffer": 0, "byteOffset": len(self.binary), "byteLength": len(data), "target": target})
        self.binary.extend(data)
        self.accessors.append(
            {"bufferView": len(self.views) - 1, "componentType": component, "count": len(array), "type": kind, **extra}
        )
        return len(self.accessors) - 1

    def material(self, name: str, material: Material, shared: bool) -> int:
        """A material for a mesh called `name`. With `shared`, every mesh of that name and material uses one."""
        if shared and (name, material) in self._shared:
            return self._shared[(name, material)]
        self.materials.append(
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*srgb_to_linear(material.colour), 1.0],
                    "metallicFactor": material.metallic,
                    "roughnessFactor": material.roughness,
                },
            }
        )
        self._shared[(name, material)] = len(self.materials) - 1
        return len(self.materials) - 1

    def mesh(self, mesh: Mesh, material: Material, shared_materials: bool = False) -> int:
        """Add a welded mesh and its material; returns the mesh's index."""
        mesh = weld(mesh)
        attributes = {
            "POSITION": self.accessor(
                mesh.positions, FLOAT, "VEC3", ARRAY_BUFFER,
                min=mesh.positions.min(axis=0).tolist(), max=mesh.positions.max(axis=0).tolist(),
            ),
        }
        if mesh.uvs is not None:
            attributes["TEXCOORD_0"] = self.accessor(mesh.uvs, FLOAT, "VEC2", ARRAY_BUFFER)
        if mesh.colours is not None:
            attributes["COLOR_0"] = self.accessor(mesh.colours, UNSIGNED_BYTE, "VEC4", ARRAY_BUFFER, normalized=True)
        primitive_extensions = {}
        if mesh.features is not None:
            # Which feature each vertex belongs to, as an index into the tile's data file.
            attributes["_FEATURE_ID_0"] = self.accessor(mesh.features, FLOAT, "SCALAR", ARRAY_BUFFER)
            count = int(mesh.features.max()) + 1
            primitive_extensions["EXT_mesh_features"] = {"featureIds": [{"featureCount": count, "attribute": 0}]}
            self.extensions.add("EXT_mesh_features")
        if len(mesh.positions) <= 65535:
            indices = self.accessor(mesh.indices.astype(np.uint16), UNSIGNED_SHORT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        else:
            indices = self.accessor(mesh.indices, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        material_index = self.material(mesh.name, material, shared_materials)
        self.meshes.append(
            {
                "name": mesh.name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": indices,
                        "material": material_index,
                        **({"extensions": primitive_extensions} if primitive_extensions else {}),
                    }
                ],
            }
        )
        return len(self.meshes) - 1

    def write(self, path: Path, scene_nodes: list[int], copyright: str, extras: dict | None) -> None:
        # The buffer is declared at its padded length, which is the length of the BIN chunk. (It used to be
        # padded after it was declared, so a layer ending in an odd number of 16-bit indices made them disagree.)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        document = {
            "asset": {"version": "2.0", "generator": "boston-pipeline", "copyright": copyright},
            "scene": 0,
            "scenes": [{"nodes": scene_nodes}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": len(self.binary)}],
        }
        if self.extensions:
            document["extensionsUsed"] = sorted(self.extensions)
        if extras:
            document["scenes"][0]["extras"] = extras

        json_chunk = json.dumps(document, separators=(",", ":")).encode()
        json_chunk += b" " * (-len(json_chunk) % 4)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        total = 12 + 8 + len(json_chunk) + 8 + len(self.binary)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<4sII", b"glTF", 2, total))
            handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
            handle.write(json_chunk)
            handle.write(struct.pack("<I4s", len(self.binary), b"BIN\0"))
            handle.write(self.binary)


def write_glb(path: Path, layers: list[tuple[Mesh, Material]], copyright: str, extras: dict | None = None) -> None:
    document = _Document()
    for mesh, material in layers:
        document.nodes.append({"name": mesh.name, "mesh": document.mesh(mesh, material)})
    document.write(path, list(range(len(document.nodes))), copyright, extras)


def write_glb_cells(
    path: Path,
    cells: list[tuple[str, tuple[float, float, float], list[tuple[Mesh, Material]]]],
    copyright: str,
    extras: dict | None = None,
) -> None:
    """A file of cells: each is a node `name` at `translation` (glTF axes) whose children are its layers.

    A cell's meshes are positioned relative to the cell's own origin, so a client can place or hide a cell as
    a whole. Materials are shared between the cells' layers of one name.
    """
    document = _Document()
    roots = []
    for name, translation, layers in cells:
        children = []
        for mesh, material in layers:
            document.nodes.append({"name": mesh.name, "mesh": document.mesh(mesh, material, shared_materials=True)})
            children.append(len(document.nodes) - 1)
        document.nodes.append({"name": name, "translation": [round(float(v), 3) for v in translation], "children": children})
        roots.append(len(document.nodes) - 1)
    document.write(path, roots, copyright, extras)

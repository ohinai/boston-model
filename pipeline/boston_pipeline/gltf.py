"""A minimal binary glTF 2.0 (.glb) writer: one node, mesh and material per layer."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import Mesh, srgb_to_linear

FLOAT, UNSIGNED_BYTE, UNSIGNED_INT = 5126, 5121, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass(frozen=True)
class Material:
    colour: str  # sRGB hex; multiplied by vertex colours where a mesh has them
    roughness: float = 0.9
    metallic: float = 0.0


def write_glb(path: Path, layers: list[tuple[Mesh, Material]], copyright: str, extras: dict | None = None) -> None:
    binary = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def accessor(array: np.ndarray, component: int, kind: str, target: int, **extra) -> int:
        binary.extend(b"\0" * (-len(binary) % 4))
        data = np.ascontiguousarray(array).tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary.extend(data)
        accessors.append(
            {"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": kind, **extra}
        )
        return len(accessors) - 1

    meshes, materials, nodes = [], [], []
    extensions: set[str] = set()
    for mesh, material in layers:
        attributes = {
            "POSITION": accessor(
                mesh.positions, FLOAT, "VEC3", ARRAY_BUFFER,
                min=mesh.positions.min(axis=0).tolist(), max=mesh.positions.max(axis=0).tolist(),
            ),
            "NORMAL": accessor(mesh.normals, FLOAT, "VEC3", ARRAY_BUFFER),
        }
        if mesh.uvs is not None:
            attributes["TEXCOORD_0"] = accessor(mesh.uvs, FLOAT, "VEC2", ARRAY_BUFFER)
        if mesh.colours is not None:
            attributes["COLOR_0"] = accessor(mesh.colours, UNSIGNED_BYTE, "VEC4", ARRAY_BUFFER, normalized=True)
        primitive_extensions = {}
        if mesh.features is not None:
            # Which feature each vertex belongs to, as an index into the tile's data file.
            attributes["_FEATURE_ID_0"] = accessor(mesh.features, FLOAT, "SCALAR", ARRAY_BUFFER)
            count = int(mesh.features.max()) + 1
            primitive_extensions["EXT_mesh_features"] = {"featureIds": [{"featureCount": count, "attribute": 0}]}
            extensions.add("EXT_mesh_features")
        indices = accessor(mesh.indices, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        materials.append(
            {
                "name": mesh.name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*srgb_to_linear(material.colour), 1.0],
                    "metallicFactor": material.metallic,
                    "roughnessFactor": material.roughness,
                },
            }
        )
        meshes.append(
            {
                "name": mesh.name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": indices,
                        "material": len(materials) - 1,
                        **({"extensions": primitive_extensions} if primitive_extensions else {}),
                    }
                ],
            }
        )
        nodes.append({"name": mesh.name, "mesh": len(meshes) - 1})

    document = {
        "asset": {"version": "2.0", "generator": "boston-pipeline", "copyright": copyright},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    if extensions:
        document["extensionsUsed"] = sorted(extensions)
    if extras:
        document["scenes"][0]["extras"] = extras

    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        handle.write(json_chunk)
        handle.write(struct.pack("<I4s", len(binary), b"BIN\0"))
        handle.write(binary)

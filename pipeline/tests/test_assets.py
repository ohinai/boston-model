"""The hand-made asset loader: glTF in, world-axes triangles with byte colours out, and the contract it enforces."""

import json
import struct

import numpy as np
import pytest

from boston_pipeline import assets
from boston_pipeline.assets import AssetError


def make_glb(path, positions, indices, colours=None, *, colour_component=5123, colour_type="VEC4", node=None,
             extra=None, interleave_positions=False):
    """A minimal .glb with one mesh, built by hand so the tests do not depend on Blender."""
    binary = bytearray()
    views, accessors = [], []

    def add(array, component, kind, target, stride=None, **more):
        binary.extend(b"\0" * (-len(binary) % 4))
        data = np.ascontiguousarray(array).tobytes()
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target}
        if stride:
            view["byteStride"] = stride
        views.append(view)
        binary.extend(data)
        accessors.append({"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": kind, **more})
        return len(accessors) - 1

    positions = np.asarray(positions, dtype=np.float32)
    if interleave_positions:  # x y z and one padding float per vertex: a stride of 16 bytes
        padded = np.column_stack([positions, np.zeros(len(positions), dtype=np.float32)])
        attributes = {"POSITION": add(padded, 5126, "VEC3", 34962, stride=16)}
        accessors[attributes["POSITION"]]["count"] = len(positions)
    else:
        attributes = {"POSITION": add(positions, 5126, "VEC3", 34962)}
    if colours is not None:
        dtype = {5121: np.uint8, 5123: np.uint16, 5126: np.float32}[colour_component]
        attributes["COLOR_0"] = add(np.asarray(colours, dtype=dtype), colour_component, colour_type, 34962,
                                    normalized=colour_component != 5126)
    index = add(np.asarray(indices, dtype=np.uint16), 5123, "SCALAR", 34963)
    binary.extend(b"\0" * (-len(binary) % 4))
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, **(node or {})}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": index}]}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}], **(extra or {}),
    }
    blob = json.dumps(document).encode()
    blob += b" " * (-len(blob) % 4)
    total = 12 + 8 + len(blob) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(blob), b"JSON") + blob
        + struct.pack("<I4s", len(binary), b"BIN\0") + bytes(binary)
    )
    return path


TRIANGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]  # glTF axes: y up


def test_glTF_axes_become_world_axes(tmp_path):
    glb = make_glb(tmp_path / "a.glb", [(1, 2, 3), (1, 2, 4), (2, 2, 3)], [0, 1, 2])
    got = assets.read_glb(glb)["positions"]
    assert got.tolist() == [[1, -3, 2], [1, -4, 2], [2, -3, 2]]  # x east, y north = -z, z up = y


def test_a_nodes_transform_is_applied(tmp_path):
    glb = make_glb(tmp_path / "a.glb", TRIANGLE, [0, 1, 2], node={"translation": [0, 5, 0], "scale": [2, 2, 2]})
    got = assets.read_glb(glb)["positions"]
    assert got[:, 2].tolist() == [5, 5, 7]  # doubled, then lifted 5 m
    assert got[1].tolist() == [2, 0, 5]


def test_a_nodes_rotation_is_applied(tmp_path):
    half = np.sqrt(0.5)  # a quarter turn about glTF's up axis (y), right-handed
    glb = make_glb(tmp_path / "a.glb", TRIANGLE, [0, 1, 2], node={"rotation": [0, half, 0, half]})
    got = assets.read_glb(glb)["positions"]
    # glTF (1, 0, 0) turns to (0, 0, -1), and glTF -z is world north
    assert got[1].round(6).tolist() == [0, 1, 0]


def test_a_mirroring_node_keeps_triangles_facing_out(tmp_path):
    plain = assets.read_glb(make_glb(tmp_path / "a.glb", TRIANGLE, [0, 1, 2]))["indices"]
    mirror = assets.read_glb(make_glb(tmp_path / "b.glb", TRIANGLE, [0, 1, 2], node={"scale": [-1, 1, 1]}))["indices"]
    assert plain.tolist() == [0, 1, 2]
    assert mirror.tolist() == [2, 1, 0]


@pytest.mark.parametrize(
    "component, colour, expected",
    [(5123, [65535, 32768, 0, 65535], [255, 128, 0, 255]), (5121, [255, 128, 0, 255], [255, 128, 0, 255]),
     (5126, [1.0, 0.5, 0.0, 1.0], [255, 128, 0, 255])],
)
def test_colours_become_linear_bytes(tmp_path, component, colour, expected):
    glb = make_glb(tmp_path / "a.glb", TRIANGLE, [0, 1, 2], [colour] * 3, colour_component=component)
    assert assets.read_glb(glb)["colours"][0].tolist() == expected


def test_rgb_colours_get_an_opaque_alpha(tmp_path):
    glb = make_glb(tmp_path / "a.glb", TRIANGLE, [0, 1, 2], [[65535, 0, 0]] * 3, colour_type="VEC3")
    assert assets.read_glb(glb)["colours"][0].tolist() == [255, 0, 0, 255]


def test_interleaved_buffers_are_read(tmp_path):
    glb = make_glb(tmp_path / "a.glb", [(1, 2, 3), (4, 5, 6), (7, 8, 9)], [0, 1, 2], interleave_positions=True)
    assert assets.read_glb(glb)["positions"][1].tolist() == [4, -6, 5]


def test_textures_are_refused(tmp_path):
    glb = make_glb(tmp_path / "a.glb", TRIANGLE, [0, 1, 2], extra={"images": [{"uri": "x.png"}]})
    with pytest.raises(AssetError, match="textures"):
        assets.read_glb(glb)


def test_not_a_glb_is_refused(tmp_path):
    bad = tmp_path / "a.glb"
    bad.write_bytes(b"not a model at all, not at all")
    with pytest.raises(AssetError, match="glb"):
        assets.read_glb(bad)


# --- Placing ---------------------------------------------------------------------------------------------------
def library(tmp_path, positions=None, **sidecar):
    """A one-asset library: a 2 m tall triangle on the ground, with a sidecar that may be overridden."""
    positions = positions if positions is not None else [(0, 0, 0), (1, 0, 0), (0, 2, 0)]  # glTF: 2 m tall
    make_glb(tmp_path / "thing.glb", positions, [0, 1, 2], [[65535, 65535, 65535, 65535]] * 3)
    meta = {"file": "thing.glb", "height": 2.0, "max_triangles": 10, "author": "a", "licence": "MIT", "why": "w", **sidecar}
    (tmp_path / "thing.toml").write_text("\n".join(f"{k} = {json.dumps(v)}" for k, v in meta.items() if v is not None))
    return tmp_path


def test_place_turns_clockwise_from_north(tmp_path):
    asset = assets.load("thing", library(tmp_path, positions=[(0, 0, 0), (0, 0, -1), (0, 2, 0)]))  # a point 1 m north of the base
    north = assets.place(asset, 0, 0, 0)[0][1]
    east = assets.place(asset, 0, 0, 90)[0][1]
    south = assets.place(asset, 0, 0, 180)[0][1]
    assert north.round(6).tolist() == [0, 1, 0]
    assert east.round(6).tolist() == [1, 0, 0]
    assert south.round(6).tolist() == [0, -1, 0]


def test_place_moves_lifts_and_scales(tmp_path):
    asset = assets.load("thing", library(tmp_path))
    positions, normals, indices, colours = assets.place(asset, 10, 20, 0, z0=1.5, scale=3.0)
    assert positions[2].tolist() == [10, 20, 1.5 + 6.0]  # the 2 m top, tripled, on a 1.5 m footing
    assert len(normals) == len(positions) == len(colours) and indices.tolist() == [0, 1, 2]
    assert assets.scale_to(asset, 6.0) == 3.0


# --- The contract --------------------------------------------------------------------------------------------
def test_a_good_asset_loads(tmp_path):
    asset = assets.load("thing", library(tmp_path))
    assert asset.triangles == 1 and asset.height == 2.0 and asset.colours.dtype == np.uint8


def test_a_missing_asset_is_none_not_an_error(tmp_path):
    assert assets.load("nothing-here", tmp_path) is None


def test_assets_can_be_switched_off(tmp_path, monkeypatch):
    directory = library(tmp_path)
    assert assets.load("thing", directory) is not None
    monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")
    assert assets.load("thing", directory) is None


@pytest.mark.parametrize(
    "change, message",
    [({"max_triangles": 0}, "over its budget"), ({"height": 3.0}, "2.00 m tall but"), ({"author": None}, "missing author"),
     ({"file": "gone.glb"}, "does not exist")],
)
def test_a_broken_sidecar_or_budget_names_the_problem(tmp_path, change, message):
    with pytest.raises(AssetError, match=message):
        assets.load("thing", library(tmp_path, **change))


def test_an_asset_floating_above_the_ground_is_refused(tmp_path):
    with pytest.raises(AssetError, match="on the ground"):
        assets.load("thing", library(tmp_path, positions=[(0, 1, 0), (1, 1, 0), (0, 3, 0)]))


def test_an_asset_without_vertex_colours_is_refused(tmp_path):
    make_glb(tmp_path / "thing.glb", [(0, 0, 0), (1, 0, 0), (0, 2, 0)], [0, 1, 2])
    (tmp_path / "thing.toml").write_text(
        'file = "thing.glb"\nheight = 2.0\nmax_triangles = 10\nauthor = "a"\nlicence = "MIT"\nwhy = "w"\n')
    with pytest.raises(AssetError, match="COLOR_0"):
        assets.load("thing", tmp_path)


def test_every_asset_in_the_library_meets_the_contract():
    """Loading enforces the triangle budget, the ground, the height, the colours and the sidecar."""
    for name in assets.available():
        asset = assets.load(name)
        assert asset is not None and 0 < asset.triangles <= asset.max_triangles, name

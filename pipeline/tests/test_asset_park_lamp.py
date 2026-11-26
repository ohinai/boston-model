"""The hand-made park lamp (assets/park-lamp.glb) and the hook in `furniture._lamp` that stands it where the data has
one: placed at the point, turned to its direction, sized to its height, recoloured where the data names a colour, and
replaced by the procedural shape when the asset is missing or switched off."""

import numpy as np
import pytest

from boston_pipeline import assets
from boston_pipeline.furniture import LAMP_ASSET, LAMP_GLASS, LAMP_GLASS_MIN_BYTE, add_object
from boston_pipeline.mesh import MeshBuilder, colour_bytes

PROCEDURAL_BUDGET = 80  # triangles: what the procedural lamp is held to in test_furniture.py


@pytest.fixture(autouse=True)
def _assets_on(monkeypatch):
    monkeypatch.delenv(assets.DISABLE_VARIABLE, raising=False)


def _build(**options):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, "lamp", options.pop("x", 10.0), options.pop("y", 20.0), **options)
    return builder.finish()


def _enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])  # undo the glTF axis swap: east, north, up


def _triangles(mesh):
    return len(mesh.indices) // 3


def _radius(points, around=(0.0, 0.0)):
    return float(np.hypot(points[:, 0] - around[0], points[:, 1] - around[1]).max())


@pytest.fixture
def asset():
    found = assets.load(LAMP_ASSET)
    assert found is not None, "assets/park-lamp.glb and park-lamp.toml are missing"
    return found


# --- the asset itself ----------------------------------------------------------------------------------------------
def test_the_asset_loads_within_its_budget(asset):
    assert asset.max_triangles <= 110  # a thousand copies: every triangle is multiplied
    assert 0 < asset.triangles <= asset.max_triangles
    assert asset.height == pytest.approx(4.9)
    assert asset.positions[:, 2].min() == pytest.approx(0.0, abs=1e-4)
    assert abs(asset.positions[:, 0].mean()) < 0.01 and abs(asset.positions[:, 1].mean()) < 0.01  # centred on its post


def test_the_asset_is_a_slim_black_post_with_a_wide_round_globe(asset):
    p = asset.positions
    shaft, foot = p[(p[:, 2] > 1.0) & (p[:, 2] < 4.35)], p[p[:, 2] < 0.05]  # above the foot, under the collar's rim
    globe = p[(p[:, 2] > 4.4) & (p[:, 2] < 4.9)]
    assert _radius(shaft) < 0.08  # slim
    assert 0.15 < _radius(foot) < 0.3  # on a flared foot
    assert 0.17 < _radius(globe) < 0.25 and _radius(globe) > 2.5 * _radius(shaft)  # a wide globe at the top
    bright = asset.colours[:, :3].mean(axis=1) >= LAMP_GLASS_MIN_BYTE
    assert bright.any() and p[bright, 2].min() > 4.3  # the opal glass is only the globe
    assert len({tuple(c) for c in asset.colours.tolist()}) == 2  # two paints: the iron and the glass
    assert (asset.colours[~bright, :3].astype(int).sum(axis=1) < 30).all()  # the iron is nearly black


def test_every_face_of_the_asset_looks_away_from_its_axis(asset):
    """The viewer draws front faces only, and Blender's own normal recalculation turns open surfaces inside out
    (it did to the collar): a lamp of open lathes is checked face by face."""
    triangles = asset.positions[asset.indices.reshape(-1, 3)]
    face = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centre = triangles.mean(axis=1)
    assert (np.einsum("ij,ij->i", face[:, :2], centre[:, :2]) > 0).all()


# --- the hook: where it stands, which way it turns, how tall it is ----------------------------------------------
def test_the_lamp_stands_at_the_point_on_the_ground():
    enu = _enu(_build(x=12.5, y=-7.25, height=4.9))
    assert enu[:, 2].min() == pytest.approx(0.0, abs=1e-4) and enu[:, 2].max() == pytest.approx(4.9, abs=1e-4)
    foot, top = enu[enu[:, 2] < 0.05], enu[enu[:, 2] > 4.89]
    assert foot[:, :2].mean(axis=0) == pytest.approx([12.5, -7.25], abs=1e-3)
    assert top[:, :2].mean(axis=0) == pytest.approx([12.5, -7.25], abs=1e-3)  # the cap's tip is over the foot


def test_a_lamp_on_a_pier_stands_on_its_base():
    enu = _enu(_build(height=4.9, base=1.5))
    assert enu[:, 2].min() == pytest.approx(1.5, abs=1e-4) and enu[:, 2].max() == pytest.approx(6.4, abs=1e-4)


@pytest.mark.parametrize("height", [3.0, 4.9, 5.0, 9.8])
def test_the_lamp_is_scaled_to_the_height_the_data_gives(height):
    enu = _enu(_build(x=0.0, y=0.0, height=height))
    assert enu[:, 2].max() == pytest.approx(height, abs=1e-4)
    globe = _radius(enu[enu[:, 2] > 0.9 * height])
    assert globe == pytest.approx(0.2 * height / 4.9, rel=0.15)  # the whole lamp grows, not just its post


def test_a_lamp_with_no_height_is_the_assets_own():
    enu = _enu(_build(x=0.0, y=0.0))
    assert enu[:, 2].max() == pytest.approx(4.9, abs=1e-4)


def test_the_lamp_is_turned_to_its_direction():
    """An octagonal foot looks the same every 45 degrees, so a quarter of that tells a turn from none."""

    def phase(direction):
        """Where the foot's corners sit within their 45 degree repeat: 22.5 as modelled, 0 after a turn of 22.5."""
        enu = _enu(_build(x=0.0, y=0.0, height=4.9, direction=direction))
        foot = enu[enu[:, 2] < 0.05]
        angles = np.degrees(np.arctan2(foot[:, 1], foot[:, 0]))
        return np.abs((angles % 45.0 + 22.5) % 45.0 - 22.5)  # distance from the nearest multiple of 45, 0 to 22.5

    assert phase(0.0) == pytest.approx(22.5, abs=0.1)
    assert phase(22.5) == pytest.approx(0.0, abs=0.1)
    assert phase(90.0) == pytest.approx(22.5, abs=0.1)  # two facets round: as modelled again


def test_a_lamp_the_data_gives_a_colour_has_iron_of_that_colour_and_the_same_globe(asset):
    plain, green = _build(height=4.9), _build(height=4.9, colour="#557a55")
    glass = {tuple(c) for c in asset.colours[asset.colours[:, :3].mean(axis=1) >= LAMP_GLASS_MIN_BYTE].tolist()}
    assert {tuple(c) for c in green.colours.tolist()} == glass | {tuple(colour_bytes("#557a55").tolist())}
    assert {tuple(c) for c in plain.colours.tolist()} == {tuple(c) for c in asset.colours.tolist()}
    assert _triangles(green) == _triangles(plain)


def test_the_arm_variant_is_not_the_asset(monkeypatch):
    with_asset = _build(variant="arm", height=8.0)
    monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")
    assert _triangles(_build(variant="arm", height=8.0)) == _triangles(with_asset)


# --- the fallback --------------------------------------------------------------------------------------------------
def test_the_asset_lamp_is_the_one_drawn_and_costs_what_it_says(asset):
    assert _triangles(_build(height=4.9)) == asset.triangles
    assert _triangles(_build(height=4.9)) > PROCEDURAL_BUDGET  # i.e. it is not the procedural shape


def test_with_assets_switched_off_the_procedural_lamp_is_drawn(monkeypatch):
    monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")
    mesh = _build(x=0.0, y=0.0, height=5.0)
    enu = _enu(mesh)
    assert _triangles(mesh) < PROCEDURAL_BUDGET
    assert enu[:, 2].max() == pytest.approx(5.0, abs=1e-6)
    assert (mesh.colours == colour_bytes(LAMP_GLASS)).all(axis=1).any()  # the procedural globe's own white


def test_a_missing_asset_falls_back_to_the_procedural_lamp(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "ASSETS_DIR", tmp_path)  # a library with no lamp in it
    assert assets.load(LAMP_ASSET) is None
    mesh = _build(height=4.9)
    assert 0 < _triangles(mesh) < PROCEDURAL_BUDGET
    assert _enu(mesh)[:, 2].max() == pytest.approx(4.9, abs=1e-6)

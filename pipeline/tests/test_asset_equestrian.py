"""The hand-made horse and rider (assets/equestrian-figure.glb) and its hook in furniture._statue."""

import numpy as np
import pytest

from boston_pipeline import assets
from boston_pipeline.furniture import EQUESTRIAN_HEIGHT, EQUESTRIAN_PEDESTAL_SHARE, add_object
from boston_pipeline.mesh import MeshBuilder, weld
from boston_pipeline.tiles import MATERIALS

NAME = "equestrian-figure"
X, Y = 10.0, 20.0


@pytest.fixture(autouse=True)
def with_assets(monkeypatch):
    """These tests are about the asset, so it is on whatever the environment says (the procedural-figure tests in
    test_monuments.py switch it off themselves)."""
    monkeypatch.delenv(assets.DISABLE_VARIABLE, raising=False)


def _build(**options):
    """(what went into the `monuments` builder, what went into the `figures` builder), each a Mesh or None."""
    monuments = MeshBuilder("monuments", colours=True, features=True)
    figures = MeshBuilder("figures", colours=True, features=True)
    add_object(monuments, "statue", X, Y, variant="equestrian", figures=figures, **options)
    return (None if monuments.empty else monuments.finish()), (None if figures.empty else figures.finish())


def _enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


def _triangles(mesh):
    return len(mesh.indices) // 3


# --- The asset itself -------------------------------------------------------------------------------------------


def test_the_asset_loads_and_is_within_its_budget():
    asset = assets.load(NAME)
    assert asset is not None
    assert asset.max_triangles == 3500
    assert 1000 < asset.triangles <= asset.max_triangles  # a sculpt, not another few bars
    assert asset.height == pytest.approx(3.5)
    assert asset.positions[:, 2].min() == pytest.approx(0.0, abs=0.05)
    assert asset.positions[:, 2].max() == pytest.approx(asset.height, rel=0.03)


def test_the_sidecar_says_who_made_it_and_what_it_stands_for():
    import tomllib

    meta = tomllib.loads((assets.ASSETS_DIR / f"{NAME}.toml").read_text(encoding="utf-8"))
    assert meta["author"].startswith("Claude") and meta["licence"] == "MIT"
    assert "statue=equestrian" in meta["why"] and "n358276954" in meta["why"]
    assert "traced" in meta["why"]  # it says outright that nothing is traced from a photograph
    assert meta["source"] == "src/equestrian_figure.py" and (assets.ASSETS_DIR / meta["source"]).exists()


def test_it_is_a_dark_bronze_with_a_few_tones():
    asset = assets.load(NAME)
    rgb = asset.colours[:, :3].astype(float) / 255  # linear
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert 0.01 < luminance.mean() < 0.12  # bronze, not stone or paint
    assert len({tuple(c) for c in asset.colours[:, :3]}) >= 12  # parts and shading, so the form reads
    assert (asset.colours[:, 3] == 255).all()


def test_it_is_a_horse_standing_on_three_legs_under_a_rider_with_an_arm_out():
    asset = assets.load(NAME)
    x, y, z = asset.positions.T
    length, height = np.ptp(y), np.ptp(z)
    assert length > 0.85 * height  # a horse is long
    assert abs((y.max() + y.min()) / 2) < 0.1 * length  # and centred on its base
    # The rider's hat is the highest thing and is over the horse's back, not over its head or tail.
    top = asset.positions[z > z.max() - 0.05]
    assert abs(top[:, 1].mean()) < 0.2 * length and abs(top[:, 0].mean()) < 0.2
    # One foreleg is raised: forward of the middle only the right foreleg's hoof reaches the ground, while both hind
    # hooves do.
    low = z < 0.15
    assert len(x[low & (y > 0.3)]) and (x[low & (y > 0.3)] > 0).all()
    assert (x[low & (y < -0.3)] > 0).any() and (x[low & (y < -0.3)] < 0).any()
    assert z[(x < -0.05) & (y > 0.3)].min() > 0.2  # the raised hoof hangs well clear of the ground
    # One arm held out to the rider's right (+x), well beyond the horse's width.
    assert x.max() > 2.0 * -x.min() and x.max() > 1.0


# --- The hook ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [0, 90, 215])
def test_the_asset_stands_on_the_pedestal_turned_to_its_direction_and_as_tall_as_the_data_says(direction):
    asset, total = assets.load(NAME), 11.5824  # the Washington statue's height tag
    monuments, figures = _build(height=total, direction=direction, material="bronze")
    z0 = EQUESTRIAN_PEDESTAL_SHARE * total
    scale = assets.scale_to(asset, total - z0)
    expected = assets.place(asset, X, Y, direction, z0, scale)[0]
    assert _triangles(figures) == asset.triangles
    assert np.allclose(_enu(figures), expected, atol=1e-3)  # the same vertices, in order, in the tile's glTF axes
    enu = _enu(figures)
    assert enu[:, 2].min() == pytest.approx(z0, abs=0.05 * scale)  # on top of the pedestal, not in it
    assert enu[:, 2].max() == pytest.approx(total, abs=0.02)  # the statue is as tall as it says
    nose = np.argmax(asset.positions[:, 1])  # the head faces +y in the asset, the statue's direction in the world
    ahead = (enu[nose, 0] - X) * np.sin(np.radians(direction)) + (enu[nose, 1] - Y) * np.cos(np.radians(direction))
    assert ahead == pytest.approx(asset.positions[nose, 1] * scale, rel=1e-3)
    # The pedestal stays procedural: it is in the other builder, of stone, below the figure.
    pedestal = _enu(monuments)
    assert pedestal[:, 2].max() == pytest.approx(z0, abs=1e-3) and pedestal[:, 2].min() >= -1e-6


def test_the_default_statue_size_is_the_same_as_the_procedural_figures():
    _, figures = _build()
    assert _enu(figures)[:, 2].max() == pytest.approx(EQUESTRIAN_HEIGHT, abs=0.02)


def test_with_assets_off_the_procedural_figure_is_drawn_instead(monkeypatch):
    monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")
    monuments, figures = _build()
    assert figures is None  # nothing in the figures layer
    assert _enu(monuments)[:, 2].max() == pytest.approx(EQUESTRIAN_HEIGHT, abs=0.02)
    assert _triangles(monuments) < 450  # the bars of the procedural horse and rider, as before


@pytest.mark.parametrize("options", [{"material": "marble"}, {"material": "granite"}, {"colour": "#8a8a8a"}])
def test_a_statue_that_is_not_bronze_keeps_the_procedural_figure(options):
    monuments, figures = _build(**options)
    assert figures is None and _triangles(monuments) < 450


def test_bronze_written_with_its_pedestals_material_still_gets_the_asset():
    _, figures = _build(material="bronze;granite")  # as on the Washington statue
    assert figures is not None and _triangles(figures) == assets.load(NAME).triangles


def test_without_a_figures_builder_the_asset_goes_into_the_main_one():
    builder = MeshBuilder("monuments", colours=True, features=True)
    add_object(builder, "statue", X, Y, variant="equestrian")
    assert _triangles(builder.finish()) > assets.load(NAME).triangles  # the figure and the pedestal under it


def test_the_figure_keeps_its_feature_id_so_a_note_can_name_the_statue():
    monuments = MeshBuilder("monuments", colours=True, features=True)
    figures = MeshBuilder("figures", colours=True, features=True)
    add_object(monuments, "statue", X, Y, variant="equestrian", figures=figures, feature=7)
    assert set(figures.finish().features.tolist()) == {7} and set(monuments.finish().features.tolist()) == {7}


# --- The layer ---------------------------------------------------------------------------------------------------


def test_figures_is_a_layer_that_takes_its_colour_from_each_vertex():
    assert MATERIALS["figures"].colour == "#ffffff"  # white, so the vertex colours are the colour


def test_the_figure_shares_its_vertices_so_the_viewer_can_smooth_it():
    """The viewer shades the figures layer smooth with computeVertexNormals, which needs the neighbouring triangles to
    share vertices. Tiles are welded on writing (mesh.weld), so the asset must survive that with shared vertices."""
    _, figures = _build()
    welded = weld(figures)
    assert len(welded.positions) < 0.75 * _triangles(welded)  # a closed surface has about half as many vertices as triangles

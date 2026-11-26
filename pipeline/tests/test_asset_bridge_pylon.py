"""The hand-made pylon (assets/bridge-pylon.glb) on the Public Garden's suspension footbridge: what the asset is, and how
bridges.py places it, turns it and sizes it. The procedural pylon stays the fallback."""

import numpy as np
import pytest

from boston_pipeline import assets, bridges
from boston_pipeline.bridges import (
    GLOBE, IRONWORK, LAMP_HEIGHT, LIP, PYLON_ASSET, PYLON_ASSET_SCALE, PYLON_PLINTH, PYLON_SHAFT, PYLON_STAY,
    SUSPENSION, add_bridge, deck_height,
)
from boston_pipeline.mesh import MeshBuilder, colour_bytes

DECK = np.array([200, 180, 150, 255], dtype=np.uint8)
STRUCTURE = np.array([90, 90, 95, 255], dtype=np.uint8)
RAIL = np.array([30, 30, 35, 255], dtype=np.uint8)
RISE = 0.7
WIDTH = 4.9
LENGTH = 40.0
EAST = np.array([[10.0, 5.0], [50.0, 5.0]])  # 40 m due east
DIAGONAL = np.array([[10.0, 5.0], [10.0 + LENGTH / 2 ** 0.5, 5.0 + LENGTH / 2 ** 0.5]])  # 40 m north-east
AT = (12.0, 28.0)  # where along the bridge its pylons stand


def _towers(lamp=4.9):
    return [(a, side, lamp) for a in AT for side in (1, -1)]


def _bridge(line=EAST, towers=None, lamp=4.9):
    builder = MeshBuilder("bridges", colours=True, features=True)
    add_bridge(builder, line, WIDTH, rise=RISE, deck_colour=DECK, structure_colour=STRUCTURE, rail_colour=RAIL,
               feature=3, style=SUSPENSION, towers=_towers(lamp) if towers is None else towers, water_z=-0.6)
    return builder.finish()


def _world(mesh):
    """Vertex positions as (east, north, up), as the builder was given them."""
    return np.column_stack([mesh.positions[:, 0], -mesh.positions[:, 2], mesh.positions[:, 1]])


def _centre(line, a, side):
    """Where a pylon stands: `a` metres along the line, on the deck's left (+1) or right (-1) edge."""
    delta = line[1] - line[0]
    t = delta / np.hypot(*delta)
    left = np.array([-t[1], t[0]])
    return line[0] + t * a + left * side * WIDTH / 2, t


def _heading(t):
    return float(np.degrees(np.arctan2(t[0], t[1])))


@pytest.fixture
def pylon():
    found = assets.load(PYLON_ASSET)
    assert found is not None, "assets/bridge-pylon.toml and .glb are in the library"
    return found


def _colour_set(colours):
    return {tuple(int(v) for v in row) for row in colours}


# --- the asset --------------------------------------------------------------------------------------------------


def test_the_asset_is_within_its_budget_and_stands_on_the_deck(pylon):
    assert pylon.triangles <= pylon.max_triangles <= 900
    assert pylon.height == pytest.approx(4.9)
    assert pylon.positions[:, 2].min() == pytest.approx(0.0, abs=0.01)
    assert pylon.positions[:, 2].max() == pytest.approx(4.9, abs=0.03)


def test_the_asset_keeps_the_procedural_pylons_footprint_so_the_rails_and_cables_still_meet_it(pylon):
    x, y, z = pylon.positions.T
    reach = np.maximum(np.abs(x), np.abs(y))
    assert reach.max() <= PYLON_PLINTH / 2 + 1e-3  # no wider than the plinth it replaces
    assert reach[z < 0.31].max() == pytest.approx(PYLON_PLINTH / 2, abs=0.02)  # the socle is as wide as the plinth
    # The die is the procedural shaft's 1.1 m, and square in plan.
    die = reach[(z > 0.8) & (z < 1.85)]
    assert die.max() == pytest.approx(PYLON_SHAFT / 2, abs=0.02)
    assert np.abs(x[(z > 0.8) & (z < 1.85)]).max() == pytest.approx(np.abs(y[(z > 0.8) & (z < 1.85)]).max(), abs=0.02)
    # Where the cables run in, 2.1 m up, there is stone at least as wide as the shaft on every face.
    stay = (np.abs(z - PYLON_STAY) < 0.25)
    assert np.abs(x[stay]).max() >= PYLON_SHAFT / 2 and np.abs(y[stay]).max() >= PYLON_SHAFT / 2


def test_the_asset_is_a_stone_pylon_with_an_iron_post_and_a_white_globe(pylon):
    colours = _colour_set(pylon.colours)
    globe, iron = tuple(colour_bytes(GLOBE)), tuple(colour_bytes(IRONWORK))
    assert globe in colours and iron in colours
    assert len(colours - {globe, iron}) >= 3  # several tones of granite, so the stepped form reads
    white = (pylon.colours == colour_bytes(GLOBE)).all(axis=1)
    assert pylon.positions[white][:, 2].max() == pytest.approx(4.9, abs=0.01)  # the globe is the top
    assert pylon.positions[white][:, 2].min() > 4.0
    stone = ~white & ~(pylon.colours == colour_bytes(IRONWORK)).all(axis=1)
    assert pylon.positions[stone][:, 2].max() < 3.1  # the post above it is iron


def test_the_stones_outer_faces_look_outward_and_so_do_the_panel_floors(pylon):
    """Counter-clockwise from outside, or the viewer would not draw them: the socle, the die's frame and the
    recessed panels' floors (the panels are darker than the die)."""
    corners = pylon.positions[pylon.indices.reshape(-1, 3)]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    centre = corners.mean(axis=1)
    reach = np.abs(centre[:, :2]).max(axis=1)
    upright = np.abs(normal[:, 2]) < 0.1
    outward = (normal[:, :2] * centre[:, :2]).sum(axis=1) > 0
    socle = upright & (centre[:, 2] < 0.31) & (np.abs(reach - 0.75) < 1e-4)
    frame = upright & (centre[:, 2] > 0.6) & (centre[:, 2] < 1.9) & (np.abs(reach - PYLON_SHAFT / 2) < 1e-4)
    panel_floor = upright & (np.abs(reach - (PYLON_SHAFT / 2 - 0.05)) < 1e-4) & (centre[:, 2] > 0.9) & (centre[:, 2] < 1.7)
    for name, found in (("socle", socle), ("frame", frame), ("panel floor", panel_floor)):
        assert found.sum() >= 8, name
        assert outward[found].all(), name


# --- the hook ---------------------------------------------------------------------------------------------------


def _expected(pylon, line, a, side, lamp=4.9, length=LENGTH):
    """Where the asset's vertices should be: standing at its pylon, facing along the bridge, sized to the lamp."""
    centre, t = _centre(line, a, side)
    scale = min(max((lamp + LIP) / pylon.height, PYLON_ASSET_SCALE[0]), PYLON_ASSET_SCALE[1])
    deck = deck_height(a, length, RISE)
    positions, *_ = assets.place(pylon, centre[0], centre[1], _heading(t), deck - LIP, scale)
    return positions


def _present(world, wanted):
    """Every row of `wanted` is a vertex of `world` (to a millimetre)."""
    near = world[np.hypot(world[:, 0] - wanted[:, 0].mean(), world[:, 1] - wanted[:, 1].mean()) < 2.0]
    gap = np.abs(near[None, :, :] - wanted[:, None, :]).max(axis=2).min(axis=1)
    return gap.max()


@pytest.mark.parametrize("line", [EAST, DIAGONAL], ids=["east", "north-east"])
def test_each_pylon_stands_at_its_tower_turned_along_the_bridge(pylon, line):
    world = _world(_bridge(line))
    for a in AT:
        for side in (1, -1):
            assert _present(world, _expected(pylon, line, a, side)) < 1e-3


def test_a_pylon_turns_with_the_bridge(pylon):
    """The square base is 1.5 m across along the bridge; seen along the world's axes a bridge running north-east shows
    the diagonal of it."""
    def base_width(line):
        world = _world(_bridge(line))
        centre, _ = _centre(line, AT[0], 1)
        stone = (colour_bytes("#7d7a74") == _bridge(line).colours).all(axis=1)  # the base's own granite
        near = stone & (np.hypot(world[:, 0] - centre[0], world[:, 1] - centre[1]) < 1.2)
        return np.ptp(world[near][:, 0])

    assert base_width(EAST) == pytest.approx(1.5, abs=0.02)
    assert base_width(DIAGONAL) == pytest.approx(1.5 * 2 ** 0.5, abs=0.03)


@pytest.mark.parametrize("lamp", [4.9, 5.5, 4.2])
def test_a_pylon_is_sized_so_its_globe_tops_out_at_the_lamps_height(lamp):
    world = _world(_bridge(lamp=lamp))
    white = (_bridge(lamp=lamp).colours == colour_bytes(GLOBE)).all(axis=1)
    for a in AT:
        near = white & (np.abs(world[:, 0] - (10.0 + a)) < 0.5)
        assert world[near][:, 2].max() == pytest.approx(deck_height(a, LENGTH, RISE) + lamp, abs=0.02)


def test_with_no_lamp_height_in_the_data_the_default_is_used():
    mesh = _bridge(towers=[(a, side, None) for a in AT for side in (1, -1)])
    assert _world(mesh)[:, 2].max() == pytest.approx(deck_height(AT[1], LENGTH, RISE) + LAMP_HEIGHT, abs=0.02)


def test_the_scale_is_kept_in_range_so_the_shaft_still_meets_the_rails(pylon):
    low, high = PYLON_ASSET_SCALE
    tall = _world(_bridge(lamp=12.0))[:, 2].max() - deck_height(AT[1], LENGTH, RISE)
    short = _world(_bridge(lamp=2.0))
    white = (_bridge(lamp=2.0).colours == colour_bytes(GLOBE)).all(axis=1)
    assert tall == pytest.approx(pylon.height * high - LIP, abs=0.03)
    assert short[white][:, 2].max() - deck_height(AT[0], LENGTH, RISE) < 3.95
    assert high > 1.0 > low > 0.5


def test_the_asset_carries_the_lamp_so_there_is_one_globe_per_pylon(pylon):
    mesh = _bridge()
    white = (mesh.colours == colour_bytes(GLOBE)).all(axis=1)
    in_asset = int((pylon.colours == colour_bytes(GLOBE)).all(axis=1).sum())
    assert white.sum() == 4 * in_asset  # not also a procedural globe


def test_the_pier_below_the_deck_stays_procedural():
    mesh = _bridge()
    world = _world(mesh)
    assert world[:, 2].min() == pytest.approx(-0.6, abs=1e-3)  # down to the water
    pier = (mesh.colours == colour_bytes(bridges.GRANITE_BASE)).all(axis=1)
    assert pier.sum() > 0 and world[pier][:, 2].max() == pytest.approx(
        max(deck_height(a, LENGTH, RISE) for a in AT) - LIP, abs=0.02)


def test_the_cables_are_unchanged():
    iron = lambda mesh: (mesh.colours == colour_bytes(IRONWORK)).all(axis=1)  # noqa: E731
    with_asset = _bridge()
    # Cables, hangers, rails and lattice are ironwork drawn by the bridge, in the same places whichever pylon it has.
    assert iron(with_asset).sum() > 1000


# --- the fallback -----------------------------------------------------------------------------------------------


def test_with_assets_switched_off_the_procedural_pylon_is_drawn(monkeypatch, pylon):
    with_asset = _bridge()
    monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")
    assert assets.load(PYLON_ASSET) is None
    procedural = _bridge()
    white = (procedural.colours == colour_bytes(GLOBE)).all(axis=1)
    assert white.sum() == 4 * 80 * 3  # four icospheres, 80 triangles each
    assert len(procedural.indices) < len(with_asset.indices)
    own_granite = tuple(colour_bytes("#7d7a74"))  # the asset's base: no tone of the procedural pylon
    assert own_granite in _colour_set(pylon.colours) and own_granite in _colour_set(with_asset.colours)
    assert own_granite not in _colour_set(procedural.colours)
    top = _world(procedural)[:, 2].max()
    assert top == pytest.approx(max(deck_height(a, LENGTH, RISE) for a in AT) + 4.9, abs=0.03)  # the same lamp height


def test_without_the_asset_file_the_procedural_pylon_is_drawn(monkeypatch):
    monkeypatch.setattr(bridges.assets, "load", lambda name, directory=None: None)
    white = (_bridge().colours == colour_bytes(GLOBE)).all(axis=1)
    assert white.sum() == 4 * 80 * 3


def test_the_triangles_the_asset_adds_per_pylon(monkeypatch, pylon):
    with_asset = len(_bridge().indices) // 3
    monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")
    procedural = len(_bridge().indices) // 3
    per_pylon = (with_asset - procedural) / 4
    assert 0 < per_pylon < pylon.triangles  # the asset replaces about 130 triangles of procedural pylon

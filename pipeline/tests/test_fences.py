"""How fences and chains look: railings (wrought-iron picket fences) and ball-topped chain posts."""

import numpy as np
import pytest
from shapely.geometry import LineString

from boston_pipeline.barriers import (
    CHAIN_BALL, CHAIN_HANG, FENCE_COLOUR, IRON_BAR_SPACING, IRON_MAX_HEIGHT, IRON_POST_GAP,
)
from boston_pipeline.features import Scene, classify
from boston_pipeline.mesh import colour_bytes

from test_barriers import _barrier, _enu, _make, _path, _points, _tile, _way

STRAIGHT = np.array([(0.0, 0.0), (10.0, 0.0)])
BENT = np.array([(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (12.0, 8.0)])
RING = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)])
RAILINGS = ("iron", "steel", "metal")


def _tips(mesh, height):
    """East-north positions of everything that ends in a point at the top of the fence."""
    enu = _enu(mesh)
    top = enu[np.abs(enu[:, 2] - height) < 1e-4]
    return np.unique(np.round(top[:, :2], 4), axis=0)


# --- Reading the tags ------------------------------------------------------------
@pytest.mark.parametrize(
    ("tags", "material"),
    [
        ({"barrier": "fence", "fence_type": "railing", "material": "metal"}, "metal"),
        ({"barrier": "fence", "fence_type": "iron", "material": "iron", "colour": "black"}, "iron"),
        ({"barrier": "fence", "fence_type": "chain_link", "material": "metal"}, "wire"),
        ({"barrier": "fence", "fence_type": "Wire_Mesh", "material": "steel"}, "wire"),
        ({"barrier": "fence", "material": "wood"}, "wood"),
        ({"barrier": "fence"}, None),
        ({"barrier": "wall", "fence_type": "chain_link", "material": "brick"}, "brick"),  # only a fence can be wire
    ],
)
def test_wire_fences_are_told_apart_from_railings(tags, material):
    (barrier,) = classify([_way(1, tags, [(0, 0), (10, 0)])]).barriers
    assert barrier.material == material


# --- Railings ----------------------------------------------------------------------
@pytest.mark.parametrize("material", RAILINGS)
@pytest.mark.parametrize("height", [1.0, 1.2, 1.5, IRON_MAX_HEIGHT])
def test_railings_are_sound(material, height):
    mesh = _make("fence", BENT, height, 0.06, material=material)
    assert len(mesh.indices) > 0 and len(mesh.indices) % 3 == 0
    up = _enu(mesh)[:, 2]
    assert np.isfinite(mesh.positions).all() and np.isfinite(mesh.normals).all()
    assert up.min() >= -1e-6 and up.max() == pytest.approx(height, abs=1e-4)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-10).all(), "degenerate triangle"
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all(), "wound the wrong way"
    assert (mesh.features == 3).all()


def test_a_metal_fence_is_railings_and_other_fences_stay_plain():
    railings = _make("fence", STRAIGHT, 1.2, 0.06, material="metal")
    plain = _make("fence", STRAIGHT, 1.2, 0.06)
    # Railings stand on a bar every quarter of a metre, with a point at the top of each.
    assert len(_tips(railings, 1.2)) >= 10 / IRON_BAR_SPACING
    assert len(railings.indices) > 2 * len(plain.indices), "the plain fence is posts and rails only"
    for material in ("wood", "brick", "concrete", "stone", "wire", "unheard_of"):
        assert np.array_equal(_make("fence", STRAIGHT, 1.2, 0.06, material=material).positions, plain.positions), material
    assert len(_make("fence", STRAIGHT, 2.0, 0.06, material="metal").indices) == len(
        _make("fence", STRAIGHT, 2.0, 0.06).indices
    ), "a tall metal fence is a security fence, not railings"


def test_railing_bars_stand_between_two_rails_and_clear_the_posts():
    height = 1.5
    mesh = _make("fence", STRAIGHT, height, 0.06, material="iron")
    enu = _enu(mesh)
    posts = (0.0, 2.5, 5.0, 7.5, 10.0)  # a post about every 2.4 m, spread evenly
    tips = _tips(mesh, height)[:, 0]
    bars = np.array([x for x in tips if min(abs(x - p) for p in posts) > 1e-3])
    assert len(tips) - len(bars) == len(posts) and len(bars) >= 30
    assert min(abs(x - p) for x in bars for p in posts) >= IRON_POST_GAP - 1e-3, "a bar would stand inside a post"
    gaps = np.diff(np.sort(bars[(bars > 0.2) & (bars < 2.3)]))
    assert 0.15 < gaps.min() and gaps.max() < 0.35, "bars at an even spacing"
    # The bottom rail is above the ground, the top rail is below the points, and bars reach from one to the other.
    between = enu[(np.abs(enu[:, 1]) < 0.03) & (enu[:, 0] > 0.5) & (enu[:, 0] < 2.3)]
    assert 0.05 < between[:, 2].min() < 0.2 and between[:, 2].max() == pytest.approx(height, abs=1e-4)
    on_a_bar = enu[np.abs(enu[:, 0] - bars[(bars > 0.5) & (bars < 2.0)][0]) < 0.03]
    assert on_a_bar[:, 2].min() == pytest.approx(between[:, 2].min() + 0.0, abs=0.2) and on_a_bar[:, 2].max() == pytest.approx(height)
    rail = enu[(np.abs(enu[:, 0]) < 1e-4) & (np.abs(enu[:, 1]) > 0.014) & (np.abs(enu[:, 1]) < 0.016)]  # a rail's end
    assert rail[:, 2].min() < 0.2 and 1.0 < rail[:, 2].max() < height - 0.1, "a bottom rail low down, a top rail below the points"
    assert np.abs(enu[:, 1]).max() < 0.09, "everything stays on the fence line"


def test_a_railing_colour_tag_wins_and_the_default_is_dark_iron():
    black = _make("fence", BENT, 1.2, 0.06, material="metal")
    assert (black.colours == colour_bytes(FENCE_COLOUR)).all()
    green = _make("fence", BENT, 1.2, 0.06, colour="#2f6b3a", material="metal")
    assert (green.colours == colour_bytes("#2f6b3a")).all()


@pytest.mark.parametrize("line", [STRAIGHT * 10, np.column_stack([np.arange(0, 100.1, 2.5), np.zeros(41)]), RING * 5])
def test_a_long_railing_stays_cheap(line):
    mesh = _make("fence", line, 1.2, 0.06, material="metal")
    length = LineString(line).length
    assert len(mesh.indices) // 3 < 24 * length, len(mesh.indices) // 3  # about 17 a metre today


def test_a_closed_railing_has_bars_all_the_way_round():
    mesh = _make("fence", RING, 1.2, 0.06, material="metal")
    tips = _tips(mesh, 1.2)
    # Along each of the four sides, from corner to corner, there are bars.
    for side in (tips[:, 1] < 0.05, tips[:, 1] > 9.95, tips[:, 0] < 0.05, tips[:, 0] > 9.95):
        assert side.sum() >= 10 / IRON_BAR_SPACING - 1


# --- Chains --------------------------------------------------------------------------
def test_chain_posts_are_topped_with_a_ball_and_the_chain_hangs_in_a_curve():
    height, span = 0.7, 2.5
    mesh = _make("chain", np.array([(0.0, 0.0), (span, 0.0)]), height, 0.05)
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(height, abs=1e-4) and enu[:, 2].min() >= -1e-6
    # A post at each end, with a ball on it: across its middle the ball is wider than the post's top.
    for x in (0.0, span):
        near = enu[np.hypot(enu[:, 0] - x, enu[:, 1]) < CHAIN_BALL + 1e-3]
        assert near[:, 2].max() == pytest.approx(height, abs=1e-4), "the ball is the top"
        radii = np.hypot(*(near[np.abs(near[:, 2] - (height - CHAIN_BALL)) < 1e-4][:, :2] - [x, 0]).T)
        assert radii.max() == pytest.approx(CHAIN_BALL, abs=1e-3) and (radii < CHAIN_BALL - 0.01).any()
    # The chain leaves each post under the ball and is lowest in the middle, never touching the ground.
    links = enu[(enu[:, 0] > 0.15) & (enu[:, 0] < span - 0.15)]
    hung = height - CHAIN_HANG
    assert links[:, 2].max() <= hung + 0.03, "nothing of the chain above where it is hung"
    lowest = links[:, 2].min()
    assert hung - lowest > 0.12, "it visibly swags"
    assert lowest > 0.15, "off the ground"
    # Symmetric: the same drop a third of the way along from either post.
    third = [links[np.abs(links[:, 0] - x) < 0.05][:, 2].min() for x in (span / 3, 2 * span / 3)]
    assert third[0] == pytest.approx(third[1], abs=1e-4) and third[0] == pytest.approx(lowest, abs=1e-4)


def test_a_short_chain_does_not_reach_the_ground():
    for height in (0.3, 0.5):
        enu = _enu(_make("chain", np.array([(0.0, 0.0), (2.5, 0.0)]), height, 0.05))
        links = enu[(enu[:, 0] > 0.2) & (enu[:, 0] < 2.3)]
        assert links[:, 2].min() > 0.02 and links[:, 2].max() < height


def test_chain_posts_are_sound_on_bends_and_rings():
    for line in (BENT, RING):
        mesh = _make("chain", line, 0.6, 0.05)
        tri = mesh.positions[mesh.indices.reshape(-1, 3)]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert (np.linalg.norm(face, axis=1) > 1e-10).all()
        assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()
        assert len(_tips(mesh, 0.6)) >= len(LineString(line).coords) - 1


# --- In a tile -----------------------------------------------------------------------
def test_a_path_through_a_railing_leaves_a_gap_with_no_bars(tmp_path):
    scene = Scene()
    scene.barriers.append(_barrier("w1", "fence", [(20, 50), (120, 50)], 1.5, 0.06, material="iron"))
    scene.ways.append(_path("w2", [(70, 0), (70, 100)]))
    path, document, layers, semantics = _tile(tmp_path, scene)
    east = _points(path, document, layers)[:, 0]
    assert not ((east > 68.5 + 0.1) & (east < 71.5 - 0.1)).any()
    assert (east < 68.5).sum() > 100 and (east > 71.5).sum() > 100, "bars on both sides"
    assert [o["kind"] for o in semantics["objects"] if o["id"] == "w1"] == ["fence"]

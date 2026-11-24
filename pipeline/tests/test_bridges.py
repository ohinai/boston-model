import numpy as np
import pytest
from shapely.geometry import LineString, Point

from boston_pipeline.bridges import LIP, RAIL_HEIGHT, add_bridge, deck_profile
from boston_pipeline.mesh import MeshBuilder

DECK = np.array([200, 180, 150, 255], dtype=np.uint8)
STRUCTURE = np.array([90, 90, 95, 255], dtype=np.uint8)
RAIL = np.array([30, 30, 35, 255], dtype=np.uint8)
STRAIGHT = np.array([[10.0, 5.0], [50.0, 5.0]])


def _bridge(line=STRAIGHT, width=4.0, rise=1.0, railings=True, feature=0):
    builder = MeshBuilder("bridges", colours=True, features=True)
    add_bridge(builder, line, width, rise=rise, deck_colour=DECK, structure_colour=STRUCTURE,
               rail_colour=RAIL, railings=railings, feature=feature)
    return builder.finish()


def _triangles(mesh, mask=None):
    index = mesh.indices.reshape(-1, 3)
    if mask is not None:
        index = index[mask[index[:, 0]]]
    return index, mesh.positions[index]


def _up(mesh):
    return mesh.positions[:, 1]  # glTF y is up


def test_deck_profile_is_an_arch_from_lip_to_lip():
    h = deck_profile(40.0, 1.0, 41)
    assert h[0] == pytest.approx(LIP) and h[-1] == pytest.approx(LIP)
    assert h[20] == pytest.approx(LIP + 1.0)
    assert np.allclose(h, h[::-1])


def test_normals_agree_with_winding_and_are_unit_length():
    mesh = _bridge()
    index, tri = _triangles(mesh)
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    stored = mesh.normals[index[:, 0]]
    assert (np.einsum("ij,ij->i", face, stored) > 0).all()
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)


def test_walking_surface_is_wide_continuous_and_gentle():
    mesh = _bridge()
    on_deck = (mesh.colours == DECK).all(axis=1)
    index, tri = _triangles(mesh, on_deck)
    normals = mesh.normals[index[:, 0]]
    assert len(index) > 0 and (normals[:, 1] > 0).all()
    pos = mesh.positions[on_deck]
    assert pos[:, 0].min() == pytest.approx(10.0, abs=1e-3) and pos[:, 0].max() == pytest.approx(50.0, abs=1e-3)
    north = -pos[:, 2]
    assert north.min() == pytest.approx(3.0, abs=1e-3) and north.max() == pytest.approx(7.0, abs=1e-3)
    middle = np.abs(pos[:, 0] - 30.0) < 1.5
    assert pos[middle, 1].max() == pytest.approx(LIP + 1.0, abs=0.02)
    slope = np.hypot(normals[:, 0], normals[:, 2]) / normals[:, 1]
    assert slope.max() < 0.12
    # Side faces face outward, the underside faces down.
    side = (mesh.colours == STRUCTURE).all(axis=1)
    s_index, _ = _triangles(mesh, side)
    s_normals = mesh.normals[s_index[:, 0]]
    assert (s_normals[:, 1] < -0.99).any() and (np.abs(s_normals[:, 2]) > 0.99).any()


def test_deck_keeps_its_width_around_a_bend():
    bend = np.array([[0.0, 0.0], [30.0, 0.0], [30 + 30 * np.cos(np.radians(30)), 30 * np.sin(np.radians(30))]])
    mesh = _bridge(bend, width=3.0, rise=0.5)
    assert np.isfinite(mesh.positions).all()
    on_deck = (mesh.colours == DECK).all(axis=1)
    centre = LineString(bend)
    corner = Point(30, 0)
    distances = []
    for x, _, z in mesh.positions[on_deck]:
        p = Point(x, -z)
        if p.distance(corner) > 3.0:
            distances.append(centre.distance(p))
    distances = np.array(distances)
    assert np.allclose(distances, 1.5, atol=0.04)


def test_railings_add_height_and_triangles():
    with_rails, without = _bridge(railings=True), _bridge(railings=False)
    assert _up(with_rails).max() == pytest.approx(LIP + 1.0 + RAIL_HEIGHT, abs=0.06)
    assert len(with_rails.indices) > len(without.indices)
    assert _up(without).max() <= LIP + 1.0 + 0.01


def test_features_and_triangle_budget():
    mesh = _bridge(feature=7)
    assert (mesh.features == 7).all()
    assert len(mesh.indices) // 3 < 2500

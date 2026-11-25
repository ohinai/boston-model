import numpy as np
import pytest

from boston_pipeline.furniture import KINDS, add_object
from boston_pipeline.mesh import MeshBuilder

BUDGETS = {"bench": 80, "lamp": 80, "waste_basket": 40, "bollard": 40, "block": 13}


def _build(kind, x=10.0, y=20.0, **options):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, kind, x, y, **options)
    return builder.finish()


def _enu(mesh):
    """Vertices as (east, north, up): undo the glTF axis swap."""
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


def _triangles(mesh):
    return mesh.positions[mesh.indices.reshape(-1, 3)]


VARIANTS = [(k, {}) for k in KINDS] + [
    ("bench", {"variant": "backless"}),
    ("lamp", {"variant": "arm", "direction": 135}),
    ("fountain", {"variant": "nozzle"}),
    ("statue", {"height": 0.3, "material": "bronze"}),
    ("sculpture", {"height": 5, "material": "marble", "direction": 40}),
    ("obelisk", {"height": 30, "direction": 17}),
    ("monument", {"direction": 63}),
    ("fountain", {"size": 12, "height": 4}),
]


@pytest.mark.parametrize("kind,options", VARIANTS)
def test_faces_wind_counter_clockwise_and_normals_are_unit(kind, options):
    mesh = _build(kind, **options)
    assert len(mesh.indices) > 0 and len(mesh.indices) % 3 == 0
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    tri = _triangles(mesh)
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-9).all(), "degenerate triangle"
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()
    # One normal per face: all three corners of a triangle share it.
    normals = mesh.normals[mesh.indices.reshape(-1, 3)]
    assert np.allclose(normals[:, 0], normals[:, 1]) and np.allclose(normals[:, 0], normals[:, 2])


@pytest.mark.parametrize("kind,options", VARIANTS)
def test_nothing_is_below_ground_and_the_object_stands_at_its_position(kind, options):
    mesh = _build(kind, x=10.0, y=20.0, **options)
    east, north, up = _enu(mesh).T
    assert up.min() >= -1e-6
    assert east.min() <= 10.0 <= east.max() and north.min() <= 20.0 <= north.max()
    assert np.hypot(east - 10.0, north - 20.0).max() < 14.0


@pytest.mark.parametrize(
    "kind,options,top",
    [
        ("lamp", {"height": 6}, 6.0),
        ("lamp", {"variant": "arm", "height": 6}, 6.0),
        ("lamp", {}, 4.9),
        ("lamp", {"variant": "arm"}, 8.0),
        ("flagpole", {}, 12.0),
        ("flagpole", {"height": 20}, 20.0),
        ("waste_basket", {}, 0.95),
        ("bollard", {}, 0.9),
        ("bollard", {"height": 1.0}, 1.0),
        ("block", {}, 0.5),
        ("stele", {"height": 2.2}, 2.2),
        ("obelisk", {}, 6.0),
        ("obelisk", {"height": 12}, 12.0),
        ("column", {}, 8.0),
        ("statue", {}, 3.4),
        ("statue", {"height": 0.3}, 0.3),
        ("sculpture", {"height": 5}, 5.0),
        ("monument", {}, 9.0),
        ("monument", {"height": 20}, 20.0),
        ("fountain", {"height": 3}, 3.0),
    ],
)
def test_stated_heights_are_honoured(kind, options, top):
    assert _enu(_build(kind, **options))[:, 2].max() == pytest.approx(top, abs=0.05)


def test_bench_length_follows_size_and_direction():
    for size in (1.2, 1.8, 3.0):
        east, north, _ = _enu(_build("bench", x=0, y=0, size=size)).T
        assert east.max() - east.min() == pytest.approx(size, abs=0.01)
        assert north.max() - north.min() < 0.6
    _, north, _ = _enu(_build("bench", x=0, y=0, size=3.0, direction=90)).T
    assert north.max() - north.min() == pytest.approx(3.0, abs=0.01)
    east, _, _ = _enu(_build("bench", x=0, y=0, size=99)).T  # clamped to 4 m
    assert east.max() - east.min() == pytest.approx(4.0, abs=0.01)


@pytest.mark.parametrize("direction,axis,sign", [(90, 0, -1), (270, 0, 1), (0, 1, -1), (180, 1, 1)])
def test_bench_backrest_is_behind_the_sitter(direction, axis, sign):
    enu = _enu(_build("bench", x=0, y=0, direction=direction))
    back = enu[enu[:, 2] > 0.6]  # only the backrest (and rear posts) rise above the seat
    assert len(back) and sign * back[:, axis].mean() > 0.15
    assert np.abs(back[:, 1 - axis]).max() < 1.0 and abs(enu[enu[:, 2] <= 0.46, axis].mean()) < 0.1


def test_backless_bench_is_simpler_and_low():
    full, backless = _build("bench"), _build("bench", variant="backless")
    assert len(backless.indices) < len(full.indices)
    assert _enu(backless)[:, 2].max() == pytest.approx(0.45)


def test_lamp_arm_points_along_direction():
    east, north, up = _enu(_build("lamp", x=0, y=0, variant="arm", direction=90)).T
    assert east.max() > 1.5 and east.min() > -0.2 and abs(north).max() < 0.5
    east, north, up = _enu(_build("lamp", x=0, y=0, variant="arm", direction=0)).T
    assert north.max() > 1.5 and abs(east).max() < 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_every_vertex_carries_the_feature_id_and_a_colour(kind):
    mesh = _build(kind, feature=7)
    assert (mesh.features == 7).all()
    assert mesh.colours.shape == (len(mesh.positions), 4) and (mesh.colours[:, 3] == 255).all()


def test_stated_colour_overrides_the_main_material():
    mesh = _build("bench", colour="#336699")
    expected = _build("bench", colour="#336699", seed=5).colours[0]
    assert (mesh.colours[0] == expected).all()  # seat first; seed variation does not touch stated colours
    assert (mesh.colours[0] != _build("bench").colours[0]).any()


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        _build("gazebo")


def test_same_seed_same_mesh_and_other_seed_may_differ():
    a, b = _build("statue", seed=12345), _build("statue", seed=12345)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)
    assert not np.array_equal(_build("statue", seed=1).colours, _build("statue", seed=2).colours)
    _build("statue", seed=-3)  # negative seeds are accepted


@pytest.mark.parametrize("kind,limit", BUDGETS.items())
def test_triangle_budgets(kind, limit):
    assert len(_build(kind).indices) // 3 < limit
    if kind == "lamp":
        assert len(_build(kind, variant="arm").indices) // 3 < limit


def test_statue_plinth_only_when_tall_enough():
    tall, small = _enu(_build("statue", height=3.4)), _enu(_build("statue", height=0.4))
    assert (tall[:, 2] < 0.1).sum() > 0 and np.ptp(tall[:, 0]) > 1.0  # wide plinth foot
    assert np.ptp(small[:, 0]) < 0.4  # figure only, no plinth

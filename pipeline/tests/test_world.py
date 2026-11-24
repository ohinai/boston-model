import numpy as np
import pytest

from boston_pipeline import world


def test_origin_is_on_beacon_hill():
    lon, lat = world.to_lonlat(0.0, 0.0)
    assert lat == pytest.approx(42.3587, abs=1e-3)
    assert lon == pytest.approx(-71.0630, abs=1e-3)


def test_lonlat_round_trip():
    x, y = world.to_local(-71.0589, 42.3601)
    lon, lat = world.to_lonlat(x, y)
    assert lon == pytest.approx(-71.0589, abs=1e-9)
    assert lat == pytest.approx(42.3601, abs=1e-9)


def test_one_degree_of_latitude_is_about_111_km():
    _, y0 = world.to_local(-71.06, 42.0)
    _, y1 = world.to_local(-71.06, 43.0)
    assert y1 - y0 == pytest.approx(111_000, rel=0.01)


def test_tiles_are_half_open_squares():
    assert world.tile_of(0.0, 0.0) == (0, 0)
    assert world.tile_of(-0.01, 255.99) == (-1, 0)
    assert world.tile_of(256.0, -256.0) == (1, -1)
    assert world.tile_bounds(-1, 2) == (-256.0, 512.0, 0.0, 768.0)


def test_tiles_covering_contains_the_box_and_fetch_bounds_contain_the_tiles():
    bbox = (-71.076, 42.350, -71.048, 42.364)
    tiles = world.tiles_covering(bbox)
    xs, ys = world.to_local(np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]]))
    assert world.tile_of(xs[0], ys[0]) in tiles and world.tile_of(xs[1], ys[1]) in tiles
    west, south, east, north = world.lonlat_bounds(tiles, margin=50)
    assert west < bbox[0] and south < bbox[1] and east > bbox[2] and north > bbox[3]


def test_ecef_transform_is_a_rigid_frame_on_the_earth():
    m = np.array(world.ecef_transform()).reshape(4, 4).T  # column-major -> matrix
    rotation, origin = m[:3, :3], m[:3, 3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.linalg.norm(origin) == pytest.approx(6.368e6, rel=1e-3)
    # Up points away from the Earth's centre; north has a positive polar component.
    assert rotation[:, 2].dot(origin) > 0
    assert rotation[2, 1] > 0
    # Grid east must match true east to within the projection's convergence (< 1 degree).
    lon, lat = (np.radians(v) for v in world.to_lonlat(0.0, 0.0))
    true_east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    assert np.degrees(np.arccos(rotation[:, 0].dot(true_east))) < 1.0

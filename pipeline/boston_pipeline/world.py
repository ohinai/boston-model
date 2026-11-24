"""The world frame: the one coordinate system every tool and client shares.

World coordinates are metres east (x) and north (y) of a fixed origin in the
Massachusetts State Plane (mainland) projection, with z up. See
docs/ARCHITECTURE.md before changing anything here: every built tile and every
hand-authored override depends on these constants.
"""

from __future__ import annotations

import math

import numpy as np
from pyproj import Proj

WORLD_CRS = "EPSG:26986"  # NAD83 / Massachusetts Mainland, metres

# Written out (rather than looked up from EPSG) so the pipeline and the browser
# viewer use the identical, datum-shift-free projection. WGS84 longitude and
# latitude are read as NAD83; the two differ by about a metre, which is below
# the accuracy of the source data.
WORLD_PROJ4 = (
    "+proj=lcc +lat_0=41 +lon_0=-71.5 +lat_1=42.6833333333333 "
    "+lat_2=41.7166666666667 +x_0=200000 +y_0=750000 +ellps=GRS80 +units=m +no_defs"
)

# State Plane coordinates of the world origin: a round number on Beacon Hill.
ORIGIN_E = 236_000.0
ORIGIN_N = 901_000.0

# Edge length of a leaf tile in metres. Tile (i, j) covers
# [i * TILE_SIZE, (i + 1) * TILE_SIZE) east and likewise north.
TILE_SIZE = 256.0

# Height of the NAVD88 zero surface above the GRS80 ellipsoid in downtown
# Boston (GEOID18, from the NGS geoid service at 42.3601 N, 71.0589 W). Used
# only to place the tileset on a globe, where heights are ellipsoidal.
GEOID_HEIGHT = -27.74

_GRS80_A = 6378137.0
_GRS80_F = 1 / 298.257222101

_proj = Proj(WORLD_PROJ4)


def to_local(lon, lat):
    """WGS84 lon/lat (scalars or arrays) -> world metres (x east, y north)."""
    e, n = _proj(lon, lat)
    return e - ORIGIN_E, n - ORIGIN_N


def to_lonlat(x, y):
    """World metres -> WGS84 lon/lat."""
    return _proj(np.asarray(x) + ORIGIN_E, np.asarray(y) + ORIGIN_N, inverse=True)


def tile_of(x: float, y: float) -> tuple[int, int]:
    return math.floor(x / TILE_SIZE), math.floor(y / TILE_SIZE)


def tile_bounds(tx: int, ty: int) -> tuple[float, float, float, float]:
    """(min x, min y, max x, max y) of a tile in world metres."""
    return tx * TILE_SIZE, ty * TILE_SIZE, (tx + 1) * TILE_SIZE, (ty + 1) * TILE_SIZE


def tiles_covering(bbox_wsen: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Every tile touched by a WGS84 (west, south, east, north) box."""
    w, s, e, n = bbox_wsen
    xs, ys = to_local(np.array([w, e, e, w]), np.array([s, s, n, n]))
    tx0, ty0 = tile_of(xs.min(), ys.min())
    tx1, ty1 = tile_of(xs.max(), ys.max())
    return [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]


def lonlat_bounds(tiles: list[tuple[int, int]], margin: float = 0.0) -> tuple[float, float, float, float]:
    """WGS84 (west, south, east, north) box enclosing tiles plus a margin in metres."""
    x0 = min(t[0] for t in tiles) * TILE_SIZE - margin
    y0 = min(t[1] for t in tiles) * TILE_SIZE - margin
    x1 = (max(t[0] for t in tiles) + 1) * TILE_SIZE + margin
    y1 = (max(t[1] for t in tiles) + 1) * TILE_SIZE + margin
    lon, lat = to_lonlat(np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]))
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _ecef(lon: float, lat: float, h: float) -> np.ndarray:
    lam, phi = math.radians(lon), math.radians(lat)
    e2 = _GRS80_F * (2 - _GRS80_F)
    n = _GRS80_A / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    return np.array(
        [
            (n + h) * math.cos(phi) * math.cos(lam),
            (n + h) * math.cos(phi) * math.sin(lam),
            (n * (1 - e2) + h) * math.sin(phi),
        ]
    )


def ecef_transform() -> list[float]:
    """Column-major 4x4 taking world (x, y, z) to Earth-centred coordinates.

    This is the 3D Tiles root transform. The world frame is treated as a plane
    tangent at the origin, so it drifts from the ellipsoid by d^2 / 2R (about
    8 m at 10 km); fine for a district, to be refined per tile for a whole city.
    """
    lon0, lat0 = (float(v) for v in to_lonlat(0.0, 0.0))
    lon1, lat1 = (float(v) for v in to_lonlat(100.0, 0.0))
    p0 = _ecef(lon0, lat0, GEOID_HEIGHT)
    lam, phi = math.radians(lon0), math.radians(lat0)
    up = np.array([math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)])
    east = _ecef(lon1, lat1, GEOID_HEIGHT) - p0
    east -= up * east.dot(up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return [*east, 0.0, *north, 0.0, *up, 0.0, *p0, 1.0]

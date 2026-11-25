"""How the ground looks up close: lawns that are not one flat colour, planted beds that are not one slab.

Nothing here changes where anything is. It only decides the colour of each vertex
of a surface, as a pure function of world position (east, north in metres). Two
builds give the same ground, and two tiles agree along their shared edge, so
there is no seam and nothing random to differ between runs.

To give a flat surface patches of colour without a texture, its polygon is cut by
a grid of square cells (aligned with the world, so a cell never straddles a tile
edge) and each cut piece is triangulated on its own. A cell that lies wholly
inside the polygon is two triangles. Pieces share their corner vertices, and a
vertex's colour depends only on where it is, so the cuts are invisible.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import shapely
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from .features import parse_number, polygons_of
from .mesh import MeshBuilder, add_flat, colour_bytes, srgb_to_linear, triangulate

# Pieces of a cut surface smaller than this (square metres) are dropped. It is far smaller than the 0.05
# that tiles.py drops from a surface, because the cuts make many slivers and each one lost is a hole in the ground.
MIN_PIECE = 0.002
# A mulch ring's pieces are dropped below this size: a ring is on top of the ground, so none is missed.
MIN_MULCH = 0.05

# Lawn varies from place to place for real: the NAIP aerial photographs of the
# Public Garden and the Common (July 2023) show open, sunlit lawn varying by
# about 8% in brightness from one 4 to 8 m block to the next, and shifting
# between a yellower and a bluer green. These are the amplitudes, in linear
# light, of three slow waves of that variation, and of the shift in hue.
LAWN_CELL = 8.0  # metres: vertex spacing. The finest wave is a little longer than this, so its patches
# are about a cell across: a vertex takes a value of its own, and a triangle fades from one to the next.
LAWN_WAVES = ((70.0, 0.26), (26.0, 0.26), (12.0, 0.24))  # (wavelength in metres, amplitude)
LAWN_HUE = (30.0, 0.22)  # how much redder or bluer, at most, at a place
LAWN_FLOOR = 0.55  # never darker than this share of the base colour

# What grows in a planted bed. Foliage, with the odd patch of colour: in the
# aerial photographs the beds are mostly green and olive, with some gold and tan
# and very little red; none of them is purple.
BED_CELL = 1.2
BED_FOLIAGE = ["#5f7f3f", "#6d8a45", "#587a4a", "#66803c", "#728a48", "#617c40"]
BED_BLOOM = ["#b59a46", "#c2a850", "#b8693a", "#c9c1a0", "#a24a42", "#b5798a"]
# How many beds in ten have patches of bloom, by the digit a bed's id hashes to.
BED_BLOOMING = 6

# Mulch: bare soil between plants in a bed, and the ring round the foot of a tree on a park lawn.
MULCH = "#5a4837"
MULCH_LINEAR = np.array(srgb_to_linear(MULCH))
MULCH_LIFT = 0.03  # a ring lies this far above the lawn, so the two never fight over the same pixel
MULCH_SIDES = 3  # quad_segs of the ring's outline: 12 sides
# Ring radius in metres: this much, plus this much per metre of trunk radius, within these limits. A tree
# with no trunk size is taken to have a 12 cm trunk radius (a 75 cm girth).
MULCH_RADIUS = (0.9, 1.2, (0.9, 2.0))
MULCH_TRUNK_DEFAULT = 0.12


def _lattice(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """A repeatable pseudo-random value in [-1, 1] for each integer lattice point."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 32767.5 - 1.0


def value_noise(x: np.ndarray, y: np.ndarray, wavelength: float, seed: int = 0) -> np.ndarray:
    """Smooth noise in about [-1, 1] that changes over `wavelength` metres; same input, same output."""
    u, v = np.asarray(x, dtype=np.float64) / wavelength, np.asarray(y, dtype=np.float64) / wavelength
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * fu * (fu * (fu * 6 - 15) + 10), fv * fv * fv * (fv * (fv * 6 - 15) + 10)
    a, b = _lattice(iu, iv, seed), _lattice(iu + 1, iv, seed)
    c, d = _lattice(iu, iv + 1, seed), _lattice(iu + 1, iv + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def lawn_factors(x: np.ndarray, y: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Multipliers (n, 3), in linear light, for the colour of lawn at each position."""
    light = np.ones(len(x))
    for k, (wavelength, amplitude) in enumerate(LAWN_WAVES):
        light = light + strength * amplitude * value_noise(x, y, wavelength, seed=11 + k)
    light = np.maximum(light, LAWN_FLOOR)
    hue = strength * LAWN_HUE[1] * value_noise(x, y, LAWN_HUE[0], seed=31)
    return np.column_stack([light * (1 + hue), light, light * (1 - 1.3 * hue)])


def cut(polygon: Polygon, cell: float) -> list[Polygon]:
    """The pieces of `polygon` in each cell of a grid of squares `cell` metres across, aligned with the world."""
    x0, y0, x1, y1 = polygon.bounds
    i0, i1, j0, j1 = math.floor(x0 / cell), math.ceil(x1 / cell), math.floor(y0 / cell), math.ceil(y1 / cell)
    if (i1 - i0) * (j1 - j0) <= 1:
        return [polygon]
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
    cells = shapely.box(ii.ravel() * cell, jj.ravel() * cell, (ii.ravel() + 1) * cell, (jj.ravel() + 1) * cell)
    shapely.prepare(polygon)
    touching = cells[shapely.intersects(polygon, cells)]
    inside = shapely.contains(polygon, touching)
    pieces = [c for c, whole in zip(touching, inside) if whole]
    for part in shapely.intersection(polygon, touching[~inside]):
        pieces += [p for p in polygons_of(part) if p.area >= MIN_PIECE]
    return pieces


def add_patchy(builder: MeshBuilder, polygon: Polygon, z: float, origin, colour_at, cell: float, feature: int = 0) -> None:
    """A horizontal face over `polygon` at height z, coloured per vertex by `colour_at(x, y)` (world metres
    to an (n, 4) array of linear RGBA bytes). `builder` must have been made with colours."""
    parts_positions, parts_indices, parts_colours, count = [], [], [], 0
    for piece in cut(polygon, cell):
        vertices, triangles = triangulate(piece)
        if not len(triangles):
            continue
        parts_positions.append(np.column_stack([vertices - origin, np.full(len(vertices), z)]))
        parts_indices.append(triangles + count)
        parts_colours.append(colour_at(vertices[:, 0], vertices[:, 1]))
        count += len(vertices)
    if count:
        builder.add(
            np.concatenate(parts_positions), (0, 0, 1), np.concatenate(parts_indices),
            colour=np.concatenate(parts_colours), feature=feature,
        )


def _bytes(linear: np.ndarray) -> np.ndarray:
    rgb = np.clip(linear, 0.0, 1.0)
    return np.column_stack([np.round(rgb * 255), np.full(len(rgb), 255)]).astype(np.uint8)


def lawn_colour(hex_colour: str, strength: float = 1.0):
    """A colour function for lawn of this base colour: the same patches wherever lawn is drawn."""
    base = np.array(srgb_to_linear(hex_colour))
    return lambda x, y: _bytes(base * lawn_factors(x, y, strength))


def add_lawn(builder: MeshBuilder, polygon: Polygon, z: float, origin, hex_colour: str, strength: float = 1.0) -> None:
    add_patchy(builder, polygon, z, origin, lawn_colour(hex_colour, strength), LAWN_CELL)


def bed_style(ident: str) -> tuple[str, str | None]:
    """(foliage colour, bloom colour or None) for a bed, from its id: the same beds every build."""
    seed = zlib.crc32(ident.encode())
    foliage = BED_FOLIAGE[seed % len(BED_FOLIAGE)]
    blooming = (seed >> 8) % 10 < BED_BLOOMING
    return foliage, BED_BLOOM[(seed >> 12) % len(BED_BLOOM)] if blooming else None


def bed_colour(ident: str):
    """A colour function for the planting of one bed: green foliage in drifts, with patches of bloom if it has any."""
    foliage, bloom = bed_style(ident)
    seed = zlib.crc32(ident.encode()) & 0xFFFF
    green = np.array(srgb_to_linear(foliage))
    flower = np.array(srgb_to_linear(bloom)) if bloom else None

    def colour_at(x, y):
        leaf = (1 + 0.22 * value_noise(x, y, 2.4, seed) + 0.12 * value_noise(x, y, 1.1, seed + 1))[:, None]
        planting = green * leaf
        if flower is not None:
            # Patches of bloom where a slow wave runs high: well defined, but not everywhere.
            patch = np.clip((value_noise(x, y, 3.6, seed + 2) + 0.35 * value_noise(x, y, 1.4, seed + 3) - 0.3) * 3.0, 0.0, 0.75)
            planting = (green * (1 - patch[:, None]) + flower * patch[:, None]) * leaf
        # Bare mulch shows between the plants in places, which also sets a bed off from the lawn round it.
        bare = np.clip((value_noise(x, y, 4.4, seed + 4) + 0.2) * 1.8, 0.0, 0.55)[:, None]
        return _bytes(planting * (1 - bare) + MULCH_LINEAR * bare)

    return colour_at


def add_bed_planting(builder: MeshBuilder, polygon: Polygon, z: float, origin, ident: str) -> None:
    add_patchy(builder, polygon, z, origin, bed_colour(ident), BED_CELL)


def mulch_radius(tags: dict) -> float:
    """How far a mulch ring reaches from the trunk: wider for a thicker trunk (`circumference`, metres)."""
    base, per_metre, (low, high) = MULCH_RADIUS
    girth = parse_number(tags.get("circumference"))
    trunk = girth / (2 * math.pi) if girth and girth > 0 else MULCH_TRUNK_DEFAULT
    return min(max(base + per_metre * trunk, low), high)


def add_mulch(builder: MeshBuilder, x: float, y: float, radius: float, origin, within=None) -> None:
    """A ring of mulch round a tree at (x, y), world metres, cut to `within` (the lawn it stands on) if given."""
    ring = ShapelyPoint(x, y).buffer(radius, quad_segs=MULCH_SIDES)
    for piece in polygons_of(ring if within is None else ring.intersection(within)):
        if piece.area >= MIN_MULCH:
            add_flat(builder, piece, MULCH_LIFT, origin, colour_bytes(MULCH))

"""Statues, pedestals, equestrian and animal figures, and the park lamp's globe."""

from pathlib import Path

import numpy as np
import pytest

from boston_pipeline import fixes, world
from boston_pipeline.features import classify
from boston_pipeline.furniture import (
    ANIMAL_LENGTH,
    EQUESTRIAN_HEIGHT,
    EQUESTRIAN_PEDESTAL_LENGTH,
    EQUESTRIAN_PEDESTAL_SHARE,
    EQUESTRIAN_PEDESTAL_WIDTH,
    LAMP_GLASS,
    STATUE_FIGURE_MAX,
    STATUE_GROUND_HEIGHT,
    STATUE_HEIGHT,
    STATUE_PEDESTAL_SHARE,
    _frame,
    _Parts,
    add_object,
    limb,
)
from boston_pipeline.mesh import MeshBuilder, colour_bytes


def _build(kind, x=10.0, y=20.0, **options):
    builder = MeshBuilder("furniture", colours=True, features=True)
    add_object(builder, kind, x, y, **options)
    return builder.finish()


def _enu(mesh):
    """Vertices as (east, north, up): undo the glTF axis swap."""
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


def _triangle_count(mesh):
    return len(mesh.indices) // 3


def _check_solid(mesh):
    """Counter-clockwise faces with unit normals and no flat triangles, as every furniture mesh must have."""
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-9).all(), "degenerate triangle"
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all()


# --- limb ------------------------------------------------------------------------


def _limb_mesh(a, b, r0, r1=None, **options):
    parts, builder = _Parts(), MeshBuilder("t", colours=True, features=True)
    limb(parts, colour_bytes("#888888"), _frame(0.0, 0.0, 0.0), a, b, r0, r1, **options)
    parts.flush(builder, 1)
    return builder.finish()


@pytest.mark.parametrize(
    "a,b,options",
    [
        ((0, 0, 0), (0, 0, 2), {}),  # upright
        ((0, 0, 1), (1, 2, 0.5), {}),  # leaning
        ((0, 0, 0), (0, 3, 0.2), {"sides": 6, "flat": 0.6}),  # lying along the front
        ((0, 0, 0), (2, 0, 0), {}),  # along the right-hand side
        ((0, 0, 0), (0, 0, 2), {"closed": False}),
    ],
)
def test_limb_is_a_sound_flat_shaded_solid(a, b, options):
    mesh = _limb_mesh(a, b, 0.2, 0.1, **options)
    _check_solid(mesh)
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    normals = mesh.normals[mesh.indices.reshape(-1, 3)]
    assert np.allclose(normals[:, 0], normals[:, 1]) and np.allclose(normals[:, 0], normals[:, 2])
    # Faces point away from the bar's middle (checked per triangle, in glTF axes).
    middle = np.array([(a[0] + b[0]) / 2, (a[2] + b[2]) / 2, -(a[1] + b[1]) / 2])
    assert (np.einsum("ij,ij->i", normals[:, 0], tri.mean(axis=1) - middle) > 0).all()


def test_limb_runs_between_its_ends_and_tapers():
    east, north, up = _enu(_limb_mesh((0, 0, 0), (0, 0, 2), 0.3, 0.1)).T
    assert up.min() == pytest.approx(0.0) and up.max() == pytest.approx(2.0)
    wide, narrow = np.ptp(east[up < 0.01]), np.ptp(east[up > 1.99])
    assert wide == pytest.approx(0.6 / 2**0.5, abs=0.01) and narrow < wide / 2.5  # a square's circumradius is 0.7 of its half-width
    assert _triangle_count(_limb_mesh((0, 0, 0), (0, 0, 2), 0.3)) == 12  # four sides and two caps


# --- the pedestal ----------------------------------------------------------------


def _section(mesh, z):
    """The points (east, north) where the mesh's edges cross the horizontal plane at height `z`."""
    enu = _enu(mesh)
    tri = enu[mesh.indices.reshape(-1, 3)]
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, q = tri[:, a], tri[:, b]
        crossing = (p[:, 2] - z) * (q[:, 2] - z) < 0
        t = (z - p[crossing, 2]) / (q[crossing, 2] - p[crossing, 2])
        points.append(p[crossing, :2] + t[:, None] * (q[crossing, :2] - p[crossing, :2]))
    return np.concatenate(points)


def _width_at(mesh, z, axis=0):
    return np.ptp(_section(mesh, z)[:, axis])


def test_standing_statue_has_a_stepped_pedestal_with_a_narrower_die():
    total = 4.0
    mesh = _build("statue", height=total)
    top = total - min(total * (1 - STATUE_PEDESTAL_SHARE), STATUE_FIGURE_MAX)
    assert top == pytest.approx(STATUE_PEDESTAL_SHARE * total)
    footing, base, die, cap, cap_top = (_width_at(mesh, f * top) for f in (0.025, 0.1, 0.5, 0.88, 0.96))
    assert footing > base > die  # it steps in from the footing, through the base, to the die
    assert cap > die and cap_top > die  # and the two-step cap overhangs the die
    assert cap > cap_top
    assert die == pytest.approx(0.34 * total * 0.88, abs=0.02)
    assert footing == pytest.approx(0.34 * total * 1.26, abs=0.02)
    # The pedestal is about as tall as the figure on it: not a short plain block.
    assert 0.4 * total < top < 0.55 * total


def test_pedestal_is_granite_coloured_and_not_the_figures_bronze():
    mesh = _build("statue", height=3.4, seed=1)
    up = _enu(mesh)[:, 2]
    low, high = mesh.colours[up < 0.5], mesh.colours[up > 2.6]
    assert not (low[0] == high[0]).all()
    assert low[:, :3].mean() > high[:, :3].mean()  # paler stone under dark bronze


def test_a_very_tall_statue_gets_a_taller_pedestal_not_a_giant_figure():
    mesh = _build("statue", height=12.0)
    up = _enu(mesh)[:, 2]
    assert up.max() == pytest.approx(12.0, abs=0.05)
    # The bronze starts where the stone ends: no figure taller than STATUE_FIGURE_MAX.
    stone_top = 12.0 - STATUE_FIGURE_MAX
    assert (up > stone_top + 0.05).any() and (up[up > stone_top + 0.05] - stone_top).max() <= STATUE_FIGURE_MAX + 0.05


def test_a_very_tall_statues_pedestal_is_slender_not_a_giant_block():
    mesh = _build("statue", height=12.0)
    assert _width_at(mesh, 0.5 * 8.6) == pytest.approx(2.6 * 0.88, abs=0.02)  # the die, capped at 2.6 m before the courses
    assert _width_at(mesh, 0.01) == pytest.approx(2.6 * 1.26, abs=0.02)


def test_small_statues_have_no_pedestal():
    mesh = _build("statue", height=1.0)
    assert np.ptp(_enu(mesh)[:, 0]) < 0.6 and _enu(mesh)[:, 2].min() >= -1e-6


# --- a person standing -------------------------------------------------------------


def test_standing_figure_has_legs_a_coat_shoulders_and_a_head():
    total = 1.2  # below the pedestal threshold, so the figure is alone
    mesh = _build("statue", height=total)
    _check_solid(mesh)
    enu = _enu(mesh)
    widths = {name: _width_at(mesh, total * f) for name, f in (("legs", 0.2), ("coat", 0.35), ("shoulders", 0.78), ("head", 0.95))}
    assert widths["coat"] > widths["legs"] and widths["shoulders"] > widths["head"]  # a flared coat; shoulders wider than the head
    assert widths["head"] < 0.2 * total
    assert enu[:, 2].max() == pytest.approx(total, abs=1e-6)
    assert _triangle_count(mesh) < 160


def test_standing_figure_arms_hang_beside_the_body():
    mesh = _build("statue", height=1.2)
    # At the hips the section is the coat with an arm on each side of it: far wider than the legs at the knees.
    assert _width_at(mesh, 0.65 * 1.2) > _width_at(mesh, 0.2 * 1.2) * 1.5
    east = _section(mesh, 0.7 * 1.2)[:, 0] - 10.0
    assert east.min() < -0.12 * 1.2 and east.max() > 0.12 * 1.2  # an arm each side, out past the coat's waist


# --- a rider on a horse --------------------------------------------------------------


@pytest.mark.parametrize("direction", [0, 90, 215])
def test_equestrian_statue_stands_on_a_longer_than_wide_pedestal_and_is_the_stated_height(direction):
    total = 11.5824
    mesh = _build("statue", variant="equestrian", height=total, direction=direction, material="bronze")
    _check_solid(mesh)
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(total, abs=0.02) and enu[:, 2].min() >= -1e-6
    # Project onto the way it faces (heading clockwise from north) and across it.
    t = np.radians(direction)
    along = (enu[:, 0] - 10.0) * np.sin(t) + (enu[:, 1] - 20.0) * np.cos(t)
    across = (enu[:, 0] - 10.0) * np.cos(t) - (enu[:, 1] - 20.0) * np.sin(t)
    footing = enu[:, 2] < 0.05 * EQUESTRIAN_PEDESTAL_SHARE * total
    assert np.ptp(along[footing]) == pytest.approx(1.26 * EQUESTRIAN_PEDESTAL_LENGTH * total, abs=0.05)
    assert np.ptp(across[footing]) == pytest.approx(1.26 * EQUESTRIAN_PEDESTAL_WIDTH * total, abs=0.05)
    assert np.ptp(along[footing]) > np.ptp(across[footing])


def test_equestrian_figure_is_a_horse_under_a_rider():
    total = 10.0
    mesh = _build("statue", variant="equestrian", height=total, direction=90, material="bronze")
    enu = _enu(mesh)
    stone_top = EQUESTRIAN_PEDESTAL_SHARE * total
    figure = enu[enu[:, 2] > stone_top + 0.01]
    fh = total - stone_top
    # The rider's hat is the highest thing and he sits well above the horse's back.
    hat, back = figure[figure[:, 2] > stone_top + 0.92 * fh], figure[(figure[:, 2] > stone_top + 0.5 * fh) & (figure[:, 2] < stone_top + 0.6 * fh)]
    assert len(hat) and np.ptp(hat[:, 1]) > 0 and np.ptp(hat[:, 0]) < 0.35 * fh  # only the rider reaches the top
    assert np.ptp(back[:, 0]) > 0.45 * fh  # at the saddle, the horse's body runs along the way it faces (east)
    # A horse is longer than it is wide and stands on a bronze base on the pedestal.
    assert np.ptp(figure[:, 0]) > 1.8 * np.ptp(figure[:, 1])
    low = figure[figure[:, 2] < stone_top + 0.1 * fh]
    assert len(low) and np.ptp(low[:, 0]) > 0.5 * fh


def test_equestrian_default_height_and_triangle_budget():
    mesh = _build("statue", variant="equestrian")
    assert _enu(mesh)[:, 2].max() == pytest.approx(EQUESTRIAN_HEIGHT, abs=0.02)
    assert _triangle_count(mesh) < 450  # few of them, but a lamp is 70


def test_equestrian_is_a_real_bronze_unless_told_otherwise():
    bronze = _build("statue", variant="equestrian", height=6, material="bronze", seed=3)
    marble = _build("statue", variant="equestrian", height=6, material="marble", seed=3)
    up = _enu(bronze)[:, 2]
    assert not np.array_equal(bronze.colours[up > 4], marble.colours[up > 4])
    assert np.array_equal(bronze.colours[up < 0.05], marble.colours[up < 0.05])  # the same stone under both


# --- animals and ground-level statues -------------------------------------------------


def test_animal_statue_is_low_and_on_the_ground_with_no_pedestal():
    mesh = _build("statue", variant="animal")
    _check_solid(mesh)
    enu = _enu(mesh)
    assert enu[:, 2].min() >= -1e-6 and enu[:, 2].min() < 0.05
    assert enu[:, 2].max() < 0.7 * ANIMAL_LENGTH  # well below a person, and nothing like 3.4 m
    assert _triangle_count(mesh) < 60


@pytest.mark.parametrize("direction", [0, 90, 225])
def test_animal_length_follows_size_and_direction(direction):
    enu = _enu(_build("statue", variant="animal", size=1.2, direction=direction))
    t = np.radians(direction)
    along = (enu[:, 0] - 10.0) * np.sin(t) + (enu[:, 1] - 20.0) * np.cos(t)
    assert np.ptp(along) == pytest.approx(1.2, abs=0.05)
    assert np.ptp(enu[:, 0]) < 1.2 * 1.05 and np.ptp(enu[:, 1]) < 1.2 * 1.05


def test_animal_height_tag_sets_its_size():
    assert _enu(_build("statue", variant="animal", height=0.8))[:, 2].max() == pytest.approx(0.8, abs=0.06)


def test_statue_on_the_ground_has_no_pedestal_and_a_person_sized_default():
    mesh = _build("statue", variant="ground")
    enu = _enu(mesh)
    assert enu[:, 2].max() == pytest.approx(STATUE_GROUND_HEIGHT, abs=0.01) and STATUE_GROUND_HEIGHT < STATUE_HEIGHT
    assert np.ptp(enu[:, 0]) < 0.5 * STATUE_GROUND_HEIGHT  # a figure, no wide footing


# --- the park lamp ------------------------------------------------------------------


def _lamp(**options):
    return _build("lamp", x=0.0, y=0.0, **options)


def test_park_lamp_is_a_slim_post_with_a_wide_globe_near_the_top():
    mesh = _lamp(height=5.0)
    _check_solid(mesh)
    enu = _enu(mesh)
    radius = lambda z: np.hypot(*(_section(mesh, z) - [0.0, 0.0]).T).max()  # noqa: E731
    assert enu[:, 2].max() == pytest.approx(5.0, abs=1e-6)
    assert radius(2.5) < 0.08  # a slim shaft
    assert 0.12 < radius(0.05) < 0.3  # on a flared foot
    globe = max(radius(z) for z in np.linspace(4.4, 4.8, 9))
    assert globe > 2.5 * radius(2.5)  # the globe is wider than the post
    assert _triangle_count(mesh) < 80


def test_park_lamp_globe_is_white_glass_on_a_dark_post():
    mesh = _lamp(height=4.9)
    up = _enu(mesh)[:, 2]
    glass = colour_bytes(LAMP_GLASS)
    in_globe = (mesh.colours == glass).all(axis=1)
    assert in_globe.any() and up[in_globe].min() > 3.8  # only near the top
    assert (mesh.colours[up < 1.0, :3].astype(int).sum(axis=1) < glass[:3].astype(int).sum() / 2).all()


@pytest.mark.parametrize("height", [2.5, 4.9, 6.0, 14.0])
def test_park_lamp_scales_to_its_height(height):
    enu = _enu(_lamp(height=height))
    assert enu[:, 2].max() == pytest.approx(height, abs=1e-6) and enu[:, 2].min() >= -1e-6


# --- how the data says which figure -----------------------------------------------------


def _node(ident, tags):
    lon, lat = (float(v) for v in world.to_lonlat(5.0, 5.0))
    return {"type": "node", "id": ident, "lon": lon, "lat": lat, "tags": tags}


def _things(*nodes):
    return {thing.id: thing for thing in classify(list(nodes)).objects}


def test_tags_choose_the_kind_of_figure():
    things = _things(
        _node(1, {"tourism": "artwork", "artwork_type": "statue", "historic": "monument", "statue": "equestrian", "height": "11.5824"}),
        _node(2, {"historic": "monument", "statue": "equestrian"}),  # no artwork tag at all
        _node(3, {"tourism": "artwork", "artwork_type": "statue", "animal": "duck", "support": "ground"}),
        _node(4, {"tourism": "artwork", "artwork_type": "sculpture", "animal": "sea_turtle", "length": "1.2", "support": "ground"}),
        _node(5, {"tourism": "artwork", "artwork_type": "statue", "support": "ground"}),
        _node(6, {"tourism": "artwork", "artwork_type": "statue", "support": "pedestal"}),
        _node(7, {"tourism": "artwork", "artwork_type": "sculpture"}),
        _node(8, {"historic": "memorial", "memorial": "plaque", "statue": "equestrian", "height": "1"}),
    )
    assert (things["n1"].kind, things["n1"].variant, things["n1"].height) == ("statue", "equestrian", pytest.approx(11.5824))
    assert (things["n2"].kind, things["n2"].variant) == ("statue", "equestrian")
    assert (things["n3"].kind, things["n3"].variant, things["n3"].size) == ("statue", "animal", None)
    assert (things["n4"].kind, things["n4"].variant, things["n4"].size) == ("statue", "animal", pytest.approx(1.2))
    assert (things["n5"].kind, things["n5"].variant) == ("statue", "ground")
    assert (things["n6"].kind, things["n6"].variant) == ("statue", None)  # a pedestal is the default
    assert (things["n7"].kind, things["n7"].variant) == ("sculpture", None)  # an abstract sculpture stays abstract
    assert things["n8"].kind == "plaque" and things["n8"].variant is None


# --- the fix files ---------------------------------------------------------------------


FIXES = Path(__file__).resolve().parents[2] / "fixes"


def test_monument_fix_files_load_and_only_touch_tags_the_pipeline_reads():
    loaded = fixes.load(FIXES)
    mine = {ident: patch for ident, patch in loaded.patches.items() if patch["file"].startswith("public-garden/monuments-")}
    assert len(mine) >= 15
    readable = {"statue", "animal", "support", "direction", "artwork_type", "material", "length", "height"}
    for ident, patch in mine.items():
        assert set(patch["set"]) <= readable, (ident, patch["set"])
        assert not patch["delete"] and not patch["unset"]
    equestrian = [i for i, p in mine.items() if p["set"].get("statue") == "equestrian"]
    assert "n358276954" in equestrian  # the Washington statue
    assert mine["n358276954"]["set"]["material"] == "bronze;granite"  # a bronze figure, not a granite one

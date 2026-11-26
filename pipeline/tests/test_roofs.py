"""Pitched roofs: gabled, hipped, pyramidal, skillion, gambrel, mansard, round and domed roofs from the roof tags.

The shapes are built in roofs.py; the tags are read in features.py (roof_tags, building_colours). A building with a roof has
walls that stop at the eaves and a roof that rises from there, so its height is `height` (to the top of the roof) minus the
roof's own height.
"""

import json
import struct

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from boston_pipeline import world
from boston_pipeline.features import (
    ROOF_ALLOWANCE,
    ROOF_LEVEL_HEIGHT,
    ROOF_MATERIAL_COLOURS,
    ROOFS,
    SHINGLE_ROOFS,
    SLATE_ROOFS,
    Building,
    Scene,
    building_colours,
    classify,
    roof_tags,
)
from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.roofs import MIN_FILL, MIN_RISE, MIN_SPAN, _sound, add_roof, plan_roof
from boston_pipeline.tiles import build


def _building(polygon, tags=None, height=12.0, source="height", min_height=0.0, wall="#8a4b3c"):
    tags = {"building": "yes", **(tags or {})}
    return Building("w1", polygon, height, min_height, source, wall, "#555555", tags, roof=roof_tags(tags))


def _zup(builder):
    """Triangles (m, 3, 3) in z-up metres, and the mesh, of a builder that is not empty."""
    mesh = builder.finish()
    p = mesh.positions.astype(float)
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])[mesh.indices.reshape(-1, 3)], mesh


def _draw(building, origin=(0.0, 0.0), neighbours=()):
    """(plan, roof triangles, wall-fill triangles, wall-fill mesh) for a building; None where the roof is flat."""
    plan = plan_roof(building, neighbours)
    if plan is None:
        return None, None, None, None
    walls = MeshBuilder("building_walls", uvs=True, colours=True, features=True)
    roofs = MeshBuilder("building_roofs", colours=True, features=True)
    add_roof(walls, roofs, building, plan, np.asarray(origin), feature=7)
    roof_tris, _ = _zup(roofs)
    if walls.empty:
        return plan, roof_tris, np.zeros((0, 3, 3)), None
    fill_tris, fill_mesh = _zup(walls)
    return plan, roof_tris, fill_tris, fill_mesh


def _area_from_above(tris):
    a, b, c = tris[:, 0, :2], tris[:, 1, :2], tris[:, 2, :2]
    return float(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum() / 2)


def _area(tris):
    return float(np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2)


def _verts(tris):
    return tris.reshape(-1, 3)


# --- reading the tags ---------------------------------------------------------------------------------------------------


def test_roof_tags_name_the_shapes_we_draw_and_nothing_else():
    assert roof_tags({"building": "yes", "roof:shape": "gabled"}).shape == "gable"
    assert roof_tags({"building": "yes", "roof:shape": "hipped"}).shape == "hip"
    assert roof_tags({"building": "yes", "roof:shape": "half-hipped"}).shape == "half_hip"
    assert roof_tags({"building": "yes", "roof:shape": "pyramidal"}).shape == "pyramid"
    assert roof_tags({"building": "yes", "roof:shape": "round"}).shape == "barrel"
    assert roof_tags({"building": "yes", "roof:shape": "quadruple_saltbox"}).shape == "mansard"
    assert roof_tags({"building": "yes", "roof:shape": "hipped;pyramidal"}).shape == "hip"  # the first one we know
    assert roof_tags({"building": "yes", "roof:shape": "Gabled "}).shape == "gable"
    assert roof_tags({"building": "yes", "roof:shape": "flat"}) is None
    assert roof_tags({"building": "yes", "roof:shape": "sawtooth"}) is None
    assert roof_tags({"building": "yes", "roof:levels": "1"}) is None  # no shape, so a flat slab
    assert roof_tags({"building": "yes"}) is None


def test_a_building_that_is_only_a_roof_is_left_to_the_pavilion_code():
    assert roof_tags({"building": "roof", "roof:shape": "hipped"}) is None


def test_the_roof_height_levels_direction_and_orientation_are_read():
    roof = roof_tags({"building": "yes", "roof:shape": "skillion", "roof:height": "4.5", "roof:levels": "2",
                      "roof:direction": "NE", "roof:orientation": "across"})
    assert (roof.height, roof.levels, roof.direction, roof.orientation) == (4.5, 2.0, 45.0, "across")
    empty = roof_tags({"building": "yes", "roof:shape": "gabled", "roof:height": "0", "roof:levels": "0", "roof:orientation": "sideways"})
    assert (empty.height, empty.levels, empty.orientation) == (None, None, None)  # nothing usable, or not a known word


def test_classify_gives_a_building_its_roof():
    lon, lat = world.to_lonlat(100.0, 100.0)
    lon2, lat2 = world.to_lonlat(112.0, 108.0)
    ring = [(lon, lat), (lon2, lat), (lon2, lat2), (lon, lat2), (lon, lat)]
    element = {
        "type": "way", "id": 5, "geometry": [{"lat": y, "lon": x} for x, y in ring],
        "tags": {"building": "yes", "height": "12", "roof:shape": "gabled", "roof:height": "3"},
    }
    (building,) = classify([element]).buildings
    assert building.roof.shape == "gable" and building.roof.height == 3.0 and building.height == 12.0


# --- shapes ---------------------------------------------------------------------------------------------------------


SHAPES = {
    "gabled": "gable", "hipped": "hip", "half-hipped": "half_hip", "pyramidal": "pyramid", "skillion": "skillion",
    "gambrel": "gambrel", "mansard": "mansard", "round": "barrel", "dome": "dome",
}


@pytest.mark.parametrize("word,shape", SHAPES.items())
def test_each_shape_covers_its_footprint_between_the_eaves_and_the_top(word, shape):
    footprint = box(0, 0, 20, 8)
    building = _building(footprint, {"roof:shape": word, "roof:height": "3"}, height=12.0)
    plan, roof, fill, _ = _draw(building)
    assert plan.shape == shape
    assert (plan.eave, plan.top) == (pytest.approx(9.0), pytest.approx(12.0))  # the walls stop 3 m under the top
    pts = _verts(roof)
    assert pts[:, 2].min() == pytest.approx(9.0, abs=1e-4) and pts[:, 2].max() == pytest.approx(12.0, abs=1e-4)
    # nothing sticks out of the footprint, and seen from above the roof leaves no hole and does not overlap itself
    assert all(footprint.buffer(1e-4).contains(Point(x, y)) for x, y, _ in pts)
    assert _area_from_above(roof) == pytest.approx(footprint.area, rel=1e-6)
    # every triangle faces up
    normals = np.cross(roof[:, 1] - roof[:, 0], roof[:, 2] - roof[:, 0])
    assert (normals[:, 2] > 0).all()
    if len(fill):
        assert fill[:, :, 2].min() >= 9.0 - 1e-4 and fill[:, :, 2].max() <= 12.0 + 1e-4


def test_a_gabled_roof_runs_its_ridge_along_the_long_side_and_its_ends_are_filled():
    building = _building(box(0, 0, 20, 8), {"roof:shape": "gabled", "roof:height": "3"}, height=12.0)
    plan, roof, fill, mesh = _draw(building)
    ridge = _verts(roof)[np.isclose(_verts(roof)[:, 2], 12.0, atol=1e-4)]
    assert np.allclose(ridge[:, 1], 4.0)  # in the middle of the 8 m span
    assert ridge[:, 0].min() == pytest.approx(0.0) and ridge[:, 0].max() == pytest.approx(20.0)  # the whole 20 m
    # the two gable ends are triangles 8 m wide and 3 m high, on the walls at x = 0 and x = 20
    assert _area(fill) == pytest.approx(2 * 0.5 * 8 * 3)
    assert set(np.round(_verts(fill)[:, 0], 6)) == {0.0, 20.0}
    # blank for the facade shader (no bays: u = 0), and in the building's feature
    assert (mesh.uvs == 0).all() and (mesh.features == 7).all()


def test_roof_orientation_across_turns_the_ridge_square_to_the_long_side():
    building = _building(box(0, 0, 20, 8), {"roof:shape": "gabled", "roof:height": "3", "roof:orientation": "across"}, height=12.0)
    _, roof, fill, _ = _draw(building)
    ridge = _verts(roof)[np.isclose(_verts(roof)[:, 2], 12.0, atol=1e-4)]
    assert np.allclose(ridge[:, 0], 10.0)  # now in the middle of the 20 m span
    assert ridge[:, 1].min() == pytest.approx(0.0) and ridge[:, 1].max() == pytest.approx(8.0)
    assert _area(fill) == pytest.approx(2 * 0.5 * 20 * 3)  # the gable ends are the long walls' share, 20 m wide


def test_a_ridge_follows_a_rotated_footprint():
    from shapely.affinity import rotate

    footprint = rotate(box(0, 0, 20, 8), 30, origin=(0, 0))
    _, roof, _, _ = _draw(_building(footprint, {"roof:shape": "gabled", "roof:height": "3"}, height=12.0))
    ridge = _verts(roof)[np.isclose(_verts(roof)[:, 2], 12.0, atol=1e-4)]
    direction = ridge.max(axis=0) - ridge.min(axis=0)
    assert np.degrees(np.arctan2(direction[1], direction[0])) == pytest.approx(30.0, abs=0.5)


def test_a_hipped_ridge_is_shorter_than_the_building_by_its_width():
    _, roof, fill, _ = _draw(_building(box(0, 0, 20, 8), {"roof:shape": "hipped", "roof:height": "3"}, height=12.0))
    ridge = _verts(roof)[np.isclose(_verts(roof)[:, 2], 12.0, atol=1e-4)]
    assert ridge[:, 0].min() == pytest.approx(4.0) and ridge[:, 0].max() == pytest.approx(16.0)  # 20 - 8 = 12 m long
    assert len(fill) == 0  # a hip slopes down to the eaves on every side: no wall above them


def test_a_pyramid_comes_to_a_point_over_the_middle():
    _, roof, fill, _ = _draw(_building(box(0, 0, 10, 10), {"roof:shape": "pyramidal", "roof:height": "6"}, height=30.0))
    apex = _verts(roof)[np.isclose(_verts(roof)[:, 2], 30.0, atol=1e-4)]
    assert len(roof) == 4 and np.allclose(apex[:, :2], [5.0, 5.0]) and len(fill) == 0


def test_a_pyramid_over_a_many_sided_footprint_is_a_fan_that_stays_inside_it():
    ring = Point(0, 0).buffer(4.0, quad_segs=8)
    _, roof, _, _ = _draw(_building(ring, {"roof:shape": "pyramidal", "roof:height": "5"}, height=20.0))
    assert len(roof) == len(ring.exterior.coords) - 1
    assert _area_from_above(roof) == pytest.approx(ring.area, rel=1e-6)


def test_a_skillion_roof_slopes_down_the_way_the_direction_says():
    footprint = box(0, 0, 20, 8)
    east = _draw(_building(footprint, {"roof:shape": "skillion", "roof:height": "2", "roof:direction": "90"}, height=10.0))
    pts = _verts(east[1])
    assert np.allclose(pts[np.isclose(pts[:, 0], 20.0)][:, 2], 8.0)  # down at the east edge: the eaves
    assert np.allclose(pts[np.isclose(pts[:, 0], 0.0)][:, 2], 10.0)  # up at the west edge
    high = lambda fill: _verts(fill)[_verts(fill)[:, 2] > 8.0 + 1e-6]  # the corners of the fill that rise above the eaves
    assert set(np.round(high(east[2])[:, 0], 6)) == {0.0}  # the roof meets the eaves at its low edge: only the high side has wall to fill
    west = _draw(_building(footprint, {"roof:shape": "skillion", "roof:height": "2", "roof:direction": "270"}, height=10.0))
    assert np.allclose(_verts(west[1])[np.isclose(_verts(west[1])[:, 0], 0.0)][:, 2], 8.0)
    assert set(np.round(high(west[2])[:, 0], 6)) == {20.0}


def test_a_skillion_roof_with_no_direction_slopes_across_the_short_side():
    plan, roof, _, _ = _draw(_building(box(0, 0, 20, 8), {"roof:shape": "skillion", "roof:height": "2"}, height=10.0))
    pts = _verts(roof)
    assert np.isclose(pts[np.isclose(pts[:, 1], 0.0)][:, 2], 8.0).all()  # south, where it is lowest
    assert np.isclose(pts[np.isclose(pts[:, 1], 8.0)][:, 2], 10.0).all()


def test_a_mansard_has_a_steep_lower_slope_and_a_flat_top():
    footprint = box(0, 0, 20, 8)
    plan, roof, _, _ = _draw(_building(footprint, {"roof:shape": "quadruple_saltbox", "roof:levels": "1"}, height=14.0, source="levels"))
    assert plan.shape == "mansard"
    top = _verts(roof)[np.isclose(_verts(roof)[:, 2], plan.top, atol=1e-4)]
    inset = top[:, :2].min(axis=0)
    assert 0.5 < inset[0] < 2.5 and np.allclose(inset, inset[0])  # the flat top is set in from every edge by the same amount
    flat = [t for t in roof if np.allclose(t[:, 2], plan.top, atol=1e-4)]
    assert _area_from_above(np.array(flat)) == pytest.approx((20 - 2 * inset[0]) * (8 - 2 * inset[0]), rel=1e-6)


def _mansard(polygon):
    return _building(polygon, {"roof:shape": "quadruple_saltbox", "roof:levels": "1"}, height=3 * 3.4 + ROOF_ALLOWANCE, source="levels")


def test_a_mansard_does_not_slope_down_toward_a_neighbour_it_stands_against():
    row = [box(0, 0, 6, 20), box(6, 0, 12, 20), box(12, 0, 18, 20)]  # three rowhouses, front and back to the street and the alley
    plan, roof, fill, _ = _draw(_mansard(row[1]))
    on = lambda tris, x: _verts(tris)[np.isclose(_verts(tris)[:, 0], x)]
    assert on(roof, 6.0)[:, 2].min() == pytest.approx(plan.eave)  # alone, it slopes down to every wall

    plan, roof, fill, _ = _draw(_mansard(row[1]), neighbours=[row[0], row[2]])
    assert on(roof, 6.0)[np.isclose(on(roof, 6.0)[:, 1], 10.0)][:, 2] == pytest.approx(plan.top)  # full height at the party wall, mid-way
    # the party walls are filled up to the roof (the neighbour's own fill is the other face of the same wall) ...
    assert set(np.round(_verts(fill)[_verts(fill)[:, 2] > plan.eave + 1e-6][:, 0], 6)) == {6.0, 12.0}
    # ... and the roof still slopes down to the street and the alley
    front = _verts(roof)[np.isclose(_verts(roof)[:, 1], 0.0)]
    assert front[:, 2].max() == pytest.approx(plan.eave) and _verts(roof)[:, 2].max() == pytest.approx(plan.top)
    assert _area_from_above(roof) == pytest.approx(row[1].area, rel=1e-6)


def test_the_end_of_a_row_keeps_its_slope_on_the_open_side():
    row = [box(0, 0, 6, 20), box(6, 0, 12, 20)]
    plan, roof, fill, _ = _draw(_mansard(row[1]), neighbours=[row[0]])
    on = lambda x: _verts(roof)[np.isclose(_verts(roof)[:, 0], x)][:, 2]
    assert on(12.0).min() == pytest.approx(plan.eave)  # the open side slopes down to the eaves
    assert on(6.0).max() == pytest.approx(plan.top) and on(6.0).min() > plan.eave - 1e-6  # the side against its neighbour does not
    far = _draw(_mansard(row[1]), neighbours=[box(30, 0, 36, 20)])  # a building that is not touching changes nothing
    assert np.array_equal(far[1], _draw(_mansard(row[1]))[1])


def test_only_a_mansard_gives_way_to_a_neighbour():
    row = [box(0, 0, 6, 20), box(6, 0, 12, 20), box(12, 0, 18, 20)]
    hipped = _building(row[1], {"roof:shape": "hipped", "roof:height": "3"}, height=12.0)
    assert np.array_equal(_draw(hipped, neighbours=[row[0], row[2]])[1], _draw(hipped)[1])


def test_a_gambrel_roof_is_steeper_low_down_than_high_up():
    plan, roof, _, _ = _draw(_building(box(0, 0, 20, 8), {"roof:shape": "gambrel", "roof:height": "4"}, height=12.0))
    ys = np.unique(np.round(_verts(roof)[:, 1], 4))
    zs = {round(y, 4): _verts(roof)[np.isclose(_verts(roof)[:, 1], y)][:, 2].max() for y in ys}
    knee = [y for y in ys if 0 < y < 4][0]  # a vertex between the wall and the ridge
    low = (zs[knee] - zs[0.0]) / knee  # slope of the lower part
    high = (zs[4.0] - zs[knee]) / (4.0 - knee)
    assert low > 2 * high


def test_a_round_roof_is_an_arch_across_its_span_and_a_dome_a_cap_over_its_footprint():
    plan, roof, _, _ = _draw(_building(box(0, 0, 20, 8), {"roof:shape": "round", "roof:height": "3"}, height=12.0))
    ridge = _verts(roof)[np.isclose(_verts(roof)[:, 2], 12.0, atol=1e-4)]
    assert np.allclose(ridge[:, 1], 4.0)  # along the long side, in the middle
    heights = sorted(set(np.round(_verts(roof)[:, 2], 3)))
    assert len(heights) >= 5  # an arch, not a tent
    ring = Point(0, 0).buffer(5.0, quad_segs=8)
    plan, dome, fill, _ = _draw(_building(ring, {"roof:shape": "dome", "roof:height": "4"}, height=20.0))
    assert plan.kind == "dome" and len(fill) == 0
    centre = _verts(dome)[np.isclose(_verts(dome)[:, 2], 20.0, atol=1e-4)]
    assert np.allclose(centre[:, :2], [0.0, 0.0], atol=1e-6)
    radius = lambda z: np.hypot(*_verts(dome)[np.isclose(_verts(dome)[:, 2], z, atol=1e-3)][:, :2].T).max()
    assert radius(16.0) == pytest.approx(5.0, abs=1e-3)  # the foot of the dome is the footprint
    middle, upper = 16.0 + 4.0 * np.cos(np.radians(65.0)), 16.0 + 4.0 * np.cos(np.radians(40.0))  # the dome's rings
    assert radius(16.0) > radius(middle) > radius(upper) > 0  # and it narrows as it climbs


def test_a_half_hip_is_a_gable_hipped_over_its_upper_half():
    plan, roof, fill, _ = _draw(_building(box(0, 0, 20, 8), {"roof:shape": "half-hipped", "roof:height": "4"}, height=12.0))
    ridge = _verts(roof)[np.isclose(_verts(roof)[:, 2], 12.0, atol=1e-4)]
    assert ridge[:, 0].min() == pytest.approx(2.0) and ridge[:, 0].max() == pytest.approx(18.0)  # a quarter of the span short of each end
    assert _verts(fill)[:, 2].max() == pytest.approx(10.0, abs=1e-4)  # the end wall goes up half the rise


# --- heights ------------------------------------------------------------------------------------------------------------


def test_the_eaves_are_the_height_less_the_roof_height():
    for tags, rise in (({"roof:height": "3"}, 3.0), ({"roof:levels": "2"}, 2 * ROOF_LEVEL_HEIGHT)):
        plan = plan_roof(_building(box(0, 0, 20, 8), {"roof:shape": "gabled", **tags}, height=15.0))
        assert plan.top == pytest.approx(15.0) and plan.top - plan.eave == pytest.approx(rise)


def test_a_building_known_by_its_levels_has_walls_of_that_many_storeys_with_the_roof_on_top():
    building = _building(box(0, 0, 20, 8), {"roof:shape": "gabled", "roof:levels": "1"}, height=3 * 3.4 + ROOF_ALLOWANCE, source="levels")
    plan = plan_roof(building)
    assert plan.eave == pytest.approx(3 * 3.4)  # three storeys, without the allowance a flat roof's parapet gets
    assert plan.top == pytest.approx(3 * 3.4 + ROOF_LEVEL_HEIGHT)


def test_without_a_roof_height_a_pitched_roof_has_about_a_thirty_degree_slope():
    plan = plan_roof(_building(box(0, 0, 20, 8), {"roof:shape": "gabled"}, height=12.0))
    rise = plan.top - plan.eave
    assert rise == pytest.approx(np.tan(np.radians(30)) * 4, rel=1e-3)
    wide = plan_roof(_building(box(0, 0, 60, 40), {"roof:shape": "gabled"}, height=20.0))
    assert wide.top - wide.eave <= 4.5 + 1e-9  # a big building's roof is not a tower of slate


def test_a_roof_cannot_start_below_the_base_of_a_building_part():
    part = _building(box(0, 0, 4, 4), {"roof:shape": "pyramidal", "roof:height": "12", "building:part": "yes"}, height=66.0, min_height=60.0)
    plan = plan_roof(part)
    assert plan.eave == pytest.approx(60.0) and plan.top == pytest.approx(66.0)  # a spire sits on the part below it


# --- when nothing is drawn --------------------------------------------------------------------------------------------


def test_a_flat_roof_and_an_untagged_building_keep_their_slab():
    assert plan_roof(_building(box(0, 0, 20, 8), {"roof:shape": "flat"})) is None
    assert plan_roof(_building(box(0, 0, 20, 8))) is None


def test_a_footprint_too_irregular_for_a_ridge_roof_is_left_flat():
    u_shape = Polygon([(0, 0), (20, 0), (20, 20), (14, 20), (14, 6), (6, 6), (6, 20), (0, 20)])
    assert u_shape.area / 400 < MIN_FILL
    assert plan_roof(_building(u_shape, {"roof:shape": "gabled", "roof:height": "3"})) is None
    assert plan_roof(_building(u_shape, {"roof:shape": "hipped", "roof:height": "3"})) is None


def test_a_notched_footprint_still_gets_a_roof_that_stays_inside_it():
    notched = Polygon([(0, 0), (20, 0), (20, 10), (14, 10), (14, 8), (0, 8)])
    assert notched.area / 200 > MIN_FILL
    plan, roof, fill, _ = _draw(_building(notched, {"roof:shape": "gabled", "roof:height": "3"}, height=12.0))
    assert plan is not None
    assert _area_from_above(roof) == pytest.approx(notched.area, rel=1e-6)
    assert all(notched.buffer(1e-4).contains(Point(x, y)) for x, y, _ in _verts(np.concatenate([roof, fill])))


def test_a_footprint_with_a_courtyard_gets_a_roof_with_no_hole_or_spike_in_it():
    # A turned square with a square courtyard (found by fuzzing): clipping the hip's planes to the frame left a facet
    # folded over itself, which made the footprint cut come out three times too large.
    footprint = Polygon(
        [(0, 0), (4.099, 19.575), (23.674, 15.476), (19.575, -4.099)],
        [[(12.276, 8.651), (4.357, 10.309), (2.973, 3.702), (10.893, 2.044)]],
    )
    building = _building(footprint, {"roof:shape": "hipped", "roof:height": "6.4"}, height=36.0, source="levels")
    plan, roof, fill, _ = _draw(building)
    assert plan is not None
    assert _area_from_above(roof) == pytest.approx(footprint.area, rel=1e-3)
    assert _verts(roof)[:, 2].min() >= plan.eave - 1e-6 and _verts(roof)[:, 2].max() <= plan.top + 1e-6


def test_a_roof_whose_surface_does_not_cover_its_footprint_is_not_drawn():
    building = _building(box(0, 0, 20, 8), {"roof:shape": "hipped", "roof:height": "3"}, height=12.0)
    plan = plan_roof(building)
    assert _sound(plan, building.polygon)
    plan.surface = plan.surface[1:]  # a hole in the roof
    assert not _sound(plan, building.polygon)
    plan.surface = plan.surface + np.array([0.0, 0.0, 5.0])  # or a roof above its own top
    assert not _sound(plan, building.polygon)
    plan.surface = np.zeros((0, 3, 3))
    assert not _sound(plan, building.polygon)


def test_degenerate_footprints_keep_the_flat_slab():
    sliver = box(0, 0, 20, 0.3)
    assert sliver.area > 2 and 0.3 < MIN_SPAN
    assert plan_roof(_building(sliver, {"roof:shape": "gabled", "roof:height": "2"})) is None
    assert plan_roof(_building(Polygon([(0, 0), (1, 0), (2, 0)]), {"roof:shape": "gabled"})) is None  # no area
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])  # self-intersecting
    assert plan_roof(_building(bowtie, {"roof:shape": "gabled", "roof:height": "2"})) is None
    ring = box(0, 0, 20, 20).difference(box(5, 5, 15, 15))  # a courtyard: a fan from one point would cover the hole
    assert plan_roof(_building(ring, {"roof:shape": "pyramidal", "roof:height": "5"})) is None
    crescent = Point(0, 0).buffer(10).difference(Point(6, 0).buffer(9))
    assert plan_roof(_building(crescent, {"roof:shape": "dome", "roof:height": "5"})) is None  # no point sees all of it


def test_a_roof_too_low_to_show_is_left_flat():
    assert 0.1 < MIN_RISE
    assert plan_roof(_building(box(0, 0, 20, 8), {"roof:shape": "gabled", "roof:height": "0.1"})) is None


def test_roofs_are_deterministic():
    building = _building(box(0, 0, 20, 8), {"roof:shape": "gambrel", "roof:height": "4"}, height=12.0)
    first, second = _draw(building), _draw(building)
    assert np.array_equal(first[1], second[1]) and np.array_equal(first[2], second[2])
    assert plan_roof(building).eave == plan_roof(building).eave


# --- colours --------------------------------------------------------------------------------------------------------


def test_a_roof_takes_its_colour_from_the_colour_tag_then_the_material():
    tags = {"building": "yes", "roof:shape": "gabled"}
    assert building_colours("w1", {**tags, "roof:colour": "#112233", "roof:material": "slate"}, 12)[1] == "#112233"
    assert building_colours("w1", {**tags, "roof:material": "copper"}, 12)[1] == ROOF_MATERIAL_COLOURS["copper"]
    assert building_colours("w1", {**tags, "roof:material": "roof_tiles"}, 12)[1] == ROOF_MATERIAL_COLOURS["roof_tiles"]
    assert building_colours("w1", {**tags, "roof:material": "unobtainium"}, 12)[1] in SHINGLE_ROOFS + SLATE_ROOFS


def test_a_pitched_roof_with_no_colour_is_slate_on_a_church_and_shingle_on_a_house():
    for ident in ("w1", "w2", "w3", "w4"):
        church = building_colours(ident, {"building": "church", "roof:shape": "gabled"}, 20)[1]
        house = building_colours(ident, {"building": "house", "roof:shape": "gabled", "building:colour": "#d9d0b8"}, 9)[1]
        assert church in SLATE_ROOFS and house in SHINGLE_ROOFS
        assert building_colours(ident, {"building": "yes", "roof:shape": "quadruple_saltbox"}, 14)[1] in SLATE_ROOFS  # a mansard


def test_a_flat_roof_with_no_colour_is_still_one_of_the_greys():
    assert building_colours("w1", {"building": "yes"}, 12)[1] in ROOFS
    assert building_colours("w1", {"building": "yes", "roof:shape": "flat"}, 12)[1] in ROOFS


# --- in a tile ---------------------------------------------------------------------------------------------------------


def _layers(tmp_path):
    data = (tmp_path / "tiles" / "0_0.glb").read_bytes()
    json_length, _ = struct.unpack_from("<I4s", data, 12)
    document = json.loads(data[20 : 20 + json_length])
    binary = data[28 + json_length :]

    def attribute(layer, name):
        primitive = [m for m in document["meshes"] if m["name"] == layer][0]["primitives"][0]
        accessor = document["accessors"][primitive["attributes"][name]]
        view = document["bufferViews"][accessor["bufferView"]]
        width = {"POSITION": 3, "TEXCOORD_0": 2}[name]
        return np.frombuffer(binary, np.float32, accessor["count"] * width, view["byteOffset"]).reshape(-1, width)

    return document, attribute


def test_a_tile_draws_walls_to_the_eaves_and_the_roof_above_them(tmp_path):
    scene = Scene()
    scene.buildings.append(_building(box(100, 100, 120, 108), {"roof:shape": "gabled", "roof:height": "3"}, height=12.0))
    scene.buildings.append(Building("w2", box(150, 100, 160, 108), 12.0, 0.0, "height", "#8a4b3c", "#5a5a5c", {"building": "yes"}))  # flat
    build(scene, [(0, 0)], tmp_path, {"name": "Test"}, None)
    document, attribute = _layers(tmp_path)
    roofs = attribute("building_roofs", "POSITION")
    walls, uv = attribute("building_walls", "POSITION"), attribute("building_walls", "TEXCOORD_0")
    x = roofs[:, 0]
    pitched, flat = roofs[x < 140], roofs[x > 140]
    assert pitched[:, 1].max() == pytest.approx(12.0, abs=1e-3) and pitched[:, 1].min() == pytest.approx(9.0, abs=1e-3)
    assert np.allclose(flat[:, 1], 12.0)  # the other building keeps its slab
    patterned = uv[:, 0] > 0  # facade walls; the gable fill is blank
    assert walls[patterned][walls[patterned][:, 0] < 140][:, 1].max() == pytest.approx(9.0, abs=1e-3)  # they stop at the eaves
    assert walls[~patterned][:, 1].max() == pytest.approx(12.0, abs=1e-3)  # the gable ends reach the ridge
    triangles = sum(e["extras"]["triangles"] for e in json.loads((tmp_path / "tileset.json").read_text())["root"]["children"])
    assert triangles > 4 + 2 * 4 + 2 * 2  # more than two boxes with slabs

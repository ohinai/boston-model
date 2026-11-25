import math

import numpy as np
import pytest

from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.trees import CROWN_BASE, EVERGREEN_FORMS, FORMS, TreeShape, _assemble, _lobe, add_tree, describe

# The most triangles any tree may take. A crown is a skin with lobes seated on it, and the detail grows with its
# width (a sapling takes about 40, a big elm about 200, the biggest weeping willow, which has hundreds of strands, 600).
MAX_TRIANGLES = 600

CM = 100 / math.pi  # girth (m) to diameter (cm): d = circumference * CM


def girth(d_cm):
    return str(d_cm / CM)


def shape(form, height=15.0, width=None, trunk=0.25):
    base = CROWN_BASE[form] * height
    if form not in EVERGREEN_FORMS:
        base = max(base, min(2.2, 0.45 * height))
    return TreeShape(form, height, width or 0.8 * height, base, trunk, "#55803c", "#5b4a3c", None, True)


def build(s, x=3.0, y=-4.0, seed=0, feature=0):
    builder = MeshBuilder("trees", colours=True, features=True)
    add_tree(builder, s, x, y, seed, feature)
    return builder.finish()


def enu(mesh):
    p = mesh.positions
    return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])


# --- describe -----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("species,d,expected", [("Tilia cordata", 40, 14), ("Tilia cordata", 90, 20), ("Malus", 20, 5.5)])
def test_checkpoint_heights(seed, species, d, expected):
    got = describe({"species": species, "circumference": girth(d)}, seed).height
    assert got == pytest.approx(expected, rel=0.10)


def test_tags_win_and_set_measured():
    s = describe({"species": "Tilia", "height": "12 m", "diameter_crown": "9", "circumference": "1.0"})
    assert (s.height, s.crown_width, s.measured) == (12, 9, True)
    assert describe({"height": "40'"}).height == pytest.approx(12.19, abs=0.01)
    # Implausible tags are ignored and the height is estimated.
    s = describe({"species": "Tilia", "height": "400", "diameter_crown": "300", "circumference": "1.0"})
    assert not s.measured and s.height < 30 and s.crown_width < 40


def test_crown_only_tree_is_sized_from_the_crown_not_a_default_trunk():
    s = describe({"natural": "tree", "diameter_crown": "3"})
    assert s.height == pytest.approx(4.0, abs=0.01)  # 3 / 0.75, not the ~13 m a default trunk gives
    assert s.crown_width == 3 and not s.measured
    assert describe({"diameter_crown": "40"}).height == 18  # capped at mature
    assert describe({"diameter_crown": "1"}).height >= 2.0
    # A mapped height still wins, and a trunk size keeps the old estimate.
    assert describe({"diameter_crown": "3", "height": "9"}).height == 9
    with_girth = {"diameter_crown": "3", "circumference": "1.0"}
    assert describe(with_girth, 5).height == describe({"circumference": "1.0"}, 5).height > 8


def test_genus_from_species_taxon_or_tag_case_insensitively():
    assert describe({"species": "ULMUS americana"}).form == "vase"
    assert describe({"taxon": "tilia cordata"}).form == "oval"
    assert describe({"genus": "salix"}).form == "weeping"
    assert describe({"genus": "Quercus", "species": "Ulmus americana"}).name == "Elm"  # the species says more


def test_species_exceptions_win_over_genus():
    assert describe({"species": "Quercus palustris"}).form == "oval"
    assert describe({"species": "Quercus rubra"}).form == "round"
    assert describe({"species": "Acer palmatum 'Bloodgood'"}).form == "small"
    assert describe({"species": "Acer palmatum"}).name == "Japanese maple"
    assert describe({"species": "Acer ginnala"}).name == "Maple"
    assert describe({"species": "Ulmus parvifolia"}).form == "round"


def test_unknown_genus_fallbacks():
    assert describe({"genus": "Xyzzy", "leaf_type": "needleleaved"}).form == "conifer"
    assert describe({"leaf_type": "needleleaved"}).name is None
    assert describe({}).form == "round" and describe({}).name is None
    assert describe({"species": "Xyzzy plugh"}).form == "round"


def test_common_name_prefers_species_en():
    assert describe({"species": "Tilia cordata", "species:en": "Small-leaved lime"}).name == "Small-leaved lime"
    assert describe({"species": "Tilia cordata"}).name == "Linden"


def test_bigger_trunk_is_taller_monotonically():
    heights = [describe({"species": "Quercus rubra", "circumference": girth(d)}, 3).height for d in (10, 20, 40, 80, 160)]
    assert heights == sorted(heights) and len(set(heights)) == len(heights)


def test_park_tree_is_bigger_than_street_tree_without_trunk_data():
    park = describe({"species": "Tilia", "denotation": "park"}, 1)
    street = describe({"species": "Tilia", "denotation": "street"}, 1)
    assert park.height > street.height and park.crown_width > street.crown_width
    assert park.trunk_radius > street.trunk_radius


def test_describe_is_deterministic_and_seed_varies_assumed_values():
    tags = {"species": "Pyrus calleryana"}
    assert describe(tags, 5) == describe(tags, 5)
    assert len({describe(tags, s).height for s in range(10)}) > 3
    describe(tags, -7)


def test_derived_limits():
    for seed in range(20):
        s = describe({"species": "Malus", "circumference": "0.2"}, seed)
        assert s.height >= 2 and 1.2 <= s.crown_width <= 0.28 * (0.2 * CM) + 1.5 + 1e-6
        assert 0.05 <= s.trunk_radius <= 1.2 and s.crown_base < s.height
    assert describe({"species": "Tilia"}).crown_base >= min(2.2, 0.45 * describe({"species": "Tilia"}).height) - 1e-9
    assert describe({"species": "Picea"}).crown_base == pytest.approx(0.12 * describe({"species": "Picea"}).height, abs=0.01)


# --- geometry -----------------------------------------------------------------------------------

SHAPES = [(f, h) for f in FORMS for h in (4.0, 15.0, 25.0)]


@pytest.mark.parametrize("form,height", SHAPES)
def test_limits_ground_budget_and_feature(form, height):
    s = shape(form, height, trunk=0.1 + height / 60)
    for seed in range(5):
        mesh = build(s, seed=seed, feature=9)
        assert len(mesh.indices) > 0 and len(mesh.indices) // 3 <= MAX_TRIANGLES
        east, north, up = enu(mesh).T
        assert up.min() >= -1e-6 and up.max() <= height * 1.03
        assert np.hypot(east - 3.0, north + 4.0).max() <= s.crown_width / 2 * 1.25 + 1e-6
        assert (mesh.features == 9).all()
        assert mesh.colours.shape == (len(mesh.positions), 4)
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("form", FORMS)
def test_every_form_reaches_near_its_height_and_is_young_tree_single_lobe(form):
    s = shape(form, 15.0)
    assert enu(build(s))[:, 2].max() > 15.0 * 0.9
    young = shape(form, 6.0, trunk=0.06)
    kinds = [p.kind for p in _assemble(young)]
    # A sapling is a trunk and one blob. Evergreens keep their own shapes at every age: a small cone, a column, a dome.
    expected = {"conifer": ["trunk"] + ["cone"] * (len(kinds) - 1), "clipped": ["canopy"]}.get(form, ["trunk", "canopy"])
    assert kinds == expected and (form != "conifer" or len(kinds) >= 4)


def test_triangle_counts_are_in_the_expected_range():
    for form in FORMS:
        count = len(build(shape(form)).indices) // 3
        assert 40 <= count <= MAX_TRIANGLES, (form, count)


@pytest.mark.parametrize("form", [f for f in FORMS if f != "conifer"])
def test_canopy_faces_wind_away_from_the_lobe_centre(form):
    for seed in range(5):
        for part in _assemble(shape(form), seed):
            if part.kind != "canopy":
                continue
            a, b, c = (part.positions[part.faces[:, i]] for i in range(3))
            outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3 - part.centre)
            assert (outward > 0).all()
            facing = np.einsum("ij,ij->i", part.normals[part.faces[:, 0]], (a + b + c) / 3 - part.centre)
            assert (facing > 0).all()


def test_lobe_is_an_indexed_icosahedron_and_deterministic():
    p, n, f = _lobe((1, 2, 3), (2, 2, 3), 7)
    assert p.shape == (12, 3) and f.shape == (20, 3) and np.allclose(np.linalg.norm(n, axis=1), 1)
    assert np.array_equal(p, _lobe((1, 2, 3), (2, 2, 3), 7)[0])
    assert not np.array_equal(p, _lobe((1, 2, 3), (2, 2, 3), 8)[0])
    assert _lobe((0, 0, 0), (1, 1, 1), 1, coarse=True)[2].shape == (8, 3)
    p, _, _ = _lobe((0, 0, 5), (2, 2, 3), -4)  # negative seeds fine; extent follows radii (+-12%)
    assert p[:, 2].max() <= 5 + 3 * 1.12 + 1e-9 and p[:, 2].min() >= 5 - 3 * 1.12 - 1e-9


@pytest.mark.parametrize("form", [f for f in FORMS if f != "clipped"])  # a clipped dome stands on the ground, no trunk
def test_trunk_faces_point_away_from_the_axis(form):
    trunk = _assemble(shape(form))[0]
    assert trunk.kind == "trunk"
    a, b, c = (trunk.positions[trunk.faces[:, i]] for i in range(3))
    centroid = (a + b + c) / 3
    radial = np.column_stack([centroid[:, 0], centroid[:, 1], np.zeros(len(centroid))])
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), radial) > 0).all()


@pytest.mark.parametrize("form", [f for f in FORMS if f != "conifer"])
def test_canopy_gets_lighter_with_height(form):
    parts = [p for p in _assemble(shape(form, 18.0), 2) if p.kind == "canopy"]
    z = np.concatenate([p.positions[:, 2] for p in parts])
    brightness = np.concatenate([p.colours[:, :3].sum(axis=1, dtype=float) for p in parts])
    assert np.corrcoef(z, brightness)[0, 1] > 0.5
    assert brightness[z > np.percentile(z, 75)].mean() > brightness[z < np.percentile(z, 25)].mean()


def test_conifer_is_a_stack_of_tiers_and_pointed():
    parts = _assemble(shape("conifer", 14.0))
    assert sum(p.kind == "cone" for p in parts) >= 4
    assert max(p.positions[:, 2].max() for p in parts) == pytest.approx(14.0)


FOLIAGE = ("canopy", "cone", "skirt")


def foliage_z(s, seed=0):
    return np.concatenate([p.positions[:, 2] for p in _assemble(s, seed) if p.kind in FOLIAGE])


def foliage_radius_at(s, z, seed=0):
    """Widest distance from the trunk where foliage triangles cross the height z."""
    best = 0.0
    for part in _assemble(s, seed):
        if part.kind not in FOLIAGE:
            continue
        for face in part.faces:
            pts = part.positions[face]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                za, zb = pts[a, 2], pts[b, 2]
                if (za - z) * (zb - z) <= 0 and za != zb:
                    q = pts[a] + (z - za) / (zb - za) * (pts[b] - pts[a])
                    best = max(best, math.hypot(q[0], q[1]))
    return best


def test_vase_foliage_is_higher_than_round_and_trunk_and_limbs_show():
    for seed in range(5):
        assert foliage_z(shape("vase", 20.0), seed).min() > foliage_z(shape("round", 20.0), seed).min() + 1.0
    parts = _assemble(shape("vase", 20.0))
    assert sum(p.kind == "limb" for p in parts) >= 3
    # The trunk forks just under the crown, so the limbs show from below.
    assert parts[0].positions[:, 2].max() == pytest.approx(shape("vase", 20.0).crown_base, abs=0.02 * 20.0)


# Changed: the weeping form is now a dome plus a hanging skirt (not lobes plus narrow cones), so
# "foliage" includes the skirt and the lowest foliage is its hem.
def test_weeping_foliage_reaches_low():
    for seed in range(5):
        assert 1.0 <= foliage_z(shape("weeping", 15.0), seed).min() <= 3.2


@pytest.mark.parametrize("height", [8.0, 15.0, 22.0])
def test_weeping_is_one_broad_drooping_mound_not_a_cone(height):
    s = shape("weeping", height, width=1.0 * height, trunk=0.3)
    for seed in range(6):
        parts = _assemble(s, seed)
        pos = np.concatenate([p.positions for p in parts if p.kind in FOLIAGE])
        widest = 2 * np.hypot(pos[:, 0], pos[:, 1]).max()
        assert widest >= 0.85 * s.crown_width and widest <= 1.1 * s.crown_width + 1e-6
        assert 1.0 <= pos[:, 2].min() <= 3.2
        assert 2 * foliage_radius_at(s, 0.45 * height, seed) >= 0.7 * s.crown_width
        assert pos[:, 2].max() == pytest.approx(height, abs=0.05 * height)


def test_weeping_hem_is_ragged_and_skirt_is_visible_from_inside():
    s = shape("weeping", 15.0, trunk=0.3)
    parts = _assemble(s, 4)
    skirt = [p for p in parts if p.kind == "skirt"]
    assert len(skirt) == 2  # an outer ring of strands and an inner one
    for p in skirt:
        hem = np.unique(np.round(p.positions[p.positions[:, 2] < 4, 2], 3))
        assert len(hem) >= 5 and hem.min() >= 1.2 - 1e-6 and hem.max() <= 3.0 + 1e-6
        # The first half faces away from the axis, the second half is the same strands facing in, a hand's width nearer
        # the trunk so the two never share a plane (sharing one made the shadow map stripe them).
        half = len(p.faces) // 2

        def signed(face):
            a, b, c = p.positions[face]
            return np.cross(b - a, c - a)

        for out, back in zip(p.faces[:half], p.faces[half:]):
            a, b, c = p.positions[out]
            centroid = (a + b + c) / 3
            assert np.dot(signed(out), [centroid[0], centroid[1], 0]) > 0
            assert np.dot(signed(back), [centroid[0], centroid[1], 0]) < 0
            gap = np.hypot(*p.positions[out, :2].T).mean() - np.hypot(*p.positions[back, :2].T).mean()
            assert 0.01 < gap < 0.2
        top, bottom = p.positions[:, 2] > 5, p.positions[:, 2] < 5
        assert np.hypot(*p.positions[top, :2].T).mean() > np.hypot(*p.positions[bottom, :2].T).mean()


def test_weeping_trunk_rises_to_half_height_with_limbs_and_colours_darken_downward():
    s = shape("weeping", 15.0, trunk=0.3)
    parts = _assemble(s, 1)
    assert parts[0].kind == "trunk" and parts[0].positions[:, 2].max() == pytest.approx(0.4 * 15.0)
    assert sum(p.kind == "limb" for p in parts) == 4
    skirt = next(p for p in parts if p.kind == "skirt")
    z = skirt.positions[:, 2]
    brightness = skirt.colours[:, :3].sum(axis=1, dtype=float)
    assert brightness[z > 8].mean() > brightness[z < 4].mean()


def test_weeping_geometry_is_deterministic_and_within_budget_for_big_and_small_trees():
    for height, trunk in ((15.0, 0.3), (6.0, 0.15), (6.0, 0.06)):
        s = shape("weeping", height, width=height, trunk=trunk)
        a, b = build(s, seed=2), build(s, seed=2)
        assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)
        assert len(a.indices) // 3 <= MAX_TRIANGLES
        assert not np.array_equal(a.positions, build(s, seed=3).positions)


@pytest.mark.parametrize("tags", [{"genus": "Salix", "circumference": "2.5"}, {"species": "Salix alba", "circumference": "3.5"}])
def test_willow_is_big_and_at_least_as_wide_as_tall(tags):
    for seed in range(8):
        s = describe(tags, seed)
        assert s.form == "weeping" and s.crown_width >= 0.9 * s.height
        assert 1.5 <= s.crown_base <= 2.0
    mid = describe({"genus": "Salix", "circumference": "2.5"}, 0)
    assert 13.0 <= mid.height <= 16.5 and 13.0 <= mid.crown_width <= 17.0


def test_weeping_in_the_name_gives_the_weeping_form_whatever_the_genus():
    cherry = describe({"genus": "Prunus", "species:en": "Weeping Higan cherry", "circumference": "1.5"})
    assert cherry.form == "weeping" and cherry.height < 9 and cherry.name == "Weeping Higan cherry"
    assert cherry.leaf == "#55823f"
    assert describe({"genus": "Prunus", "species:en": "Higan cherry"}).form == "small"
    beech = describe({"species": "Fagus sylvatica", "species:en": "Weeping European beech", "circumference": "2.5"})
    assert beech.form == "weeping" and beech.height > 15
    assert describe({"species": "Fagus sylvatica 'Pendula'"}).form == "weeping"
    assert describe({"species": "Fagus sylvatica"}).form == "round"
    assert describe({"species:en": "WEEPING willow", "genus": "Salix"}).form == "weeping"


def test_small_tree_is_as_wide_as_it_is_tall():
    for seed in range(8):
        east, north, _ = enu(build(shape("small", 5.0, width=5.0, trunk=0.15), x=0, y=0, seed=seed)).T
        assert max(np.ptp(east), np.ptp(north), np.ptp(east + north) / 1.4142, np.ptp(east - north) / 1.4142) >= 4.0


def test_same_inputs_same_mesh_and_seed_changes_it():
    s = shape("round")
    a, b = build(s, seed=3), build(s, seed=3)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)
    assert not np.array_equal(a.positions, build(s, seed=4).positions)


def test_described_trees_build_within_budget():
    for species in ("Ulmus americana", "Salix alba", "Pinus strobus", "Malus", "Gleditsia triacanthos", "Pyrus calleryana"):
        for d in (8, 30, 100):
            s = describe({"species": species, "circumference": girth(d)}, 1)
            mesh = build(s)
            assert len(mesh.indices) // 3 <= MAX_TRIANGLES
            assert enu(mesh)[:, 2].max() <= s.height * 1.03


def test_species_wins_over_a_genus_tag_that_disagrees():
    from boston_pipeline.trees import describe

    # A fix that corrects only the species must still change the tree's shape.
    corrected = describe({"species": "Quercus palustris", "genus": "Ulmus"})
    assert (corrected.form, corrected.name) == ("oval", "Pin oak")
    # A record that names only a genus still works.
    assert describe({"genus": "Ulmus"}).form == "vase"


def test_pendula_as_a_species_name_is_not_a_weeping_tree():
    from boston_pipeline.trees import describe

    assert describe({"species": "Betula pendula"}).form != "weeping"  # silver birch
    assert describe({"species": "Fagus sylvatica 'Pendula'"}).form == "weeping"
    assert describe({"species": "Fagus sylvatica f. pendula"}).form == "weeping"


"""How a tree looks: the crown's skin and lobes, hidden faces, colours, the evergreen forms, willows.

test_trees.py has the sizes and the limits every tree keeps; this file has what makes one tree not look like
another. Nothing here depends on the viewer: it checks the geometry and the vertex colours the tiles carry."""

import math
import zlib

import numpy as np
import pytest

from boston_pipeline import trees
from boston_pipeline.mesh import MeshBuilder
from boston_pipeline.trees import (
    CROWN_BASE, EVERGREEN_FORMS, FORMS, TreeShape, _POLED, _POLED1, _POLED1_FACES, _POLED_FACES,
    _assemble, _hide_inside, _skin, _stem, describe,
)

CM = 100 / math.pi


def girth(d_cm):
    return str(d_cm / CM)


def shape(form, height=15.0, width=None, trunk=0.25, leaf="#55803c"):
    base = CROWN_BASE[form] * height
    if form not in EVERGREEN_FORMS:
        base = max(base, min(2.2, 0.45 * height))
    if form == "weeping":
        base = min(1.75, 0.45 * height)
    return TreeShape(form, height, width or 0.8 * height, base, trunk, leaf, "#6a5846", None, True)


def build(s, seed=0, x=0.0, y=0.0):
    builder = MeshBuilder("trees", colours=True, features=True)
    trees.add_tree(builder, s, x, y, seed, 0)
    return builder.finish()


def tris(s, seed=0):
    return len(build(s, seed).indices) // 3


def foliage(parts):
    return [p for p in parts if p.kind in ("canopy", "cone", "skirt")]


# --- which form a tree gets -------------------------------------------------------------------------


@pytest.mark.parametrize("tags,form", [
    ({"species": "Picea abies"}, "conifer"),
    ({"species": "Tsuga canadensis"}, "conifer"),
    ({"species": "Pinus strobus"}, "pine"),
    ({"genus": "Cedrus"}, "pine"),
    ({"species": "Thuja occidentalis"}, "spire"),
    ({"species": "Juniperus virginiana"}, "spire"),
    ({"species": "Taxus baccata"}, "clipped"),
    ({"species": "Salix babylonica"}, "weeping"),
    ({"species": "Salix sepulcralis"}, "weeping"),
    ({"species": "Quercus phellos"}, "oval"),
    ({"species": "Prunus x yedoensis"}, "small"),
    ({"species": "Aesculus hippocastanum"}, "round"),
    ({"species": "Sassafras albidum"}, "round"),
])
def test_species_and_genera_get_their_forms(tags, form):
    assert describe(tags).form == form


def test_every_form_is_reachable_and_has_a_crown_base():
    assert set(CROWN_BASE) == set(FORMS)
    genera = ("Ulmus", "Tilia", "Quercus", "Ginkgo", "Malus", "Picea", "Pinus", "Thuja", "Taxus", "Salix")
    assert {describe({"genus": g}).form for g in genera} == set(FORMS)


def test_evergreens_keep_their_foliage_to_the_ground_and_the_rest_do_not():
    for tags in ({"species": "Picea"}, {"species": "Thuja"}, {"species": "Taxus"}):
        s = describe({**tags, "circumference": girth(60)}, 1)
        assert s.form in EVERGREEN_FORMS and s.crown_base <= 0.13 * s.height
    for tags in ({"species": "Quercus"}, {"species": "Ginkgo"}, {"species": "Pinus"}):
        s = describe({**tags, "circumference": girth(60)}, 1)
        assert s.crown_base >= min(2.2, 0.45 * s.height) - 1e-9


def test_an_unnamed_needleleaved_tree_is_a_dome_when_it_is_nearly_as_wide_as_tall():
    # The evergreens along the Public Garden's walks: 2.5 m across and 3.5 m tall, no species.
    dome = describe({"natural": "tree", "leaf_type": "needleleaved", "height": "3.5", "diameter_crown": "2.5"})
    assert dome.form == "clipped" and (dome.height, dome.crown_width) == (3.5, 2.5)
    # A narrow one of the same height is a spruce-like cone, and so is one with no measured size.
    assert describe({"leaf_type": "needleleaved", "height": "10", "diameter_crown": "3.5"}).form == "conifer"
    assert describe({"leaf_type": "needleleaved", "height": "3.5"}).form == "conifer"
    # A named tree keeps its own form: a spruce that happens to be wide is still a spruce.
    assert describe({"species": "Picea abies", "height": "3.5", "diameter_crown": "3"}).form == "conifer"


# --- the geometry primitives --------------------------------------------------------------------------


def edges_of(faces):
    count = {}
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            count[tuple(sorted(e))] = count.get(tuple(sorted(e)), 0) + 1
    return count


@pytest.mark.parametrize("verts,faces,vertex_count,face_count", [
    (_POLED, _POLED_FACES, 12, 20), (_POLED1, _POLED1_FACES, 42, 80),
])
def test_the_poled_icosahedron_is_closed_with_a_vertex_at_each_pole(verts, faces, vertex_count, face_count):
    assert verts.shape == (vertex_count, 3) and faces.shape == (face_count, 3)
    assert np.allclose(np.linalg.norm(verts, axis=1), 1)
    assert verts[:, 2].max() == pytest.approx(1) and verts[:, 2].min() == pytest.approx(-1)
    assert set(edges_of(faces).values()) == {2}  # every edge is shared by exactly two faces: no holes


def test_stem_has_a_ring_per_point_with_the_radii_given_and_faces_pointing_out():
    p, n, f = _stem([(0, 0, 0), (0.1, 0, 2), (0.3, 0, 5)], [1.0, 0.7, 0.3], 6)
    assert p.shape == (18, 3) and f.shape == (24, 3) and np.allclose(np.linalg.norm(n, axis=1), 1)
    for ring, radius in zip(range(3), (1.0, 0.7, 0.3)):
        pts = p[ring * 6 : ring * 6 + 6]
        assert np.allclose(np.linalg.norm(pts[:, :2] - pts[:, :2].mean(axis=0), axis=1), radius, atol=0.02)
    a, b, c = (p[f[:, i]] for i in range(3))
    centroid = (a + b + c) / 3
    axis = np.column_stack([centroid[:, 0] - np.interp(centroid[:, 2], [0, 2, 5], [0, 0.1, 0.3]), centroid[:, 1], np.zeros(len(f))])
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), axis) > 0).all()


def test_an_upright_stem_has_flat_rings_even_when_it_leans():
    p, _, _ = _stem([(0, 0, 0), (0.2, 0, 1.0), (0.5, 0, 4.0)], [1, 0.8, 0.5], 6, upright=True)
    assert np.allclose(p[:6, 2], 0) and np.allclose(p[6:12, 2], 1.0) and np.allclose(p[12:, 2], 4.0)


@pytest.mark.parametrize("fine", [False, True])
@pytest.mark.parametrize("form", ["round", "oval", "vase", "columnar", "small"])
def test_the_skin_is_a_closed_outward_surface_inside_the_crowns_limits(fine, form):
    s = shape(form, 16.0, trunk=0.3)
    for seed in range(6):
        part = _skin(s, seed, fine)
        z = part.positions[:, 2]
        assert z.min() == pytest.approx(s.crown_base, abs=0.1 * (s.height - s.crown_base))
        assert z.max() <= s.height * 1.02 and z.min() >= s.crown_base - 1e-6
        assert np.hypot(*part.positions[:, :2].T).max() <= 1.55 * s.crown_width / 2
        assert part.skin and part.bulge is not None and len(part.bulge) == len(part.positions)
        a, b, c = (part.positions[part.faces[:, i]] for i in range(3))
        assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3 - part.centre) > 0).all()
        assert set(edges_of(part.faces).values()) == {2}


def test_skins_differ_between_trees_and_are_the_same_for_one_tree():
    s = shape("round", 16.0)
    a, b = _skin(s, 3, True), _skin(s, 3, True)
    assert np.array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, _skin(s, 4, True).positions)
    # A crown is lumpy: its radius varies with direction by a good margin.
    reach = np.linalg.norm(a.positions - a.centre, axis=1)
    assert reach.max() / reach.min() > 1.25


# --- hidden faces ------------------------------------------------------------------------------------


def test_hiding_faces_inside_other_lobes_keeps_every_vertex_used_and_loses_no_outer_face():
    s = shape("round", 18.0, trunk=0.35)
    for seed in range(4):
        parts = _assemble(s, seed)
        for part in parts:
            used = np.unique(part.faces)
            assert len(used) == len(part.positions) and len(part.faces) > 0
    # The cull really does remove faces: a tree with six lobes does not carry six lobes' worth.
    lobes = [p for p in _assemble(s, 1) if p.kind == "canopy" and not p.skin]
    assert len(lobes) >= 3 and sum(len(p.faces) for p in lobes) < 20 * len(lobes)


def test_a_lobe_wholly_inside_another_is_dropped_and_a_lobe_apart_is_kept():
    from boston_pipeline.trees import _canopy

    big, inside, apart = _canopy((0, 0, 5), (4, 4, 4), 1), _canopy((0.2, 0, 5), (1, 1, 1), 2), _canopy((20, 0, 5), (1, 1, 1), 3)
    parts = [big, inside, apart]
    _hide_inside(parts)
    assert any(p is big for p in parts) and any(p is apart for p in parts) and not any(p is inside for p in parts)
    assert len(apart.faces) == 20  # nothing of it was inside anything


# --- colour -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", [f for f in FORMS if f not in ("weeping",)])
def test_a_crown_has_light_and_shade_in_it(form):
    s = shape(form, 14.0, trunk=0.25)
    for seed in range(3):
        parts = foliage(_assemble(s, seed))
        brightness = np.concatenate([p.colours[:, :3].sum(axis=1, dtype=float) for p in parts])
        assert brightness.std() / brightness.mean() > 0.12  # not one flat green
        assert brightness.max() / max(brightness.min(), 1) > 1.8
        assert (np.concatenate([p.colours[:, 3] for p in parts]) == 255).all()


def test_the_top_of_a_crown_is_lighter_than_its_underside_and_the_green_is_yellower_on_top():
    s = shape("round", 18.0, trunk=0.3)
    parts = [p for p in _assemble(s, 2) if p.kind == "canopy"]
    z = np.concatenate([p.positions[:, 2] for p in parts])
    colours = np.concatenate([p.colours for p in parts]).astype(float)
    top, bottom = z > np.percentile(z, 80), z < np.percentile(z, 20)
    assert colours[top, :3].sum(axis=1).mean() > 1.5 * colours[bottom, :3].sum(axis=1).mean()
    ratio = lambda rows: rows[:, 0].mean() / rows[:, 2].mean()  # red over blue: higher is yellower
    assert ratio(colours[top]) > ratio(colours[bottom])


def test_trees_of_one_species_are_not_all_the_same_green():
    s = shape("round", 12.0)
    means = [np.concatenate([p.colours[:, :3] for p in foliage(_assemble(s, seed))]).mean(axis=0) for seed in range(8)]
    assert np.ptp([m.sum() for m in means]) > 1.0 and len({tuple(np.round(m, 1)) for m in means}) == 8


def test_trunk_is_lighter_higher_up_and_never_black():
    s = shape("round", 14.0, trunk=0.3)
    trunk = _assemble(s, 3)[0]
    assert trunk.kind == "trunk"
    brightness = trunk.colours[:, :3].sum(axis=1, dtype=float)
    assert brightness[trunk.positions[:, 2] > 3].mean() > brightness[trunk.positions[:, 2] < 0.5].mean()
    assert brightness.min() > 40


# --- trunks and limbs -------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["round", "oval", "vase", "columnar", "small", "pine"])
def test_trunk_flares_at_the_foot_and_leans_a_little(form):
    s = shape(form, 15.0, trunk=0.4)
    trunk = _assemble(s, 5)[0]
    z = trunk.positions[:, 2]
    foot = np.hypot(*(trunk.positions[z < 0.01, :2] - trunk.positions[z < 0.01, :2].mean(axis=0)).T).max()
    higher = trunk.positions[z > 1.0]
    head = np.hypot(*(higher[:, :2] - higher[:, :2].mean(axis=0)).T).max()
    assert foot == pytest.approx(0.4, abs=0.03) and head < 0.8 * foot
    assert np.hypot(*trunk.positions[z == z.max(), :2].mean(axis=0)) <= 0.02 * 15.0 + 1e-9  # leans by at most 2% of the height


def test_limbs_appear_with_the_crown_and_reach_into_it():
    small = [p.kind for p in _assemble(shape("round", 6.0, width=4.0, trunk=0.15), 1)]
    medium = [p.kind for p in _assemble(shape("round", 10.0, width=7.0, trunk=0.2), 1)]
    large = [p.kind for p in _assemble(shape("round", 18.0, width=15.0, trunk=0.4), 1)]
    assert small.count("limb") == 0 and medium.count("limb") == 3 and large.count("limb") == 4
    s = shape("round", 18.0, width=15.0, trunk=0.4)
    parts = _assemble(s, 1)
    for limb in (p for p in parts if p.kind == "limb"):
        assert limb.positions[:, 2].min() < s.crown_base + 0.1 * (s.height - s.crown_base)  # starts at the fork
        assert limb.positions[:, 2].max() > s.crown_base  # and rises to the crown


# --- how much a tree costs --------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["round", "oval", "vase", "columnar", "small"])
def test_a_bigger_crown_gets_more_detail(form):
    sizes = [(4.0, 2.5, 0.1), (8.0, 5.5, 0.2), (14.0, 10.0, 0.35), (22.0, 17.0, 0.5)]
    counts = [np.mean([tris(shape(form, h, width=w, trunk=t), seed) for seed in range(6)]) for h, w, t in sizes]
    assert counts == sorted(counts) and counts[0] < 90 and counts[-1] > 140 and counts[-1] < 400


def test_the_pilot_like_mix_costs_about_a_hundred_triangles_a_tree():
    # Crown widths and species roughly as the pilot has them: mostly street trees of 5-10 m, some saplings, a few big ones.
    mix = [("Gleditsia", 25)] * 6 + [("Tilia", 22)] * 4 + [("Quercus", 60)] * 3 + [("Ulmus", 90)] * 2 + [("Malus", 15)] * 2 + [("Ginkgo", 12)] * 2
    counts = []
    for k, (genus, d) in enumerate(mix):
        s = describe({"genus": genus, "circumference": girth(d)}, zlib.crc32(f"{genus}{k}".encode()))
        counts.append(tris(s, k))
    assert 50 <= np.mean(counts) <= 190


# --- evergreens -------------------------------------------------------------------------------------


def foliage_radius_at(parts, z):
    best = 0.0
    for part in foliage(parts):
        for face in part.faces:
            pts = part.positions[face]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                za, zb = pts[a, 2], pts[b, 2]
                if (za - z) * (zb - z) <= 0 and za != zb:
                    q = pts[a] + (z - za) / (zb - za) * (pts[b] - pts[a])
                    best = max(best, math.hypot(q[0], q[1]))
    return best


def test_a_spruce_is_a_cone_of_tiers_widest_at_the_bottom_with_a_ragged_rim():
    s = shape("conifer", 16.0, width=6.4, trunk=0.3, leaf="#2f5236")
    for seed in range(4):
        parts = _assemble(s, seed)
        cones = [p for p in parts if p.kind == "cone"]
        assert 5 <= len(cones) <= 8
        low, mid, high = (foliage_radius_at(parts, f * 16.0) for f in (0.2, 0.55, 0.9))
        assert low > mid > high > 0 and low <= 1.25 * 3.2
        top = max(p.positions[:, 2].max() for p in parts)
        assert top == pytest.approx(16.0) and s.crown_base <= min(p.positions[:, 2].min() for p in cones) + 1.0
        # Each tier's rim is irregular: no two rim vertices of a tier are at one radius.
        rim = cones[0].positions[:-2]
        assert np.ptp(np.hypot(rim[:, 0], rim[:, 1])) > 0.1


def test_tiers_have_dark_gaps_between_them_and_light_tips():
    s = shape("conifer", 14.0, width=5.6, trunk=0.3, leaf="#2f5236")
    cone = next(p for p in _assemble(s, 2) if p.kind == "cone")
    brightness = cone.colours[:, :3].sum(axis=1, dtype=float)
    rim, apex = brightness[: len(brightness) - 2], brightness[-2]  # the last two vertices are the apex and the cap's centre
    assert rim.mean() > 1.3 * apex


def test_a_pine_is_a_bare_trunk_with_pads_above_it():
    s = shape("pine", 18.0, width=9.0, trunk=0.35, leaf="#2f5236")
    parts = _assemble(s, 3)
    pads = [p for p in parts if p.kind == "canopy"]
    trunk = parts[0]
    assert len(pads) >= 5 and trunk.positions[:, 2].max() == pytest.approx(0.92 * 18.0, abs=0.1)
    # Every pad is flat (wider than tall) and the lowest is well above the ground.
    for pad in pads:
        spread = np.hypot(*(pad.positions[:, :2] - pad.centre[:2]).T).max()
        assert np.ptp(pad.positions[:, 2]) < 1.3 * spread
    assert min(p.positions[:, 2].min() for p in pads) >= 0.2 * 18.0
    # Pads get smaller towards the top.
    order = sorted(pads, key=lambda p: p.centre[2])
    width = lambda p: np.hypot(*(p.positions[:, :2] - p.centre[:2]).T).max()
    assert width(order[0]) > width(order[-1])


def test_a_spire_is_a_narrow_pointed_column_and_a_clipped_evergreen_a_dome_to_the_ground():
    spire = _assemble(shape("spire", 10.0, width=3.5, trunk=0.2), 1)
    body = next(p for p in spire if p.kind == "canopy")
    assert np.ptp(body.positions[:, 2]) > 2.5 * np.hypot(*body.positions[:, :2].T).max()  # much taller than it is wide
    assert body.positions[:, 2].max() == pytest.approx(10.0)
    dome = _assemble(shape("clipped", 3.5, width=2.5, trunk=0.1), 1)
    assert [p.kind for p in dome] == ["canopy"] and len(dome[0].faces) >= 40
    z, rho = dome[0].positions[:, 2], np.hypot(*dome[0].positions[:, :2].T)
    assert z.min() <= 0.2 and z.max() == pytest.approx(3.5)
    assert rho[z < 1.0].max() > 1.0  # nearly as wide as the crown (radius 1.25) near the ground
    assert rho[z > 3.0].max() < 0.6 * rho.max()  # rounded off at the top


def test_the_evergreen_forms_are_dark_but_not_black():
    for form in ("conifer", "pine", "spire", "clipped"):
        parts = foliage(_assemble(shape(form, 12.0, width=5.0, trunk=0.25, leaf="#2f5236"), 1))
        brightness = np.concatenate([p.colours[:, :3].sum(axis=1, dtype=float) for p in parts])
        assert 20 < brightness.mean() < 450 and brightness.min() > 3


# --- willows ----------------------------------------------------------------------------------------


def test_a_willow_has_a_dome_on_limbs_and_strands_that_hang_to_head_height():
    s = shape("weeping", 16.0, width=17.0, trunk=0.5, leaf="#8aa35a")
    for seed in range(5):
        parts = _assemble(s, seed)
        kinds = [p.kind for p in parts]
        assert kinds.count("skirt") == 2 and kinds.count("limb") == 4 and kinds[0] == "trunk"  # a big one has four limbs
        dome = [p for p in parts if p.kind == "canopy"]
        strands = [p for p in parts if p.kind == "skirt"]
        z_dome = np.concatenate([p.positions[:, 2] for p in dome])
        z_strands = np.concatenate([p.positions[:, 2] for p in strands])
        assert z_dome.min() >= 0.35 * 16.0 and z_dome.max() == pytest.approx(16.0, abs=0.5)  # the dome is up in the tree
        assert 1.2 - 1e-6 <= z_strands.min() <= 2.0 and z_strands.max() < z_dome.max()  # strands hang from it to head height
        # Nothing but the trunk stands in the first metre: a walker can stand under it.
        assert min(p.positions[:, 2].min() for p in dome + strands) >= 1.2 - 1e-6


def test_strands_are_streaky_each_a_different_length_and_shade():
    s = shape("weeping", 16.0, width=17.0, trunk=0.5, leaf="#8aa35a")
    strand = next(p for p in _assemble(s, 2) if p.kind == "skirt")
    tips = strand.positions[strand.positions[:, 2] < 4]
    assert len(np.unique(np.round(tips[:, 2], 2))) >= 8  # a ragged hem, many lengths
    brightness = strand.colours[:, :3].sum(axis=1, dtype=float)
    assert brightness.max() > 1.8 * brightness.min()


def test_a_small_weeping_tree_costs_far_less_than_a_big_willow():
    big = tris(shape("weeping", 18.0, width=19.0, trunk=0.6, leaf="#8aa35a"), 1)
    small = tris(shape("weeping", 7.0, width=7.0, trunk=0.15, leaf="#55823f"), 1)
    assert small < 0.6 * big and big < 600


@pytest.mark.parametrize("name", ["Salix alba", "Salix babylonica"])
def test_willow_mesh_is_deterministic_and_unlike_its_neighbour(name):
    s = describe({"species": name, "circumference": girth(120)}, 7)
    a, b = build(s, seed=7), build(s, seed=7)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.colours, b.colours)
    assert not np.array_equal(a.positions, build(s, seed=8).positions)


# --- the whole thing --------------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_every_form_builds_at_every_age_with_valid_geometry(form):
    for height, width, trunk in ((2.5, 1.5, 0.05), (5.0, 3.0, 0.1), (10.0, 7.0, 0.25), (20.0, 16.0, 0.5)):
        s = shape(form, height, width=width, trunk=trunk)
        for seed in (0, 1, -3, 2**31 + 5):
            mesh = build(s, seed=seed)
            assert len(mesh.indices) > 0 and mesh.indices.max() < len(mesh.positions)
            assert np.isfinite(mesh.positions).all() and mesh.colours.shape == (len(mesh.positions), 4)
            assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-4)


def test_no_degenerate_triangles_in_a_mature_tree():
    for form in FORMS:
        s = shape(form, 16.0, width=12.0, trunk=0.35)
        for part in _assemble(s, 4):
            a, b, c = (part.positions[part.faces[:, i]] for i in range(3))
            assert (np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-9).all(), (form, part.kind)


def test_glb_buffer_length_matches_its_padded_chunk_when_a_layer_ends_in_an_odd_number_of_16_bit_indices(tmp_path):
    """Tree meshes made this show up: 16-bit indices of an odd triangle count left the declared buffer 2 bytes short."""
    import json
    import struct

    from boston_pipeline.gltf import Material, write_glb

    odd = next(m for m in (build(shape("round", 6.0, width=4.0, trunk=0.1), seed=k) for k in range(40)) if len(m.indices) % 6)
    assert len(odd.indices) // 3 % 2 == 1  # an odd number of triangles
    write_glb(tmp_path / "t.glb", [(odd, Material("#55803c"))], "test")
    data = (tmp_path / "t.glb").read_bytes()
    json_length, _ = struct.unpack_from("<I4s", data, 12)
    document = json.loads(data[20 : 20 + json_length])
    bin_length, bin_type = struct.unpack_from("<I4s", data, 20 + json_length)
    assert bin_type == b"BIN\0" and bin_length == document["buffers"][0]["byteLength"] and bin_length % 4 == 0

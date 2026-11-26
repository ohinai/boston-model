"""Gates: a stone pier at each end of the opening and iron leaves hinged on the piers and swung back."""

import json
import math

import numpy as np
import pytest
from shapely.geometry import LineString, box

from boston_pipeline import world
from boston_pipeline.barriers import (
    GATE_HEIGHT_MAX, GATE_LEAF_MAX, GATE_MIN_OPENING, PIER_ABOVE_LEAF, PIER_CAP, PIER_STONE, WOOD_COLOUR, _finial_kind,
    add_barrier, add_gate, gate_plan, gate_swing_side, join_gates,
)
from boston_pipeline.features import Area, Barrier, Building, Scene, classify
from boston_pipeline.mesh import MeshBuilder, colour_bytes
from boston_pipeline.tiles import build

from test_barriers import _barrier, _enu, _path, _way
from test_parks import _mesh_data, _read_glb

IRON = "#112233"  # a colour no pier has, so the iron of a gate can be told from its stone
PLAYER_RADIUS = 0.3  # web/src/player.ts
EDGE = 0.1  # the leaves' uprights and the balls on them reach at most this far into the opening (0.09)


def _gate(a, b, height=1.5, **more):
    builder = MeshBuilder("barriers", colours=True, features=True)
    more.setdefault("colour", IRON)
    add_gate(builder, np.array([a, b], dtype=float), height, feature=5, **more)
    return builder.finish()


def _frame(mesh, a, b):
    """(u, n, z, is_iron) for every vertex: u along the way from `a` (0 at a, the opening's length at b), n to its left."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    u = (b - a) / np.hypot(*(b - a))
    enu = _enu(mesh)
    rel = enu[:, :2] - a
    iron = (mesh.colours == colour_bytes(IRON)).all(axis=1)
    return rel @ u, rel @ np.array([-u[1], u[0]]), enu[:, 2], iron


def _sound(mesh):
    assert len(mesh.indices) > 0 and len(mesh.indices) % 3 == 0
    assert np.isfinite(mesh.positions).all() and np.isfinite(mesh.normals).all()
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    tri = mesh.positions[mesh.indices.reshape(-1, 3)]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.linalg.norm(face, axis=1) > 1e-10).all(), "degenerate triangle"
    assert (np.einsum("ij,ij->i", face, mesh.normals[mesh.indices.reshape(-1, 3)[:, 0]]) > 0).all(), "wound the wrong way"
    assert (mesh.features == 5).all()
    assert _enu(mesh)[:, 2].min() >= -1e-6, "nothing below the ground"


# Straight openings in every direction, from the narrowest gate in the pilot to the widest.
OPENINGS = [0.97, 1.3, 1.34, 2.0, 2.97, 5.8, 10.28, 12.06]
DIRECTIONS = [0.0, 33.0, 90.0, 215.0]


def _opening(length, degrees, start=(40.0, 60.0)):
    angle = math.radians(degrees)
    a = np.array(start)
    return a, a + length * np.array([math.cos(angle), math.sin(angle)])


# --- Reading the tags ---------------------------------------------------------------------------
def test_a_gate_way_is_a_gate_barrier_with_the_leaves_height():
    (gate,) = classify([_way(1, {"barrier": "gate", "colour": "black", "material": "iron", "height": "2"}, [(0, 0), (4, 0)])]).barriers
    assert (gate.kind, gate.height, gate.colour, gate.material) == ("gate", 2.0, "#2b2b2d", "iron")
    (gate,) = classify([_way(1, {"barrier": "gate"}, [(0, 0), (4, 0)])]).barriers
    assert gate.height == 1.5  # the parks' railings


@pytest.mark.parametrize("tags", [
    {"barrier": "gate", "gate": "ticket_barrier"},
    {"barrier": "gate", "gate": "turnstile"},
    {"barrier": "gate", "tunnel": "yes"},
    {"barrier": "gate", "location": "indoor", "indoor": "yes"},
    {"barrier": "gate", "layer": "-1"},
    {"barrier": "gate", "highway": "footway"},
])
def test_a_ticket_gate_an_underground_gate_and_a_footway_are_not_drawn(tags):
    assert classify([_way(1, tags, [(0, 0), (4, 0)])]).barriers == []


# --- The opening stays clear --------------------------------------------------------------------
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("degrees", DIRECTIONS)
@pytest.mark.parametrize("length", OPENINGS)
def test_the_whole_opening_is_clear_for_a_walker(length, degrees, side):
    """Nothing but the piers stands in the opening, at any height, and what the leaves leave is wider than a player."""
    a, b = _opening(length, degrees)
    mesh = _gate(a, b, side=side)
    u, n, z, iron = _frame(mesh, a, b)
    assert not ((u > EDGE + 1e-6) & (u < length - EDGE - 1e-6)).any(), "something stands in the opening"
    assert length - 2 * EDGE > 2 * PLAYER_RADIUS + 0.1, "even the narrowest gate lets a player through"
    # The piers are outside the opening altogether: nothing stone is past the ends of the way.
    stone = ~iron
    assert not ((u[stone] > 0.01) & (u[stone] < length - 0.01)).any()


@pytest.mark.parametrize("length", [0.97, 2.0, 5.8, 12.06])
def test_the_leaves_are_swung_back_to_the_chosen_side(length):
    a, b = _opening(length, 20.0)
    for side in (1, -1):
        mesh = _gate(a, b, side=side)
        u, n, z, iron = _frame(mesh, a, b)
        leaf = side * n[iron]
        assert leaf.min() > 0.0, "the iron is all on one side of the opening"
        assert leaf.max() > 0.5 * min(length / 2, GATE_LEAF_MAX) * 0.8, "swung well clear of the opening, not left across it"


def test_a_leaf_is_hinged_on_the_pier_and_never_longer_than_the_limit():
    a, b = _opening(12.06, 0.0)
    mesh = _gate(a, b)
    u, n, z, iron = _frame(mesh, a, b)
    plan = gate_plan(np.array([a, b]), 1.5)
    for k, end in ((0, 0.0), (1, 12.06)):
        hinge, _ = plan.leaf(k, 1)
        hinge_u, hinge_n = (hinge - a) @ plan.u, (hinge - a) @ plan.n
        mine = iron & (np.abs(u - end) < GATE_LEAF_MAX + 0.5)
        assert mine.any()
        reach = np.hypot(u[mine] - hinge_u, n[mine] - hinge_n)
        assert reach.max() <= GATE_LEAF_MAX + 0.06, "a leaf of a wide gate is capped"
        assert reach.max() > GATE_LEAF_MAX - 0.1
        assert reach.min() < 0.1, "it is hinged at the pier's face"
        assert 0.0 < abs(hinge_n) < plan.pier_width / 2 + 0.01, "on the face of the pier"
    assert not (iron & (u > 0.5 * 12.06 - 2.0) & (u < 0.5 * 12.06 + 2.0)).any(), "the middle of a wide gate is open"


def test_a_narrow_opening_has_one_leaf_and_a_wide_one_two():
    one = _frame(_gate(*_opening(1.0, 0.0)), *_opening(1.0, 0.0))
    two = _frame(_gate(*_opening(4.0, 0.0)), *_opening(4.0, 0.0))
    assert (one[0][one[3]] < 0.5).all(), "one leaf, at the first end"
    assert (two[0][two[3]] < 0.5 * 4.0).any() and (two[0][two[3]] > 0.5 * 4.0).any()
    assert gate_plan(np.array([(0, 0), (1.0, 0)]), 1.5).hinges == (0,)
    assert gate_plan(np.array([(0, 0), (4.0, 0)]), 1.5).hinges == (0, 1)


# --- The piers ----------------------------------------------------------------------------------
@pytest.mark.parametrize("length", [1.1, 2.4, 5.1, 10.25])
def test_a_pier_stands_at_each_end_on_the_line_of_the_way(length):
    a, b = _opening(length, 57.0)
    mesh = _gate(a, b, 1.5)
    u, n, z, iron = _frame(mesh, a, b)
    stone = ~iron
    width = gate_plan(np.array([a, b]), 1.5).pier_width
    for inside, outer in ((u < length / 2, "start"), (u >= length / 2, "end")):
        mine = stone & inside
        assert mine.any(), f"no pier at the {outer}"
        centre = (u[mine].min() + u[mine].max()) / 2
        expected = -width * PIER_CAP / 2 if outer == "start" else length + width * PIER_CAP / 2
        assert centre == pytest.approx(expected, abs=0.02)  # the cap, the widest part, just touches the end of the way
        assert abs((n[mine].min() + n[mine].max()) / 2) < 0.01, "on the line of the way, not beside it"
        assert z[mine].max() > 1.5 + PIER_ABOVE_LEAF, "taller than the leaves"
        assert z[mine].min() == pytest.approx(0.0, abs=1e-6), "standing on the ground"


def test_a_pier_is_at_least_as_tall_as_the_fence_that_runs_into_it():
    a, b = (0.0, 0.0), (4.0, 0.0)
    plain = _frame(_gate(a, b, 1.0), a, b)
    meets = _frame(_gate(a, b, 1.0, fence=3.0), a, b)
    assert plain[2][~plain[3]].max() < 2.3
    assert meets[2][~meets[3]].max() > 3.0 + PIER_ABOVE_LEAF
    assert meets[2][meets[3]].max() == pytest.approx(1.0, abs=1e-4), "the leaves keep their own height"


@pytest.mark.parametrize("opening,height,kind", [
    (1.0, 1.5, "cap"), (1.7, 2.0, "cap"), (2.0, 1.5, "ball"), (3.9, 1.5, "ball"), (4.0, 1.5, "urn"), (2.0, 2.0, "urn"), (10.0, 1.2, "urn"),
])
def test_a_narrow_gate_has_a_plain_cap_a_wider_one_a_ball_and_a_wide_or_tall_one_an_urn(opening, height, kind):
    assert _finial_kind(opening, height) == kind


def test_piers_are_granite_unless_the_tags_say_brick():
    a, b = (0.0, 0.0), (3.0, 0.0)
    stone = lambda mesh: mesh.colours[~(mesh.colours == colour_bytes(IRON)).all(axis=1)]  # noqa: E731
    granite = stone(_gate(a, b))
    assert (np.abs(granite[:, :3].astype(int) - colour_bytes(PIER_STONE)[:3]) <= colour_bytes(PIER_STONE)[:3] * 0.25 + 3).all()
    brick = stone(_gate(a, b, tags={"material": "brick;iron"}))
    assert brick[:, 0].mean() > granite[:, 0].mean() * 0.8 and not np.array_equal(brick, granite)


# --- Leaves: colour, material, height -----------------------------------------------------------
def test_the_colour_tag_is_the_irons_and_the_default_is_dark():
    default = _gate((0, 0), (3, 0), colour=None)
    assert (default.colours == colour_bytes("#23262a")).all(axis=1).any()
    assert (_gate((0, 0), (3, 0), colour="#2f6b3a").colours == colour_bytes("#2f6b3a")).all(axis=1).any()


@pytest.mark.parametrize("material", [None, "iron", "steel", "metal", "wire", "wood", "timber", "unheard_of"])
@pytest.mark.parametrize("height", [0.5, 1.0, 1.5, 2.0, 2.5, 5.0])
def test_every_kind_of_gate_is_sound_and_no_taller_than_the_limit(material, height):
    mesh = _gate((0, 0), (3, 0), height, material=material, tags={"material": material or ""})
    _sound(mesh)
    u, n, z, iron = _frame(mesh, (0, 0), (3, 0))
    leaf_top = min(max(height, 0.9), GATE_HEIGHT_MAX)
    assert z[iron].max() == pytest.approx(leaf_top, abs=1e-4)
    assert z.max() < GATE_HEIGHT_MAX + PIER_ABOVE_LEAF + 1.2, "the piers do not tower"


def test_a_wooden_gate_hangs_on_wooden_posts_with_boards_and_a_railing_has_bars():
    wood = _gate((0, 0), (3, 0), 1.5, material="wood", colour=None)
    colours = {tuple(c) for c in wood.colours}
    assert colours == {tuple(colour_bytes(WOOD_COLOUR)), tuple(colour_bytes(WOOD_COLOUR, 1.1))}, "no stone, no iron"
    railing = _gate((0, 0), (3, 0), 1.5, material="iron")
    def tips(mesh):  # the distinct places where something reaches the leaves' full height
        top = _enu(mesh)[np.abs(_enu(mesh)[:, 2] - 1.5) < 1e-4]
        return len(np.unique(np.round(top[:, :2], 3), axis=0))

    assert tips(railing) >= 2 * 8, "a bar for every 0.16 m of a leaf"
    assert tips(wood) <= 2 * 4, "boards are one slab"


def test_a_gate_is_cheap():
    # About 150 triangles to a pier with an urn, about 100 to a leaf.
    assert len(_gate((0, 0), (5.1, 0), 2.0).indices) // 3 < 600
    assert len(_gate((0, 0), (1.1, 0), 1.5).indices) // 3 < 300


# --- Degenerate ways ----------------------------------------------------------------------------
@pytest.mark.parametrize("line", [
    [(5.0, 5.0)],
    [(5.0, 5.0), (5.0, 5.0)],
    [(5.0, 5.0), (5.0 + GATE_MIN_OPENING * 0.9, 5.0)],
    [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],  # a ring has no opening
    [(0.0, 0.0), (float("nan"), 1.0)],
])
def test_a_way_that_is_no_gate_draws_nothing(line):
    builder = MeshBuilder("barriers", colours=True, features=True)
    add_gate(builder, np.array(line, dtype=float), 1.5)
    add_barrier(builder, "gate", np.array(line, dtype=float), 1.5, 0.06)
    assert builder.empty
    assert gate_plan(np.array(line, dtype=float), 1.5) is None
    assert gate_swing_side(np.array(line, dtype=float), 1.5, box(0, 0, 1, 1), None) == 1


def test_a_gate_is_the_same_every_time():
    a, b = _opening(5.8, 33.0)
    first, second = _gate(a, b, side=-1), _gate(a, b, side=-1)
    assert np.array_equal(first.positions, second.positions) and np.array_equal(first.colours, second.colours)
    assert np.array_equal(first.indices, second.indices)
    # add_barrier draws the same gate, with the leaves to the left.
    builder = MeshBuilder("barriers", colours=True, features=True)
    add_barrier(builder, "gate", np.array([a, b]), 1.5, 0.06, IRON, None, 5)
    assert np.array_equal(builder.finish().positions, _gate(a, b, side=1).positions)


def test_a_bent_way_is_the_straight_opening_between_its_ends():
    bent = np.array([(0, 0), (1, 0.4), (3, 0.5), (4, 0)], dtype=float)
    builder = MeshBuilder("barriers", colours=True, features=True)
    add_gate(builder, bent, 1.5, IRON, feature=5)
    straight = _gate((0, 0), (4, 0))
    assert np.array_equal(builder.finish().positions, straight.positions)


# --- Which way the leaves swing -----------------------------------------------------------------
def test_leaves_swing_away_from_what_is_in_the_way_then_toward_the_lawn():
    line = np.array([(0.0, 0.0), (4.0, 0.0)])
    left, right = box(-3, 0.3, 7, 3), box(-3, -3, 7, -0.3)
    assert gate_swing_side(line, 1.5) == 1
    assert gate_swing_side(line, 1.5, left, None) == -1  # a footway runs along the left
    assert gate_swing_side(line, 1.5, right, None) == 1
    assert gate_swing_side(line, 1.5, left.union(right), right) == -1  # equally blocked: the side with the lawn
    assert gate_swing_side(line, 1.5, left.union(right), left) == 1
    assert gate_swing_side(line, 1.5, left.union(right), None) == 1
    assert gate_swing_side(line, 1.5, None, right) == -1  # the lawn is on the right
    assert gate_swing_side(line, 1.5, None, left) == 1
    # The same gate mapped the other way round: left and right swap.
    assert gate_swing_side(line[::-1], 1.5, None, right) == 1


# --- Joining the ways of a double gate ----------------------------------------------------------
def _g(ident, points, **more):
    return Barrier(ident, LineString(points), "gate", 1.5, 0.06, tags=more.pop("tags", {}), **more)


def test_gates_mapped_end_to_end_become_one_gate_with_one_opening():
    first, second = _g("w1", [(0, 0), (1.5, 0.4)], colour="#111111"), _g("w2", [(3.0, 0.8), (1.5, 0.4)], material="iron")
    fence = _barrier("w3", "fence", [(-5, 0), (0, 0)])
    other = _g("w4", [(50, 50), (53, 50)])
    out = join_gates([fence, first, other, second])
    assert [b.id for b in out] == ["w3", "w1", "w4"], "the order of everything else is kept"
    joined = out[1]
    assert joined.line.length == pytest.approx(3.2, abs=0.1) and joined.colour == "#111111" and joined.material == "iron"
    ends = {tuple(np.round(p, 2)) for p in (joined.line.coords[0], joined.line.coords[-1])}
    assert ends == {(0.0, 0.0), (3.0, 0.8)}
    plan = gate_plan(np.asarray(joined.line.coords), 1.5)
    assert plan.opening == pytest.approx(math.hypot(3.0, 0.8), abs=1e-6) and plan.hinges == (0, 1)


def test_three_ways_in_a_row_become_one_and_a_gate_turning_a_corner_does_not():
    row = [_g("a", [(0, 0), (2, 0)]), _g("c", [(4, 0), (6, 0)]), _g("b", [(2, 0), (4, 0)])]
    (only,) = join_gates(row)
    assert only.line.length == pytest.approx(6.0)
    corner = [_g("a", [(0, 0), (2, 0)]), _g("b", [(2, 0), (2, 2)])]
    assert [b.id for b in join_gates(corner)] == ["a", "b"]
    apart = [_g("a", [(0, 0), (2, 0)]), _g("b", [(2.5, 0), (4, 0)])]
    assert len(join_gates(apart)) == 2, "only ways that share an end are one gate"
    assert join_gates([]) == []


# --- In a tile ----------------------------------------------------------------------------------
def _build(tmp_path, scene, tiles):
    build(scene, tiles, tmp_path, {"name": "Test"}, None)
    out = {}
    for tile in tiles:
        path = tmp_path / "tiles" / f"{tile[0]}_{tile[1]}.glb"
        document = _read_glb(path)
        layers = {mesh["name"]: mesh["primitives"][0] for mesh in document["meshes"]}
        positions = None
        if "barriers" in layers:
            positions, _ = _mesh_data(path, document, layers["barriers"])
            positions = np.column_stack([positions[:, 0], -positions[:, 2], positions[:, 1]])
        semantics = json.loads((tmp_path / "tiles" / f"{tile[0]}_{tile[1]}.json").read_text(encoding="utf-8"))
        out[tile] = (positions, semantics)
    return out


def test_a_gate_in_a_tile_is_listed_once_and_drawn_where_the_way_is(tmp_path):
    scene = Scene()
    scene.barriers += [
        _barrier("w1", "fence", [(20, 50), (60, 50)], 1.5, 0.06, material="iron"),
        _barrier("w2", "gate", [(60, 50), (65, 50)], 1.5, 0.06, material="iron"),
        _barrier("w3", "fence", [(65, 50), (110, 50)], 1.5, 0.06, material="iron"),
    ]
    (positions, semantics), = _build(tmp_path, scene, [(0, 0)]).values()
    gates = [o for o in semantics["objects"] if o["kind"] == "gate"]
    assert len(gates) == 1 and gates[0]["id"] == "w2" and gates[0]["pos"] == [62.5, 50.0]
    # Both piers stand beyond the ends of the way, and the opening between them is open at every height.
    tall = positions[positions[:, 2] > 1.9]
    assert (tall[:, 0] < 60.01).any() and (tall[:, 0] > 64.99).any()
    assert not ((tall[:, 0] > 60.2) & (tall[:, 0] < 64.8) & (np.abs(tall[:, 1] - 50) < 0.4)).any()
    inside = positions[(positions[:, 0] > 60.2) & (positions[:, 0] < 64.8) & (np.abs(positions[:, 1] - 50) < 0.3)]
    assert len(inside) == 0, "the opening is clear: no rails or bars across it"


def test_a_gate_across_a_tile_edge_is_drawn_once_whole(tmp_path):
    edge = world.TILE_SIZE
    scene = Scene()
    scene.barriers.append(_barrier("w1", "gate", [(edge - 2.0, 50), (edge + 3.0, 50)], 1.5, 0.06, material="iron"))
    built = _build(tmp_path, scene, [(0, 0), (1, 0)])
    listed = [tile for tile, (_, semantics) in built.items() if any(o["kind"] == "gate" for o in semantics["objects"])]
    assert len(listed) == 1, "the tile of the middle has it, the other does not"
    (positions, _) = built[listed[0]]
    shift = listed[0][0] * edge  # the tile's own coordinates
    assert (positions[:, 0] + shift < edge - 2.0).any() and (positions[:, 0] + shift > edge + 3.0).any(), "both piers, in one tile"
    other = built[(1, 0) if listed[0] == (0, 0) else (0, 0)][0]
    assert other is None or len(other) == 0


def test_a_path_through_a_gate_does_not_open_it_and_its_leaves_swing_to_the_lawn(tmp_path):
    scene = Scene()
    scene.barriers.append(_barrier("w2", "gate", [(60, 50), (65, 50)], 1.5, 0.06, material="iron"))
    scene.ways.append(_path("p1", [(62.5, 0), (62.5, 100)], width=3.0))
    scene.areas["green"].append(Area("g1", box(40, 50, 90, 100), {"leisure": "park"}, "lawn"))  # the park is north
    (positions, _), = _build(tmp_path, scene, [(0, 0)]).values()
    leaves = positions[(positions[:, 2] > 0.1) & (positions[:, 2] < 1.4) & (np.abs(positions[:, 1] - 50) > 0.5)]
    assert (leaves[:, 1] > 50).sum() > 0 and (leaves[:, 1] < 50 - 1.0).sum() == 0, "the leaves stand in the park"


def test_a_pier_inside_a_building_is_left_out(tmp_path):
    scene = Scene()
    scene.buildings.append(Building("b1", box(40, 40, 60.2, 60), 12.0, 0.0, "default", "#aaaaaa", "#555555", {}))
    scene.barriers.append(_barrier("w2", "gate", [(60, 50), (64, 50)], 1.5, 0.06, material="iron"))
    (positions, _), = _build(tmp_path, scene, [(0, 0)]).values()
    piers = positions[positions[:, 2] > 1.7]  # the leaves are 1.5 m high
    assert not (piers[:, 0] < 60.0).any(), "no pier on the building's side"
    assert (piers[:, 0] > 64.0).any(), "the pier in the open stays"


def test_the_piers_rise_with_the_fence_that_runs_into_them(tmp_path):
    scene = Scene()
    scene.barriers += [
        _barrier("w1", "fence", [(20, 50), (60, 50)], 3.0, 0.06),
        _barrier("w2", "gate", [(60, 50), (64, 50)], 1.0, 0.06, material="iron"),
    ]
    (positions, _), = _build(tmp_path, scene, [(0, 0)]).values()
    piers = positions[(positions[:, 0] > 60.05) & (positions[:, 0] < 64 + 1.0) & (np.abs(positions[:, 1] - 50) < 0.5)]
    assert positions[:, 2].max() > 3.0 + PIER_ABOVE_LEAF
    assert piers[:, 2].max() > 3.0 + PIER_ABOVE_LEAF or (positions[positions[:, 0] > 64.0][:, 2].max() > 3.3)

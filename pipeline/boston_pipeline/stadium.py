"""Ballparks and stadiums: raked stands, field walls, the walls of a stadium's parts, foul poles and light towers.

OpenStreetMap maps a stadium as a `building=stadium` outline (the ring of structure round the playing surface) holding
`building:part` polygons, or as a lone `building=stadium` or `building=grandstand` polygon, next to a `leisure=pitch`. Drawn as
plain buildings those are blocks with windows round a lawn. This module makes them look like a ballpark.

What is drawn, for any stadium that has an open pitch within `FIELD_REACH` metres of its outline (an enclosed arena with no
pitch is left alone):

- **Stands (terraces).** The part of the outline that no part covers, and every `leisure=bleachers` or
  `building:part=bleachers` part wide enough to hold seats, becomes a raked terrace (`augment` adds each as a `Building` whose
  tag `stand=terrace` selects this module's drawing). Rows of `ROW_DEPTH` metres rise `ROW_RISE` metres each from a low front wall
  at the field, along the outline's edge nearest the field, up to a concourse, and the terrace's back wall is a ballpark facade.
  The seat backs (the risers) are the seat colour and the treads a lighter one; see SEAT and TREAD.
- **Parts of the stadium** (`building:part` polygons that belong to the outline, other than floating roofs). Walls facing the field
  are drawn as a deck's fascia (a shaded deck, a rail, the roof edge), the other walls with the ballpark facade (`style` 1 in
  `mesh.facade_u`, drawn by web/src/facade.ts: brick, pilasters, arched openings, a base course and a band under the cornice).
- **Field walls.** A narrow `bleachers` part with a `height` tag (the Green Monster) is a wall: flat colour (`building:colour`)
  on both faces, a cap, and where tagged `stand:scoreboard=yes` a scoreboard panel.
- **Foul poles** (`baseball=foul_pole`, with `height` and a `direction` toward fair territory) and **light towers**
  (`man_made=floodlight`, or a `man_made=mast` with `mast:type=floodlight`; `height`, and a `direction` toward the field), points
  added by files in fixes/, drawn through extras.py.
- **Landmark signs.** A large named sign that OpenStreetMap maps as an attraction (`advertising=sign` or `billboard` with
  `tourism=attraction` and a `height` of 15 m or more: the Citgo sign over Kenmore Square, the Boston Wharf Co. sign) is drawn as
  a board on a frame in one flat colour (and, with `sign:face=triangle`, a plain triangle and a band), never with lettering. It is here
  because the Citgo sign is Fenway Park's neighbour.

The tags this module reads, besides OpenStreetMap's: `stand` (`terrace`, `wall`, `deck` or `none`, to override the choice made
here), `seat:colour` and `seat:tread` (on the outline or the part), `stand:scoreboard`. Every height and depth is an
interpretation of a kind of thing (a lower deck is 1 to 9 m above the field; a row is 0.9 m deep and 0.4 m high), not a
measurement: the data says where the stand is and how tall its roof is, not how it is raked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

from .features import Building, Scene, parse_colour, parse_length, polygons_of
from .furniture import _frame, _Opts, cuboid, face, limb, prism
from .mesh import BAY_WIDTH, FLOOR_HEIGHT, MAX_STOREYS, STOREY_STRIDE, colour_bytes, facade_u

# --- What counts as a stadium ---------------------------------------------------------------------------------------
STADIUM_BUILDINGS = {"stadium", "grandstand"}
FIELD_REACH = 30.0  # a pitch this close to a stadium's outline is its field
FIELD_MIN_AREA = 800.0  # smaller pitches are courts, not a stadium's field
FACADE_STYLE_OUTSIDE = 1  # `style` in mesh.facade_u: web/src/facade.ts draws a ballpark's outside wall
PART_MIN_AREA = 15.0  # a part smaller than this (a stair, a booth) is left as an ordinary building

# --- The stands (metres) --------------------------------------------------------------------------------------------
FRONT_WALL = 1.15  # the wall between the field and the first row
ROW_DEPTH, ROW_RISE = 0.9, 0.4  # a row of seats: deep and high (a slope of 24 degrees)
CONCOURSE_MAX = 9.5  # the lower deck rises to at most this above the field
CONCOURSE_SHARE = 0.5  # and to at most this share of the height of the structure behind it
TERRACE_MIN_AREA = 60.0  # smaller leftovers between parts are not stands
SLIVER = 1.5  # leftovers narrower than twice this are dropped
WALL_WIDTH = 8.0  # a bleachers part narrower than this, with a `height` of its own, is a field wall, not a terrace
NEIGHBOUR = 1.0  # a part this close to a terrace is what is behind it
INSIDE_REACH = 45.0  # a part's wall this close to the field and facing it is an inside wall
INSIDE_FACING = 0.35  # the cosine of the angle between a wall's normal and the way to the field, above which it faces the field
SIDE_REACH = 30.0  # a wall that faces neither toward nor away from the field (the end of a stand) is an inside wall this close to it

# --- Colours --------------------------------------------------------------------------------------------------------
# A generic stand: plastic seats in a muted blue over pale concrete. Fenway Park's are set by tags in fixes/fenway/.
SEAT, TREAD = "#46617f", "#a1a5a8"
CONCRETE = "#8d8b85"
DECK_VOID, DECK_RAIL, DECK_FASCIA = "#26332d", "#c9c4b4", "#b8b5a8"
WALL_CAP = "#2c3a33"
SCOREBOARD, SCOREBOARD_LINES = "#151a17", "#d9d6c8"
FASCIA_DARKEN = 0.55
SIDE_DARKEN = 0.6  # the ends of a terrace (a dugout's walls, the end of a stand) are the seat colour, darkened
AISLE, AISLE_RISER = "#b9b7ae", "#9b9990"  # the stairs between blocks of seats
AISLE_SPACING, AISLE_WIDTH = 8.0, 1.2  # a stair every this many metres along the front of a terrace
# Seats on top of a field wall (the Green Monster's): this many rows, each this deep and this high, over a low rail.
TOPPER_ROWS, TOPPER_DEPTH, TOPPER_RISE, TOPPER_RAIL = 3, 1.25, 0.42, 0.55

# --- Foul poles and light towers (metres) ------------------------------------------------------------------------------
FOUL_POLE_HEIGHT, FOUL_POLE_RANGE = 15.0, (6.0, 40.0)
FOUL_POLE_RADIUS, FOUL_POLE_SCREEN = 0.1, (0.55, 0.05)  # the pole's radius, and the mesh screen's width and thickness
FOUL_POLE_YELLOW = "#e3c21c"
LIGHT_TOWER_HEIGHT, LIGHT_TOWER_RANGE = 32.0, (8.0, 60.0)
LIGHT_BANK = (7.0, 0.6, 3.6)  # the lamp bank on top: wide, deep, tall
TOWER_STEEL, TOWER_HOUSING, TOWER_LAMP = "#9aa0a4", "#43484c", "#efe9c8"
TOWER_FOOT, TOWER_HEAD = 1.5, 0.9  # half the width of the lattice at its foot and under the lamps
# A landmark sign: a board `width` wide (the first number of its `size` tag, else this) and as tall as it is wide (else the second
# number), two-sided if its `direction` names two ways or `sides` is 2, standing on four legs.
LANDMARK_SIGN_MIN, SIGN_WIDTH, SIGN_DEPTH = 15.0, 18.0, 1.0
SIGN_FRAME, SIGN_LEG, SIGN_FACE, SIGN_RED, SIGN_BLUE = "#30343a", "#4c5258", "#e8e8e3", "#cf2a2f", "#1c3f8c"


@dataclass
class Plan:
    """How one building is drawn by `add_part`."""

    role: str  # "terrace", "deck" or "wall"
    field: object  # the playing surface the stadium looks onto: a shapely geometry
    seat: str = SEAT  # the colour of seat backs
    tread: str = TREAD
    facade: str = CONCRETE  # the colour of the outside wall
    front: object = None  # terrace: the line along the field the rows run parallel to
    rows: int = 1
    row_depth: float = ROW_DEPTH
    rise: float = ROW_RISE
    blockers: object = None  # terrace: what stands behind or beside it (a wall of the terrace there is hidden)
    scoreboard: bool = False
    topper: bool = False  # wall: seats stand on top of it (a `leisure=bleachers` part: the Green Monster's)


PLANS: dict[int, Plan] = {}  # by id() of the Building
_KEEP: list = []  # the buildings PLANS is keyed by, so that no id is reused while the scene lives


def part_plan(building: Building) -> Plan | None:
    """How a building is drawn here, or None for an ordinary one."""
    return PLANS.get(id(building))


# --- Finding stadiums -------------------------------------------------------------------------------------------------
@dataclass
class _Group:
    ident: str
    outline: Polygon
    tags: dict
    wall: str  # the outline's wall colour
    parts: list[Building] = field(default_factory=list)
    standalone: Building | None = None  # a lone building drawn in place of an outline with parts


def _is_stadium(tags: dict) -> bool:
    return tags.get("building") in STADIUM_BUILDINGS or tags.get("leisure") == "stadium"


def _groups(scene: Scene) -> list[_Group]:
    groups = []
    for outline in scene.outlines:
        if _is_stadium(outline.tags):
            parts = [b for b in scene.buildings if b.part_of == outline.id]
            groups.append(_Group(outline.id, outline.polygon, outline.tags, outline.wall_colour, parts))
    for building in scene.buildings:
        if not building.part and building.tags.get("building") in STADIUM_BUILDINGS:
            groups.append(_Group(building.id, building.polygon, building.tags, building.wall_colour, [], building))
    return groups


def _width_ok(polygon: Polygon, width: float) -> bool:
    """Whether a circle of diameter `width` fits inside the polygon: it is wider than a wall."""
    return not polygon.buffer(-width / 2).is_empty


def _role(part: Building) -> str | None:
    """"terrace", "wall" or "deck" for a part of a stadium, None for what is left as an ordinary building."""
    tags = part.tags
    chosen = (tags.get("stand") or "").lower()
    if chosen == "none":
        return None
    if chosen in ("terrace", "wall", "deck"):
        return chosen
    if part.min_height > 0.5 or tags.get("building:part") == "roof" or part.polygon.area < PART_MIN_AREA:
        return None  # a roof deck floating above the stand, or a small booth
    if tags.get("leisure") == "bleachers" or tags.get("building:part") == "bleachers":
        if tags.get("height") and not _width_ok(part.polygon, WALL_WIDTH):
            return "wall"  # a thin strip with a height of its own is a wall that bleachers stand on (the Green Monster)
        return "terrace"
    return "deck"


def _seat_colours(*tag_sets: dict) -> tuple[str, str]:
    seat = tread = None
    for tags in tag_sets:
        seat = seat or parse_colour(tags.get("seat:colour"))
        tread = tread or parse_colour(tags.get("seat:tread"))
    return seat or SEAT, tread or TREAD


def _clean(geometry, keep_out) -> list[Polygon]:
    """The pieces of a geometry wide enough to be stands: slivers between parts are opened away. Opening with a polygonal
    buffer rounds corners a little off, so what must stay clear (the field, the parts) is taken off again."""
    opened = geometry.buffer(-SLIVER, quad_segs=2).buffer(SLIVER, quad_segs=2).difference(keep_out)
    return [p for p in polygons_of(opened) if p.area >= TERRACE_MIN_AREA]


def augment(scene: Scene) -> None:
    """Find the stadiums in a scene and turn them into stands and parts (see the module's text). Called at the end of
    `features.classify`: it adds Buildings (the stands) to `scene.buildings`, removes the ones that stands replace, and
    records in PLANS how `add_part` is to draw each."""
    PLANS.clear()
    _KEEP.clear()
    fields = [a.polygon for a in scene.areas["green"] if a.tags.get("leisure") == "pitch" and a.polygon.area >= FIELD_MIN_AREA]
    groups = _groups(scene)
    if not fields or not groups:
        return
    tree = STRtree(fields)
    added: list[Building] = []
    removed: set[int] = set()
    for group in groups:
        near = [fields[i] for i in tree.query(group.outline.buffer(FIELD_REACH), predicate="intersects")]
        if not near:
            continue  # an enclosed arena or a gym: no open field to look onto
        play = unary_union(near)
        made = _stands(group, play, added, removed)
        if group.standalone is not None and made:
            removed.add(id(group.standalone))
    if removed:
        scene.buildings = [b for b in scene.buildings if id(b) not in removed]
    scene.buildings.extend(added)


def _stands(group: _Group, play, added: list[Building], removed: set[int]) -> int:
    """Plan the parts and make the stands of one stadium; return how many stands were made."""
    seat, tread = _seat_colours(group.tags)
    parts = {id(p): (p, _role(p)) for p in group.parts}
    blockers, terraced = [], []
    for part, role in parts.values():
        if role in ("deck", "wall"):
            own_seat, own_tread = _seat_colours(part.tags, group.tags)
            bleachers = part.tags.get("leisure") == "bleachers" or part.tags.get("building:part") == "bleachers"
            PLANS[id(part)] = Plan(
                role, play, own_seat, own_tread, part.wall_colour, scoreboard=part.tags.get("stand:scoreboard") == "yes",
                topper=role == "wall" and bleachers,
            )
            _KEEP.append(part)
            blockers.append(part.polygon)
        elif role == "terrace":
            terraced.append(part)
    behind = unary_union(blockers).buffer(0.05, quad_segs=2) if blockers else Polygon()
    heights = [(p.polygon, p.height) for p, role in parts.values() if role in ("deck", "wall")]

    pieces: list[tuple[Polygon, Building | None]] = []
    for part in terraced:  # a bleachers part is a terrace itself
        for piece in polygons_of(part.polygon.difference(play)):
            if piece.area >= TERRACE_MIN_AREA / 4:
                pieces.append((piece, part))
        removed.add(id(part))
    covered = unary_union([p.polygon for p in terraced]) if terraced else Polygon()
    gap = group.outline.difference(play).difference(behind).difference(covered)
    pieces += [(piece, None) for piece in _clean(gap, unary_union([play, behind, covered]))]

    made = 0
    for piece, part in pieces:
        dmin = piece.distance(play)
        if dmin > FIELD_REACH:
            continue
        front = piece.boundary.intersection(play.buffer(dmin + 6.0, quad_segs=3))
        if front.is_empty or front.length < 3.0:
            continue
        depth = max(front.distance(Point(c)) for c in piece.exterior.coords)
        # How high the terrace rises: with the slope of the rows, to the concourse (half the height of the structure behind
        # it, at most CONCOURSE_MAX), or for a bleachers part to its own height.
        slope = ROW_RISE / ROW_DEPTH
        if part is not None:
            ceiling = part.height
        else:
            neighbours = [h for polygon, h in heights if polygon.distance(piece) < NEIGHBOUR]
            ceiling = min(CONCOURSE_MAX, CONCOURSE_SHARE * max(neighbours)) if neighbours else CONCOURSE_MAX
        top = max(FRONT_WALL + ROW_RISE, min(ceiling, FRONT_WALL + slope * depth))
        run = (top - FRONT_WALL) / slope
        rows = max(1, round(run / ROW_DEPTH))
        own_seat, own_tread = _seat_colours(part.tags, group.tags) if part else (seat, tread)
        number = made
        ident = part.id if part is not None else f"stand:{group.ident}#{number}"
        wall = part.wall_colour if part is not None else group.wall
        tags = {**(part.tags if part is not None else {}), "stand": "terrace"}
        stand = Building(ident, piece, top, 0.0, "default", wall, CONCRETE, tags, True, group.ident, group.tags.get("name"))
        PLANS[id(stand)] = Plan(
            "terrace", play, own_seat, own_tread, wall, front, rows, run / rows, (top - FRONT_WALL) / rows,
            behind.union(unary_union([q for q, _ in pieces if q is not piece and q.distance(piece) < 0.2])),
        )
        _KEEP.append(stand)
        added.append(stand)
        made += 1
    return made


# --- Drawing ----------------------------------------------------------------------------------------------------------
def _segments(geometry) -> tuple[np.ndarray, np.ndarray]:
    """The (start, end) points of every straight piece of the lines in a geometry (n x 2 each)."""
    starts, ends = [], []
    for line in getattr(geometry, "geoms", [geometry]):
        if line.geom_type == "LineString" and len(line.coords) >= 2:
            coords = np.asarray(line.coords)[:, :2]
            starts.append(coords[:-1])
            ends.append(coords[1:])
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(starts), np.concatenate(ends)


def _faced(a: np.ndarray, b: np.ndarray, region, towards_region: bool) -> tuple[np.ndarray, np.ndarray]:
    """Order each segment (a, b) so that a wall drawn from a to b faces out of the region (or into it, if `towards_region`).
    A wall from a to b faces to its right: (dy, -dx)."""
    delta = b - a
    length = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-9)
    right = np.column_stack([delta[:, 1], -delta[:, 0]]) / length[:, None]
    middle = (a + b) / 2
    probe = middle + right * 0.06
    inside = shapely.contains_xy(region, probe[:, 0], probe[:, 1])
    flip = inside if not towards_region else ~inside  # the right-hand side is inside the region: that wall faces in
    a2, b2 = np.where(flip[:, None], b, a), np.where(flip[:, None], a, b)
    return a2, b2


def _flat_walls(builder, a: np.ndarray, b: np.ndarray, z0, z1, colour, feature: int) -> None:
    """Vertical quads from a to b (each facing to its right), z0 to z1 (numbers or arrays), in one flat colour."""
    count = len(a)
    if not count:
        return
    low = np.broadcast_to(np.asarray(z0, dtype=float), (count,))
    high = np.broadcast_to(np.asarray(z1, dtype=float), (count,))
    keep = high - low > 1e-3
    if not keep.any():
        return
    a, b, low, high = a[keep], b[keep], low[keep], high[keep]
    count = len(a)
    delta = b - a
    length = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-9)
    normal = np.column_stack([delta[:, 1], -delta[:, 0], np.zeros(count)]) / length[:, None]
    positions = np.stack(
        [np.column_stack([a, low]), np.column_stack([b, low]), np.column_stack([b, high]), np.column_stack([a, high])], axis=1
    )
    indices = np.arange(count)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])
    builder.add(positions.reshape(-1, 3), np.repeat(normal, 4, axis=0), indices, colour=colour, feature=feature)


def _facade_walls(builder, a: np.ndarray, b: np.ndarray, z0: float, z1: float, colour, feature: int, style: int) -> None:
    """Walls from a to b (each facing to its right) with facade UVs, as mesh.add_walls writes them but for the given edges."""
    count = len(a)
    if not count or z1 - z0 < 1e-3:
        return
    delta = b - a
    length = np.hypot(delta[:, 0], delta[:, 1])
    keep = length > 1e-3
    a, b, delta, length = a[keep], b[keep], delta[keep], length[keep]
    count = len(a)
    if not count:
        return
    normal = np.column_stack([delta[:, 1], -delta[:, 0], np.zeros(count)]) / length[:, None]
    bays = np.where(length >= 1.5, np.maximum(1, np.round(length / BAY_WIDTH)), 0.0)
    floors = max(1, round((z1 - z0) / FLOOR_HEIGHT))
    start, end = facade_u(a, length, bays, raised=z0 > 0.5, style=style)
    bottom = np.full(count, float(floors * STOREY_STRIDE if floors <= MAX_STOREYS else 0))
    tops = bottom + float(floors)
    positions = np.stack(
        [
            np.column_stack([a, np.full(count, z0)]), np.column_stack([b, np.full(count, z0)]),
            np.column_stack([b, np.full(count, z1)]), np.column_stack([a, np.full(count, z1)]),
        ],
        axis=1,
    )
    uvs = np.stack(
        [np.column_stack([start, bottom]), np.column_stack([end, bottom]), np.column_stack([end, tops]), np.column_stack([start, tops])],
        axis=1,
    )
    indices = np.arange(count)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])
    builder.add(positions.reshape(-1, 3), np.repeat(normal, 4, axis=0), indices, uvs=uvs.reshape(-1, 2), colour=colour, feature=feature)


def _ring_edges(polygon: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Every edge of a polygon's rings, each from a to b with the polygon on its left (counter-clockwise outside, clockwise
    round holes), so that a wall from a to b faces away from the polygon."""
    polygon = orient(polygon, 1.0)
    starts, ends = [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords)[:, :2]
        starts.append(coords[:-1])
        ends.append(coords[1:])
    return np.concatenate(starts), np.concatenate(ends)


def _faces_field(a: np.ndarray, b: np.ndarray, play) -> np.ndarray:
    """For wall edges that face away from their polygon: which of them face the field and are within reach of it."""
    result = np.zeros(len(a), dtype=bool)
    delta = b - a
    length = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-9)
    normal = np.column_stack([delta[:, 1], -delta[:, 0]]) / length[:, None]
    for i, (p, q) in enumerate(zip(a, b)):
        middle = Point((p + q) / 2)
        distance = middle.distance(play)
        if distance > INSIDE_REACH:
            continue
        near = nearest_points(middle, play)[1]
        ahead = np.array([near.x - middle.x, near.y - middle.y])
        norm = float(np.hypot(*ahead))
        cosine = 1.0 if norm < 1e-6 else float(ahead @ normal[i]) / norm
        result[i] = cosine > INSIDE_FACING or (cosine > -INSIDE_FACING and distance < SIDE_REACH)
    return result


def _shade(hex_colour: str, factor: float) -> np.ndarray:
    return colour_bytes(hex_colour, factor)


def _deck_inside(roofs, a: np.ndarray, b: np.ndarray, height: float, roof: str, feature: int) -> None:
    """The face of a stand's upper deck as seen from the field: a shaded deck, a rail, a fascia and the edge of the roof."""
    band = min(2.2, 0.2 * height)
    fascia = min(3.0, 0.3 * height)
    rail = min(0.9, 0.08 * height)
    top = height - band
    zones = (
        (0.0, top - fascia - rail, colour_bytes(DECK_VOID)),
        (top - fascia - rail, top - fascia, colour_bytes(DECK_RAIL)),
        (top - fascia, top, colour_bytes(DECK_FASCIA)),
        (top, height, _shade(roof, 0.8)),
    )
    for z0, z1, colour in zones:
        _flat_walls(roofs, a, b, z0, z1, colour, feature)


def _scoreboard(roofs, a: np.ndarray, b: np.ndarray, height: float, feature: int) -> None:
    """A panel in the middle of the longest wall face: a dark board with pale lines, as on the Green Monster's hand-operated one."""
    lengths = np.hypot(*(b - a).T)
    i = int(np.argmax(lengths))
    if lengths[i] < 12.0:
        return
    pa, pb = a[i], b[i]
    along = (pb - pa) / lengths[i]
    middle = (pa + pb) / 2 + along * 0.1 * lengths[i]
    half = 5.0
    lo, hi = middle - along * half, middle + along * half
    rise = max(2.0, 0.22 * height)
    z0 = 0.34 * height
    out = np.array([along[1], -along[0]])  # the wall faces to the right of travel
    off = out * 0.05
    _flat_walls(roofs, (lo + off)[None], (hi + off)[None], z0, z0 + 3 * rise / 2, colour_bytes(SCOREBOARD), feature)
    for k in range(3):
        row = z0 + 0.08 + k * rise / 2
        _flat_walls(roofs, (lo + 0.4 * along + off * 2)[None], (hi - 0.4 * along + off * 2)[None], row, row + 0.12, colour_bytes(SCOREBOARD_LINES), feature)


def add_part(walls, roofs, building: Building, plan: Plan, origin, feature: int = 0) -> None:
    """Draw a stand, a field wall or a part of a stadium. `walls` is the tile's building_walls builder (facade UVs), `roofs`
    the building_roofs one (flat colour)."""
    origin = np.asarray(origin, dtype=float)
    if plan.role == "terrace":
        _terrace(walls, roofs, building, plan, origin, feature)
        return
    polygon = building.polygon
    a, b = _ring_edges(polygon)
    inside = _faces_field(a, b, plan.field)
    if plan.role == "wall":
        colour = colour_bytes(building.wall_colour)
        _flat_walls(roofs, a[inside] - origin, b[inside] - origin, 0.0, building.height - 0.5, colour, feature)
        _flat_walls(roofs, a[inside] - origin, b[inside] - origin, building.height - 0.5, building.height, colour_bytes(WALL_CAP), feature)
        _flat_walls(roofs, a[~inside] - origin, b[~inside] - origin, 0.0, building.height, colour, feature)
        if plan.scoreboard:
            _scoreboard(roofs, a[inside] - origin, b[inside] - origin, building.height, feature)
        if plan.topper and inside.any():
            _topper(roofs, polygon, a[inside], b[inside], building.height, plan, origin, feature)
    else:
        _facade_walls(walls, a[~inside] - origin, b[~inside] - origin, 0.0, building.height, colour_bytes(building.wall_colour), feature, FACADE_STYLE_OUTSIDE)
        _deck_inside(roofs, a[inside] - origin, b[inside] - origin, building.height, building.roof_colour, feature)
    from .mesh import add_flat  # a flat roof

    add_flat(roofs, polygon, building.height, origin, colour_bytes(building.roof_colour), feature=feature)


def _bearing(line, at: float, reach: float) -> tuple[float, float] | None:
    """The unit direction of a line round the distance `at` along it, from the points `reach` metres either side."""
    p, r = line.interpolate(max(0.0, at - reach)), line.interpolate(min(line.length, at + reach))
    dx, dy = r.x - p.x, r.y - p.y
    norm = math.hypot(dx, dy)
    return None if norm < 1e-6 else (dx / norm, dy / norm)


def _aisles(piece: Polygon, front, depth: float):
    """Narrow strips running back from the front of a terrace every AISLE_SPACING metres: the stairs between blocks of seats.
    Only along stretches of the front that run straight (a stair at a notch or a corner would cross its neighbours)."""
    lines = [front] if front.geom_type == "LineString" else list(getattr(front, "geoms", []))
    strips = []
    for line in lines:
        if line.geom_type != "LineString" or line.length < AISLE_SPACING * 1.5:
            continue
        count = max(1, int(line.length // AISLE_SPACING))
        for i in range(count):
            at = (i + 0.5) * line.length / count
            near, wide = _bearing(line, at, 2.5), _bearing(line, at, 8.0)
            if near is None or wide is None or near[0] * wide[0] + near[1] * wide[1] < 0.94:  # more than 20 degrees apart: a corner
                continue
            q = line.interpolate(at)
            nx, ny = -wide[1], wide[0]
            if not piece.contains(Point(q.x + nx, q.y + ny)):
                nx, ny = -nx, -ny
                if not piece.contains(Point(q.x + nx, q.y + ny)):
                    continue
            reach = depth + 6.0
            strips.append(LineString([(q.x - nx * 0.5, q.y - ny * 0.5), (q.x + nx * reach, q.y + ny * reach)]).buffer(AISLE_WIDTH / 2, cap_style=2))
    return unary_union(strips).intersection(piece) if strips else Polygon()


def _surface(roofs, polygon, front, rows: int, depth: float, heights: list, seat, tread, origin, feature: int, aisles=None, on_row=None) -> None:
    """The treads and risers of rows + 1 rows of seats on a polygon, running back from `front`, row k at heights[k] (the
    last row runs on to the back of the polygon). Treads and risers in the aisles are drawn in the stairs' colours.
    `on_row(k, piece)` is called for each tread polygon, to draw what stands under it."""
    from .mesh import add_flat

    stairs = aisles if aisles is not None and not aisles.is_empty else None
    cover = [Polygon()] + [polygon.intersection(front.buffer(k * depth, quad_segs=3)) for k in range(1, rows + 1)] + [polygon]
    for k in range(rows + 1):
        row = cover[k + 1].difference(cover[k]) if k else cover[1]
        for piece in polygons_of(row):
            if piece.area < 0.05:
                continue
            piece = piece.simplify(0.08)
            if piece.is_empty or piece.geom_type != "Polygon":
                continue
            if on_row is not None:
                on_row(k, piece)
            steps = piece.intersection(stairs) if stairs is not None else Polygon()
            for part, colour in ((piece.difference(steps) if stairs is not None else piece, tread), (steps, colour_bytes(AISLE))):
                for flat in polygons_of(part):
                    if flat.area >= 0.05:
                        add_flat(roofs, flat, heights[k], origin, colour, feature=feature)
    # The risers: the face of each row that looks at the field, from the tread in front of it up to its own.
    inner = polygon.buffer(-0.04, quad_segs=2)
    for k in range(1, rows + 1):
        line = cover[k].boundary.intersection(inner)
        pieces = [(line, seat)] if stairs is None else [(line.difference(stairs), seat), (line.intersection(stairs), colour_bytes(AISLE_RISER))]
        for part, colour in pieces:
            if part.is_empty:
                continue
            a, b = _segments(part)
            if len(a):
                a, b = _faced(a, b, cover[k], towards_region=True)
                _flat_walls(roofs, a - origin, b - origin, heights[k - 1], heights[k], colour, feature)


def _terrace(walls, roofs, building: Building, plan: Plan, origin: np.ndarray, feature: int) -> None:
    """Rows of seats, with stairs between blocks of them, on a solid base: the walls at its ends, and at its back a facade."""
    polygon = building.polygon
    front, rows = plan.front, plan.rows
    heights = [FRONT_WALL + k * plan.rise for k in range(rows + 1)]
    edge = polygon.boundary
    blockers = plan.blockers
    side = _shade(plan.seat, SIDE_DARKEN)

    def under(k: int, piece: Polygon) -> None:
        # Where the terrace stops (its ends, and its back if this is the concourse), walls down to the ground.
        a, b = _ring_edges(piece)
        middle = (a + b) / 2
        points = shapely.points(middle[:, 0], middle[:, 1])
        on_edge = shapely.distance(points, edge) < 0.08
        if blockers is not None and not blockers.is_empty:
            on_edge &= shapely.distance(points, blockers) > 0.25
        a, b, middle = a[on_edge] - origin, b[on_edge] - origin, middle[on_edge]
        if not len(a):
            return
        if k == rows:
            _facade_walls(walls, a, b, 0.0, heights[k], colour_bytes(plan.facade), feature, FACADE_STYLE_OUTSIDE)
        elif k == 0:
            near = shapely.distance(shapely.points(middle[:, 0] + origin[0], middle[:, 1] + origin[1]), front) < 0.15  # along the field: the front wall
            _flat_walls(roofs, a[near], b[near], 0.0, heights[0], _shade(plan.seat, FASCIA_DARKEN), feature)
            _flat_walls(roofs, a[~near], b[~near], 0.0, heights[0], side, feature)
        else:
            _flat_walls(roofs, a, b, 0.0, heights[k], side, feature)

    aisles = _aisles(polygon, front, rows * plan.row_depth)
    _surface(roofs, polygon, front, rows, plan.row_depth, heights, colour_bytes(plan.seat), colour_bytes(plan.tread), origin, feature, aisles, under)


def _topper(roofs, polygon: Polygon, a: np.ndarray, b: np.ndarray, top: float, plan: Plan, origin: np.ndarray, feature: int) -> None:
    """Rows of seats on top of a field wall, stepping up from the edge that faces the field, behind a low rail."""
    front = unary_union([LineString([p, q]) for p, q in zip(a, b)])
    heights = [top + TOPPER_RAIL + k * TOPPER_RISE for k in range(TOPPER_ROWS + 1)]
    _flat_walls(roofs, a - origin, b - origin, top, top + TOPPER_RAIL, colour_bytes(WALL_CAP), feature)
    _surface(roofs, polygon, front, TOPPER_ROWS, TOPPER_DEPTH, heights, colour_bytes(plan.seat), colour_bytes(plan.tread), origin, feature)


# --- Foul poles and light towers ---------------------------------------------------------------------------------------
def _foul_pole(p, x, y, d, o: _Opts) -> None:
    """A yellow pole with a mesh screen on its fair side (the side `direction` points to), from a metre or two up to the top."""
    height = o.tall(FOUL_POLE_HEIGHT, *FOUL_POLE_RANGE)
    yellow = o.paint(FOUL_POLE_YELLOW)
    prism(p, yellow, x, y, 0.0, height, FOUL_POLE_RADIUS, FOUL_POLE_RADIUS * 0.8, 6, top=True)
    width, thick = FOUL_POLE_SCREEN
    W = _frame(x, y, d)
    cuboid(p, yellow, W, -thick / 2, thick / 2, FOUL_POLE_RADIUS * 0.7, FOUL_POLE_RADIUS * 0.7 + width, 1.5, height, faces="tfrlR")


def _light_tower(p, x, y, d, o: _Opts) -> None:
    """A floodlight tower: four tapering legs with rings and diagonals between them, under a bank of lamps that faces `direction`
    (about 330 triangles). `height` is to the top of the lamps above the ground, and a `min_height` is where the tower's foot is."""
    height = o.tall(LIGHT_TOWER_HEIGHT, *LIGHT_TOWER_RANGE)
    base = parse_length((o.tags or {}).get("min_height")) or 0.0  # a tower on a roof starts at the roof's height
    bank_w, bank_d, bank_h = LIGHT_BANK
    height -= base
    mast = height - bank_h * 0.55
    W = _frame(x, y, d, base)
    steel, housing, lamp = o.paint(TOWER_STEEL), colour_bytes(TOWER_HOUSING), colour_bytes(TOWER_LAMP)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    def at(corner, z):
        half = TOWER_FOOT + (TOWER_HEAD - TOWER_FOOT) * min(1.0, z / mast)
        return (corner[0] * half, corner[1] * half, z)

    levels = [0.0, mast * 0.34, mast * 0.67, mast]
    for corner in corners:
        limb(p, steel, W, at(corner, 0.0), at(corner, mast), 0.09, 0.06, sides=4)
    for z in levels[1:]:
        for i in range(4):
            limb(p, steel, W, at(corners[i], z), at(corners[(i + 1) % 4], z), 0.04, sides=4)
    for k in range(3):
        for i in range(4):
            limb(p, steel, W, at(corners[i], levels[k]), at(corners[(i + 1) % 4], levels[k + 1]), 0.035, sides=4)
    z0 = height - bank_h
    cuboid(p, housing, W, -bank_w / 2, bank_w / 2, -bank_d / 2, bank_d / 2, z0, height, faces="tbrlR")
    face_v = bank_d / 2
    front = np.array(W(0.0, 1.0, 0.0)) - np.array(W(0.0, 0.0, 0.0))
    face(p, housing, [W(-bank_w / 2, face_v, z0), W(bank_w / 2, face_v, z0), W(bank_w / 2, face_v, height), W(-bank_w / 2, face_v, height)], front)
    cols, rows = 5, 3
    for i in range(cols):
        for j in range(rows):
            u0 = -bank_w / 2 + (i + 0.15) * bank_w / cols
            u1 = -bank_w / 2 + (i + 0.85) * bank_w / cols
            v0 = z0 + (j + 0.15) * bank_h / rows
            v1 = z0 + (j + 0.85) * bank_h / rows
            face(p, lamp, [W(u0, face_v + 0.03, v0), W(u1, face_v + 0.03, v0), W(u1, face_v + 0.03, v1), W(u0, face_v + 0.03, v1)], front)


def _size_tag(tags: dict) -> list[float]:
    """The numbers of a `size` tag such as "18*9*4.5" (width, height, depth, metres)."""
    out = []
    for part in str((tags or {}).get("size") or "").replace("x", "*").split("*"):
        number = parse_length(part)
        if number:
            out.append(number)
    return out


def _landmark_sign(p, x, y, d, o: _Opts) -> None:
    """A big board on a frame, facing `direction` (and the other way too, if the data says it faces both). The face is one flat
    colour (the `colour` tag, else pale), and with `sign:face=triangle` (set by a file in fixes/) a plain red triangle over a blue
    band: no lettering is ever drawn. `height` is to its top. About 130 triangles."""
    tags = o.tags or {}
    top = o.tall(30.0, 6.0, 90.0)
    numbers = _size_tag(tags)
    width = numbers[0] if numbers else SIGN_WIDTH
    board = numbers[1] if len(numbers) > 1 else width
    board = min(board, top * 0.8)
    two_sided = ";" in str(tags.get("direction") or "") or str(tags.get("sides") or "") == "2"
    W = _frame(x, y, d)
    z0 = top - board
    frame, leg = colour_bytes(SIGN_FRAME), colour_bytes(SIGN_LEG)
    half = SIGN_DEPTH / 2
    cuboid(p, frame, W, -width / 2, width / 2, -half, half, z0, top, faces="tbrlR")
    for u in (-width / 2 + 1.2, width / 2 - 1.2):  # the legs
        for v in (-half * 0.8, half * 0.8):
            cuboid(p, leg, W, u - 0.2, u + 0.2, v - 0.2, v + 0.2, 0.0, z0, faces="tfrlR")
    for v in (-half * 0.8, half * 0.8):  # a brace across each pair of legs
        limb(p, leg, W, (-width / 2 + 1.2, v, max(0.0, z0 - 3.0)), (width / 2 - 1.2, v, z0), 0.12, sides=4)
    sides = [(1.0, "f")] + ([(-1.0, "b")] if two_sided else [])
    for sign, _name in sides:
        toward = np.array(W(0.0, sign, 0.0)) - np.array(W(0.0, 0.0, 0.0))
        v = sign * (half + 0.02)
        face(p, o.paint(SIGN_FACE), [W(-width / 2, v, z0), W(width / 2, v, z0), W(width / 2, v, top), W(-width / 2, v, top)], toward)
        if tags.get("sign:face") != "triangle":
            continue
        v = sign * (half + 0.05)
        red = [(-0.38 * width, z0 + 0.34 * board), (0.38 * width, z0 + 0.34 * board), (0.0, z0 + 0.92 * board)]
        face(p, colour_bytes(SIGN_RED), [W(u * sign, v, z) for u, z in red], toward)
        band = [(-0.46 * width, z0 + 0.06 * board), (0.46 * width, z0 + 0.06 * board), (0.46 * width, z0 + 0.26 * board), (-0.46 * width, z0 + 0.26 * board)]
        face(p, colour_bytes(SIGN_BLUE), [W(u * sign, v, z) for u, z in band], toward)


SHAPES = {"foul_pole": _foul_pole, "light_tower": _light_tower, "landmark_sign": _landmark_sign}


def interpret(tags: dict) -> tuple | None:
    """`features._object`'s hook (see extras.py): a point tagged `baseball=foul_pole`, a floodlight tower or a landmark sign."""
    if tags.get("baseball") == "foul_pole":
        return "foul_pole", None, None
    mast = tags.get("man_made") == "mast" and str(tags.get("mast:type") or tags.get("tower:type") or "").lower() in ("floodlight", "lighting", "light")
    if tags.get("man_made") == "floodlight" or mast:
        return "light_tower", None, None
    sign_height = parse_length(tags.get("height"))
    if tags.get("advertising") in ("sign", "billboard") and tags.get("tourism") == "attraction" and sign_height and sign_height >= LANDMARK_SIGN_MIN:
        return "landmark_sign", None, None
    return None

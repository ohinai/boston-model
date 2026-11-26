"""Paint on the road: crosswalks, centre lines, lane lines, stop lines, and the worn asphalt they are painted on.

Nothing here moves a road. A road is still the flat strip `tiles.py` draws; this module decides what is
painted on it and what colour it is.

Paint is flat. Every mark is a thin polygon that lies MARK_LIFT metres above the carriageway, in its own layer
(`markings`), and is cut to the carriageway that is actually drawn in the tile, so a crosswalk never hangs over a
pavement. A walker passes straight through the layer (`NOT_SOLID` in web/src/city.ts).

The whole district is planned in one pass (`plan`) from the interpreted scene, so a mark that crosses a tile edge is
the same in both tiles, and every rule is local: it reads the tags and the geometry of the road and nothing else.
A road, crossing or junction that is not understood gets no paint: a bare road is better than a wrong mark.

What is drawn, and the rule for each:

* Crosswalks, from `footway=crossing` ways (and from `highway=crossing` nodes on a road that has no such way).
  `crossing:markings` says the style: `ladder` (two edge lines and bars across), `zebra` (bars), `lines` (the two
  edge lines, the usual American crosswalk). A marked crossing with no style gets `lines`. `crossing=unmarked`,
  `crossing:markings=no`, `surface` and anything else get nothing. Only the stretch over a carriageway is painted.
* A double yellow centre line on a two-way street of `trunk`, `primary`, `secondary` or `tertiary` class, and on a
  two-way `unclassified` or `residential` street that states `lanes` of 2 or more. None where the road has an odd
  number of lanes (a turn lane in the middle: the markings there are not guessable).
* Dashed white lines between lanes that run the same way, on motorways and on the classes above, where `lanes` (or
  `lanes:forward` and `lanes:backward`) says there are two or more.
* A white stop line across the approaching lanes, before a marked crosswalk that is signalised
  (`crossing=traffic_signals`) or that has a `highway=stop` right behind it.
* Nothing on a service road, a track, a roundabout, a carriageway under MIN_WIDTH metres wide, or an unpaved road.
* Repair patches (a trench across the street, a strip along a lane) on main streets, in the same layer a little lower
  than the paint; and the asphalt itself, a dark warm grey that varies slowly from place to place (`add_asphalt`,
  drawn into the `roads` layer by tiles.py).

Lines are drawn only between junctions. Where three or more roads meet, the road that runs straight through
keeps its centre line and the others stop at its edge; where two roads of the same class cross, all four stop.
Every line also stops short of a crosswalk (and of a stop line, which stands in front of it).
"""

from __future__ import annotations

import math
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union
from shapely.strtree import STRtree

from .features import Scene, Way, _parking_sides, parse_length, parse_number, polygons_of
from .groundlook import add_patchy, value_noise
from .mesh import MeshBuilder, add_flat, colour_bytes, srgb_to_linear

# --- Paint --------------------------------------------------------------------------------------------------
# Paint lies this far above the carriageway. The viewer's depth buffer cannot tell two surfaces apart that are
# closer than about z^2 / (near x 2^24) metres at distance z: with the 0.2 m near plane of a walker that is
# 3 mm at 50 m, 1.5 cm at 120 m and 2.5 cm at 300 m, and it only gets coarser with distance. Four centimetres keeps
# paint from flickering out to about 400 m, and is not a step anyone can see from eye level. (The near plane grows
# as the camera climbs, which keeps the ratio as good.)
MARK_LIFT = 0.04
WHITE = "#e9e6dc"  # a little off white: thermoplastic is never clean
YELLOW = "#dcb42b"
MIN_PIECE = 0.01  # square metres; a smaller scrap of paint is dropped

# Crosswalk. A US crosswalk is 6 to 10 feet wide (2 to 3 m), a 12-inch line (0.3 m) either side of the walking
# strip; a bar in a ladder is as wide as a line, and a zebra bar half a metre. The values are Boston's, round.
CROSSWALK_WIDTH, CROSSWALK_LIMITS = 3.0, (2.0, 6.0)  # default and sensible limits of the `width` tag, metres
RAIL = 0.3
RUNG, RUNG_PITCH = 0.4, 1.0
ZEBRA_BAR, ZEBRA_PITCH = 0.5, 1.1
CROSSING_RUN = (2.0, 30.0)  # a stretch of crossing over a carriageway shorter or longer than this is not painted
CROSSING_SKEW = 0.8  # |cos| of the angle between the walk and the road; more than this is a way along the road
CROSSING_BEND = 1.5  # a crossing way that wanders this far from straight over the carriageway is not painted
CROSSING_NODE_REACH = 3.0  # a `highway=crossing` node this close to a crossing way belongs to it
CROSSING_NODE_EXTENT = 3.0  # a node's own crossing is drawn across the road and this much beyond each edge

# Centre and lane lines.
MIN_WIDTH = 5.0  # no markings on a carriageway narrower than this
CENTRE_LINE = 0.15  # each of the two yellow lines
CENTRE_GAP = 0.13
LANE_LINE = 0.15
DASH, GAP = 3.0, 6.0  # a lane line: 3 m of paint, 6 m of road
LANE_WIDTH = 3.2  # the lane width `features._road_width` assumes
PARKING_LANE = 2.2
LANE_LIMITS = (2.6, 4.0)
MIN_RUN = 1.5  # a stretch of line shorter than this is not drawn
MIN_STUB = 1.0  # nor is what is left of a line after a crosswalk or a stop line has cut it back
DASH_KINDS = {"motorway", "trunk", "primary", "secondary", "tertiary"}  # lane lines
CENTRE_KINDS = {"trunk", "primary", "secondary", "tertiary"}  # two-way streets of these classes have a centre line
CENTRE_IF_LANES = {"unclassified", "residential"}  # and these have one when the data says they have two lanes
UNPAVED = {"unpaved", "dirt", "gravel", "ground", "grass", "sand", "mud", "earth", "compacted", "fine_gravel", "pebblestone"}

# Junctions: where three or more carriageways meet.
ARM_LOOK = 3.0  # metres along a road that give its direction at the junction
STRAIGHT = -math.cos(math.radians(25))  # two arms with a dot product below this run straight through
JUNCTION_MARGIN = 1.0  # a line stops this far short of the edge of the road it meets
RANK = {
    "motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4, "unclassified": 5, "residential": 6,
    "motorway_link": 7, "trunk_link": 7, "primary_link": 7, "secondary_link": 7, "tertiary_link": 7,
}
NOT_JUNCTION = {"service", "track", "busway"}  # a driveway or an alley meeting a street is not a junction

# Stop lines.
STOP_LINE = 0.5
STOP_GAP = 1.2  # between a stop line and the crosswalk
STOP_INSET = 0.12  # a stop line does not touch the centre line or the kerb
STOP_JUNCTION = 16.0  # traffic that has just left a junction this close behind a crosswalk stops for the junction, not here
STOP_SIGN_REACH = 15.0  # a stop sign this far behind a crosswalk (on the road) stops the traffic that passes it

# Lines stay this far from a crosswalk's edge (along the road); a crosswalk with a stop line allows for the line too.
CROSSWALK_CLEAR = 1.0
STOP_CLEAR = STOP_GAP + STOP_LINE + 0.6


# --- Asphalt ------------------------------------------------------------------------------------------------
# The carriageway is a dark, slightly warm grey, not the cool slate it was: asphalt is the darkest large surface in
# the USGS NAIP aerial (7 July 2023) and in photographs from the ground. It varies slowly: a few percent in
# brightness over 30 to 90 metres (patching, wear, which pour is older), and a little warmer or cooler with it.
# Nothing is random: the colour is a function of world position, so two builds and two tiles agree.
ASPHALT = "#5f5a52"
ASPHALT_CELL = 16.0  # metres: vertex spacing (see groundlook.add_patchy)
ASPHALT_WAVES = ((90.0, 0.14), (34.0, 0.12), (14.0, 0.06))  # (wavelength in metres, share of the brightness)
ASPHALT_WARMTH = (55.0, 0.07)
ASPHALT_FLOOR = 0.72
# A main street that has been dug up and mended has patches: a trench across it, a strip along a lane, a square where
# a pothole was. Fresh ones are a little darker than the road round them and old ones paler. A few to a street: one
# site every PATCH_EVERY metres of road, and PATCH_SHARE of the sites have one. Which sites, and what each is, come
# from the road's id, so a build is repeatable. Patches lie a little lower than paint. Side streets have none: they
# are dug up less, and it is the arterials a walker sees most of.
PATCH_EVERY, PATCH_SHARE = 55.0, 0.4
PATCH_DARK, PATCH_LIGHT = "#4a4742", "#6b655c"
PATCH_LIFT = 0.02
_ASPHALT_LINEAR = np.array(srgb_to_linear(ASPHALT))


def asphalt_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear RGBA bytes (n, 4) of asphalt at each world position."""
    light = np.ones(len(x))
    for k, (wavelength, amplitude) in enumerate(ASPHALT_WAVES):
        light = light + amplitude * value_noise(x, y, wavelength, seed=71 + k)
    light = np.maximum(light, ASPHALT_FLOOR)
    warm = ASPHALT_WARMTH[1] * value_noise(x, y, ASPHALT_WARMTH[0], seed=91)
    tint = np.column_stack([1 + warm, np.ones(len(x)), 1 - 1.4 * warm])
    rgb = np.clip(_ASPHALT_LINEAR[None, :] * light[:, None] * tint, 0.0, 1.0)
    return np.column_stack([np.round(rgb * 255), np.full(len(x), 255)]).astype(np.uint8)


def add_asphalt(builder: MeshBuilder, polygon: Polygon, z: float, origin) -> None:
    """A piece of carriageway at height z in worn asphalt. `builder` must have been made with colours."""
    add_patchy(builder, polygon, z, origin, asphalt_at, ASPHALT_CELL)


# --- Small helpers ------------------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    n = math.hypot(v[0], v[1])
    return np.asarray(v, dtype=float) / n if n > 1e-9 else np.zeros(2)


def _perp(v) -> np.ndarray:
    """`v` turned a quarter turn to the left."""
    return np.array([-v[1], v[0]])


def _quad(p, along, across, a0: float, a1: float, c0: float, c1: float) -> Polygon:
    """The rectangle from a0 to a1 along the unit vector `along` and c0 to c1 along `across`, measured from p."""
    p = np.asarray(p, dtype=float)
    corners = [(a0, c0), (a1, c0), (a1, c1), (a0, c1)]
    return Polygon([p + along * a + across * c for a, c in corners])


def _lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    return [line for part in getattr(geom, "geoms", []) for line in _lines(part)]


def _tangent(line: LineString, along: float) -> np.ndarray:
    a, b = line.interpolate(max(0.0, along - 1.0)), line.interpolate(min(line.length, along + 1.0))
    return _unit((b.x - a.x, b.y - a.y))


# --- The plan -----------------------------------------------------------------------------------------------
@dataclass
class Plan:
    """Every piece of paint in the district: flat polygons in world metres, each white or yellow."""

    polygons: list[Polygon] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    tree: STRtree | None = None
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))  # marks drawn, by kind

    def add(self, geom, colour: str, kind: str) -> None:
        pieces = [p for p in polygons_of(geom) if p.area >= MIN_PIECE]
        if pieces:
            self.counts[kind] += 1
        for polygon in pieces:
            self.polygons.append(polygon)
            self.colours.append(colour)

    def cut(self, start: int, cutters: list[Polygon]) -> None:
        """Take `cutters` out of the paint added since index `start`."""
        if not cutters:
            return
        tree = STRtree(cutters)
        polygons, colours = self.polygons[:start], self.colours[:start]
        for polygon, colour in zip(self.polygons[start:], self.colours[start:]):
            near = tree.query(polygon, predicate="intersects")
            if len(near):
                polygon = polygon.difference(unary_union([cutters[int(i)] for i in near]))
            for piece in polygons_of(polygon):
                x0, y0, x1, y1 = piece.bounds
                if piece.area >= MIN_PIECE and math.hypot(x1 - x0, y1 - y0) >= MIN_STUB:  # no stumps of a dash
                    polygons.append(piece)
                    colours.append(colour)
        self.polygons, self.colours = polygons, colours

    def finish(self) -> Plan:
        self.tree = STRtree(self.polygons) if self.polygons else None
        return self


def _patches(plan: Plan, way: Way) -> None:
    """The repair patches on one road (see PATCH_EVERY)."""
    if way.kind not in DASH_KINDS:
        return
    for k in range(int(way.line.length // PATCH_EVERY)):
        h = zlib.crc32(f"{way.id}/{k}".encode())
        if h % 100 >= PATCH_SHARE * 100:
            continue

        def unit(shift: int) -> float:  # a repeatable number in [0, 1] for this site
            return ((h >> shift) & 0xFF) / 255.0

        s = (k + 0.15 + 0.7 * unit(8)) * PATCH_EVERY
        base, t = way.line.interpolate(s), _tangent(way.line, s)
        kind = (h >> 16) % 3
        half = way.width / 2
        # (half its length along the road, half its width across it, where across the road its middle is)
        if kind == 0:  # a trench dug across the street and filled
            ahead, abeam, off = 0.35 + 0.2 * unit(0), half - 0.3, 0.0
        elif kind == 1:  # a strip along a lane
            ahead, abeam, off = 1.5 + 2.0 * unit(0), 0.6 + 0.2 * unit(24), (unit(0) - 0.5) * (half - 1.5)
        else:  # a square
            ahead, abeam, off = 0.7 + 0.4 * unit(0), 0.8 + 0.4 * unit(24), (unit(0) - 0.5) * (half - 1.5)
        colour = PATCH_DARK if (h >> 24) & 1 else PATCH_LIGHT
        plan.add(_quad((base.x, base.y), t, _perp(t), -ahead, ahead, off - abeam, off + abeam), colour, "patch")


# --- Road layout --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """The travel lanes of a carriageway, measured across it from the way's centre line (positive to the left of
    the way's direction)."""

    forward: int  # lanes that go the way the way is drawn, on its right
    backward: int  # lanes that go against it, on its left
    lane: float  # width of one lane, metres
    divide: float  # where the forward lanes end and the backward lanes begin (the centre line, if there is one)
    ambiguous: bool  # two-way with an odd number of lanes and no say in which way the odd one goes


def _oneway(tags: dict, kind: str) -> int:
    value = (tags.get("oneway") or "").strip().lower()
    if value in ("yes", "true", "1"):
        return 1
    if value in ("-1", "reverse"):
        return -1
    if value:
        return 0
    return 1 if tags.get("junction") in ("roundabout", "circular") or kind in ("motorway", "motorway_link") else 0


def _count(value) -> int | None:
    number = parse_number(value)
    return int(number) if number is not None and 1 <= number <= 10 else None


def _parked(tags: dict, side: str) -> bool:
    value = (tags.get(f"parking:{side}") or tags.get("parking:both") or "no").strip().lower()
    return value not in ("no", "no_parking", "no_stopping", "no_standing", "separate", "")


def layout(way: Way) -> Layout | None:
    """Where the lanes of a road lie, from `lanes`, `oneway`, `parking:*` and the width features.py gave it; None
    when the road is too narrow to hold them."""
    tags = way.tags
    direction = _oneway(tags, way.kind)
    lanes = _count(tags.get("lanes"))
    ahead, behind = _count(tags.get("lanes:forward")), _count(tags.get("lanes:backward"))
    ambiguous = False
    if direction:
        forward, backward = (lanes or 1, 0) if direction > 0 else (0, lanes or 1)
    elif ahead and behind:
        forward, backward = ahead, behind
    else:
        total = lanes or 2
        forward, backward = total // 2, total - total // 2
        ambiguous = total % 2 == 1
    count = forward + backward
    # The travel lanes share what the parking lanes leave, the way `_road_width` counts it.
    free = way.width - _parking_sides(tags, way.kind) * PARKING_LANE - 0.6
    lane = min(max(free / count, LANE_LIMITS[0]), LANE_LIMITS[1]) if free > 0 else LANE_WIDTH
    # With parking on one side only, the travel lanes lie to the other side of the centre line.
    left, right = _parked(tags, "left"), _parked(tags, "right")
    shift = (right - left) * PARKING_LANE / 2 if left != right else 0.0
    room = way.width / 2 - abs(shift)  # the lanes must fit inside the carriageway
    if count * lane / 2 > room:
        lane = 2 * room / count
    if lane < 1.5:
        return None
    return Layout(forward, backward, lane, shift + (forward - backward) * lane / 2, ambiguous)


def _markable(way: Way) -> bool:
    """Whether a road is wide, paved and important enough to have lines (the rest are left bare)."""
    tags = way.tags
    return (
        (way.kind in DASH_KINDS or way.kind in CENTRE_IF_LANES)
        and way.width >= MIN_WIDTH
        and tags.get("junction") not in ("roundabout", "circular")
        and tags.get("area") != "yes"
        and (tags.get("surface") or "") not in UNPAVED
        and way.line.length >= MIN_RUN
    )


# --- Junctions ----------------------------------------------------------------------------------------------
def _keep(length: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """What is left of [0, length] when each (low, high) in `cuts` is taken out; stretches under MIN_RUN go."""
    spans = [(0.0, length)]
    for low, high in cuts:
        left = []
        for a, b in spans:
            if high <= a or low >= b:
                left.append((a, b))
                continue
            if low > a:
                left.append((a, low))
            if high < b:
                left.append((high, b))
        spans = left
    return [(a, b) for a, b in spans if b - a >= MIN_RUN]


def _junctions(ways: list[Way]) -> tuple[dict[int, list[tuple[float, float]]], list[Point]]:
    """(for each way, the stretches near a junction where its lines stop, the position of every junction).

    Roads meet where they share a vertex (OpenStreetMap ways are split at every junction, or pass through one).
    At a junction the pair of arms that run straight on, the highest class first, goes through; every other arm
    stops short of the edge of the widest road it meets. If a second straight pair of the same class crosses the
    first, nothing goes through. A meeting of fewer than three arms is a bend and changes nothing.
    """
    coords: list[np.ndarray] = []
    along: list[np.ndarray] = []
    nodes: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, way in enumerate(ways):
        c = np.asarray(way.line.coords, dtype=float)[:, :2]
        coords.append(c)
        along.append(np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))]))
        for k, (x, y) in enumerate(c):
            nodes[(round(x * 20), round(y * 20))].append((i, k))

    cuts: dict[int, list[tuple[float, float]]] = defaultdict(list)
    points: list[Point] = []
    for incident in nodes.values():
        arms = []  # (way, direction along it, vertex, unit vector away from the junction)
        for i, k in incident:
            c, s = coords[i], along[i]
            for sign in (-1, 1):
                j = k + sign
                if not 0 <= j < len(c):
                    continue
                while 0 <= j + sign < len(c) and abs(s[j] - s[k]) < ARM_LOOK:
                    j += sign
                v = c[j] - c[k]
                if math.hypot(v[0], v[1]) > 0.05:
                    arms.append((i, sign, k, _unit(v)))
        if len(arms) < 3:
            continue
        i, k = incident[0]
        points.append(Point(*coords[i][k]))

        def klass(pair):
            a, b = (ways[arms[n][0]] for n in pair)
            return max(RANK.get(a.kind, 9), RANK.get(b.kind, 9)), -min(a.width, b.width)

        straight = [p for p in combinations(range(len(arms)), 2) if float(arms[p[0]][3] @ arms[p[1]][3]) < STRAIGHT]
        through: set[int] = set()
        if straight:
            best = min(straight, key=klass)
            through = set(best)
            if any(not set(p) & through and klass(p)[0] <= klass(best)[0] for p in straight):
                through = set()  # two roads of the same class cross: neither runs through
        for a, (i, sign, k, _) in enumerate(arms):
            if a in through:
                continue
            reach = max((ways[arms[b][0]].width for b in range(len(arms)) if b != a), default=0.0) / 2 + JUNCTION_MARGIN
            s0 = float(along[i][k])
            cuts[i].append((s0 - reach, s0) if sign < 0 else (s0, s0 + reach))
    return cuts, points


# --- Crossings ----------------------------------------------------------------------------------------------
MARKED_KINDS = {"marked", "uncontrolled", "traffic_signals", "pedestrian_signals", "zebra", "pelican", "puffin", "toucan"}


def crossing_style(tags: dict) -> str | None:
    """How a crossing is painted: "ladder", "zebra", "lines", or None for nothing (unmarked, or marked some other way)."""
    marks = (tags.get("crossing:markings") or "").split(";")[0].strip().lower()
    kind = (tags.get("crossing") or "").split(";")[0].strip().lower()
    if marks in ("no", "none") or kind in ("unmarked", "informal", "no"):
        return None
    if marks.startswith("ladder"):
        return "ladder"
    if marks.startswith("zebra") or kind == "zebra":
        return "zebra"
    if marks in ("lines", "line", "dashes", "dashed", "yes", "") and (marks or kind in MARKED_KINDS):
        return "lines"  # marked, but the data does not say how: the plain two-line crosswalk
    return None


def _signalised(tags: dict) -> bool:
    return tags.get("crossing") == "traffic_signals" or tags.get("crossing:signals") == "yes"


@dataclass
class Crossing:
    start: np.ndarray
    end: np.ndarray
    width: float
    style: str
    signalised: bool


def _node_tags(ways: list[Way], nodes: list) -> list[dict]:
    """Each crossing way's tags, with `crossing`, `crossing:markings` and `crossing:signals` filled in from a
    `highway=crossing` node on it where the way itself does not say."""
    if not nodes:
        return [w.tags for w in ways]
    tree = STRtree([Point(n.x, n.y) for n in nodes])
    merged = []
    for way in ways:
        tags = dict(way.tags)
        for index in sorted(tree.query(way.line, predicate="dwithin", distance=0.8)):
            for key in ("crossing", "crossing:markings", "crossing:signals"):
                if key not in tags and key in nodes[index].tags:
                    tags[key] = nodes[index].tags[key]
        merged.append(tags)
    return merged


def _crossings(scene: Scene, carriageways: list[Way], roads, road_tree: STRtree) -> list[Crossing]:
    """Every marked crossing over a carriageway: one per stretch of a crossing way that lies on the road."""
    ways = [
        w for w in scene.ways
        if w.surface == "path" and not w.bridge and (w.tags.get("footway") == "crossing" or w.tags.get("cycleway") == "crossing")
    ]
    crossing_nodes = [n for n in scene.roadnodes if n.kind == "crossing"]
    entries = list(zip((w.line for w in ways), _node_tags(ways, crossing_nodes)))
    # A crossing mapped only as a node on the road is drawn straight across it.
    owned = STRtree([w.line for w in ways]) if ways else None
    for node in crossing_nodes:
        point = Point(node.x, node.y)
        if owned is not None and len(owned.query(point, predicate="dwithin", distance=CROSSING_NODE_REACH)):
            continue
        road = carriageways[int(road_tree.nearest(point))]
        if road.line.distance(point) > road.width / 2:
            continue
        across = _perp(_tangent(road.line, road.line.project(point))) * (road.width / 2 + CROSSING_NODE_EXTENT)
        entries.append((LineString([(node.x - across[0], node.y - across[1]), (node.x + across[0], node.y + across[1])]), node.tags))

    found = []
    for line, tags in entries:
        style = crossing_style(tags)
        if style is None or line.length < 1.0:
            continue
        width = parse_length(tags.get("width"))
        width = width if width and CROSSWALK_LIMITS[0] <= width <= CROSSWALK_LIMITS[1] else CROSSWALK_WIDTH
        surface = unary_union(roads.near(line))
        for run in _lines(line.intersection(surface)):
            a, b = np.asarray(run.coords[0]), np.asarray(run.coords[-1])
            span = float(np.hypot(*(b - a)))
            if not CROSSING_RUN[0] <= span <= CROSSING_RUN[1]:
                continue
            chord = LineString([a, b])
            if run.length > span * 1.1 or max(chord.distance(Point(p)) for p in run.coords) > CROSSING_BEND:
                continue
            found.append(Crossing(a, b, width, style, _signalised(tags)))
    return found


def _crosswalk(plan: Plan, crossing: Crossing, walk: np.ndarray, length: float) -> None:
    """Paint one crosswalk. `walk` is the unit vector from its start to its end."""
    side = _perp(walk)
    half = crossing.width / 2
    origin = crossing.start

    def part(a0, a1, c0, c1):
        plan.add(_quad(origin, walk, side, a0, a1, c0, c1), WHITE, "crosswalk")

    if crossing.style != "zebra":  # the edge lines of the walking strip
        part(0, length, half - RAIL, half)
        part(0, length, -half, -half + RAIL)
    if crossing.style == "ladder":
        bar, pitch, inner = RUNG, RUNG_PITCH, half - RAIL
    elif crossing.style == "zebra":
        bar, pitch, inner = ZEBRA_BAR, ZEBRA_PITCH, half
    else:
        return
    # Bars as near a pitch apart as will fit, the first and last against the two kerbs.
    count = max(2, round((length - bar) / pitch) + 1)
    step = (length - bar) / (count - 1)
    for n in range(count):
        part(n * step, n * step + bar, -inner, inner)


# --- Stop lines ---------------------------------------------------------------------------------------------
def _stop_lines(
    plan: Plan, crossing: Crossing, centre: np.ndarray, road: Way, junctions: STRtree | None, found: list[Point],
    stops: STRtree | None,
) -> int:
    """A stop line before a crosswalk for each direction of traffic that has to stop at it; how many were drawn."""
    lay = layout(road)
    if lay is None:
        return 0
    line = road.line
    mid = Point(*centre)
    tangent = _tangent(line, line.project(mid))
    drawn = 0
    for sign, lanes in ((1, lay.forward), (-1, lay.backward)):  # traffic along the way, and against it
        if not lanes:
            continue
        heading = tangent * sign
        sideways = _perp(heading)
        stopped = crossing.signalised
        if stopped and junctions is not None:
            for index in junctions.query(mid.buffer(STOP_JUNCTION)):
                relative = np.array([found[int(index)].x, found[int(index)].y]) - centre
                if -STOP_JUNCTION < float(relative @ heading) < -0.5 and abs(float(relative @ sideways)) < 12.0:
                    stopped = False  # it has just left a junction
        if not stopped and stops is not None:
            for index in stops.query(mid.buffer(STOP_SIGN_REACH)):
                p = stops.geometries[int(index)]
                relative = np.array([p.x, p.y]) - centre
                if -STOP_SIGN_REACH < float(relative @ heading) < -0.5 and line.distance(p) < 3.0:
                    stopped = True
        if not stopped:
            continue
        # Across the lanes of this direction, from the centre line (or the edge of the lanes) to the kerb side.
        low, high = (lay.divide - lay.forward * lay.lane, lay.divide) if sign > 0 else (lay.divide, lay.divide + lay.backward * lay.lane)
        target = centre - heading * (crossing.width / 2 + STOP_GAP + STOP_LINE / 2)
        s = line.project(Point(*target))
        base = line.interpolate(s)
        t = _tangent(line, s)
        plan.add(_quad((base.x, base.y), t, _perp(t), -STOP_LINE / 2, STOP_LINE / 2, low + STOP_INSET, high - STOP_INSET), WHITE, "stop line")
        drawn += 1
    return drawn


# --- Lines along the road -----------------------------------------------------------------------------------
def _offset(piece: LineString, distance: float) -> LineString | None:
    """`piece` moved sideways (positive to the left); None where the offset curve folds over itself."""
    if abs(distance) < 1e-6:
        return piece
    lines = _lines(shapely.offset_curve(piece, distance, join_style="mitre", mitre_limit=2.0))
    if len(lines) != 1 or abs(lines[0].length - piece.length) > 0.25 * piece.length + 1.0:
        return None
    return lines[0]


def _strip(curve: LineString, width: float):
    return curve.buffer(width / 2, cap_style="flat", join_style="mitre", mitre_limit=2.0)


def _dashes(curve: LineString) -> list[LineString]:
    """Dashes of DASH metres with GAP between, the pattern centred on the stretch."""
    period = DASH + GAP
    count = int((curve.length + GAP) // period)
    if count < 1:
        return []
    start = (curve.length - (count * period - GAP)) / 2
    return [substring(curve, start + n * period, start + n * period + DASH) for n in range(count)]


def _road_lines(plan: Plan, way: Way, spans: list[tuple[float, float]]) -> None:
    """The centre line and lane lines of one road, along each of the stretches in `spans`."""
    lay = layout(way)
    if lay is None or lay.ambiguous:
        return
    two_way = lay.forward > 0 and lay.backward > 0
    explicit = _count(way.tags.get("lanes")) is not None or _count(way.tags.get("lanes:forward")) is not None
    centre = two_way and (way.kind in CENTRE_KINDS or (way.kind in CENTRE_IF_LANES and explicit))
    # The dividers between lanes that go the same way, as offsets from the way's centre line.
    dashed = []
    if way.kind in DASH_KINDS:
        dashed = [lay.divide - k * lay.lane for k in range(1, lay.forward)] + [lay.divide + k * lay.lane for k in range(1, lay.backward)]
    if not centre and not dashed:
        return
    for low, high in spans:
        piece = substring(way.line, low, high).simplify(0.05)
        if piece.geom_type != "LineString" or piece.length < MIN_RUN:
            continue
        if centre:
            for side in (-1, 1):
                curve = _offset(piece, lay.divide + side * (CENTRE_GAP + CENTRE_LINE) / 2)
                if curve is not None:
                    plan.add(_strip(curve, CENTRE_LINE), YELLOW, "centre line")
        for offset in dashed:
            curve = _offset(piece, offset)
            for dash in _dashes(curve) if curve is not None else ():
                plan.add(_strip(dash, LANE_LINE), WHITE, "lane line")


# --- Putting it together ------------------------------------------------------------------------------------
def plan(scene: Scene, roads) -> Plan:
    """All the paint in the scene. `roads` is a `PolygonIndex` of every carriageway polygon (see tiles.py)."""
    out = Plan()
    carriageways = [w for w in scene.ways if w.surface == "road" and w.kind not in NOT_JUNCTION and w.line.length > 0.5]
    if not carriageways:
        return out.finish()
    road_tree = STRtree([w.line for w in carriageways])
    cuts, found = _junctions(carriageways)
    junctions = STRtree(found) if found else None
    stop_signs = [Point(n.x, n.y) for n in scene.roadnodes if n.kind == "stop"]
    stops = STRtree(stop_signs) if stop_signs else None

    bands: list[Polygon] = []  # what the lines along the road must stay out of
    for crossing in _crossings(scene, carriageways, roads, road_tree):
        centre = (crossing.start + crossing.end) / 2
        mid = Point(*centre)
        road = carriageways[int(road_tree.nearest(mid))]
        if road.line.distance(mid) > road.width / 2 + CROSSING_BEND:
            continue  # on no carriageway that is a road
        walk = crossing.end - crossing.start
        length = float(math.hypot(*walk))
        walk = walk / length
        if abs(float(walk @ _tangent(road.line, road.line.project(mid)))) > CROSSING_SKEW:
            continue  # it runs along the road, not across it
        _crosswalk(out, crossing, walk, length)
        reach = STOP_CLEAR if _stop_lines(out, crossing, centre, road, junctions, found, stops) else CROSSWALK_CLEAR
        bands.append(_quad(crossing.start, walk, _perp(walk), 0, length, -crossing.width / 2 - reach, crossing.width / 2 + reach))

    start = len(out.polygons)
    for i, way in enumerate(carriageways):
        if _markable(way):
            spans = _keep(way.line.length, cuts.get(i, []))
            if spans:
                _road_lines(out, way, spans)
    out.cut(start, bands)  # lines stop short of crosswalks and stop lines
    for way in carriageways:
        if _markable(way):
            _patches(out, way)
    return out.finish()


_colours: dict[str, np.ndarray] = {}


def add_paint(builder: MeshBuilder, paint: Plan | None, tile_box: Polygon, road_surface: list[Polygon], z: float, origin) -> None:
    """The paint in a tile, cut to the tile and to the carriageway that is drawn in it, MARK_LIFT above height z (the road).

    `road_surface` is the polygons of the tile's carriageway; `builder` must have been made with colours.
    """
    if paint is None or paint.tree is None or not road_surface:
        return
    hits = sorted(int(i) for i in paint.tree.query(tile_box, predicate="intersects"))
    if not hits:
        return
    surface = unary_union(road_surface)
    shapely.prepare(surface)
    for i in hits:
        piece = paint.polygons[i].intersection(tile_box)
        if not surface.contains(piece):
            piece = piece.intersection(surface)
        colour = paint.colours[i]
        if colour not in _colours:
            _colours[colour] = colour_bytes(colour)
        lift = PATCH_LIFT if colour in (PATCH_DARK, PATCH_LIGHT) else MARK_LIFT
        for polygon in polygons_of(piece):
            if polygon.area >= MIN_PIECE:
                add_flat(builder, polygon, z + lift, origin, _colours[colour])

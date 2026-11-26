"""Trees whose size and shape follow their species and trunk diameter.

`describe` turns OpenStreetMap-style tags into a TreeShape (what kind of tree, how big);
`add_tree` turns a TreeShape into low-poly geometry. Metres, z up, ground at z = 0.
Tiles are written without normals (the viewer shades every face flat), so the look of a
crown comes from its outline and from per-vertex colours: the lobes of a crown are placed
by an envelope that depends on the form, faces hidden inside neighbouring lobes are dropped,
and each vertex is coloured by its height in the crown, which way it faces and a little noise.
Everything is derived from the tree's seed, so a tree looks the same in every build.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .mesh import MeshBuilder, srgb_to_linear

FORMS = ("round", "oval", "vase", "weeping", "small", "columnar", "conifer", "pine", "spire", "clipped")

# Forms that keep their foliage to the ground (no bare trunk below the crown).
EVERGREEN_FORMS = ("conifer", "spire", "clipped")

BARK = "#6a5846"
_CONIFER_LEAF, _BROADLEAF_LEAF, _CLIPPED_LEAF = "#2f5236", "#55803c", "#386440"

# genus -> (form, mature height m, spread ratio = crown width / height, leaf, bark, common name).
# Typical mature urban trees of the US Northeast: representative figures from standard
# arboricultural references (for example Dirr's Manual of Woody Landscape Plants), rounded;
# not measurements of any tree. The trunk size of each tree decides how close it is to mature.
GENERA: dict[str, tuple] = {
    "Ulmus": ("vase", 26, 0.80, "#3f6b33", BARK, "Elm"),
    "Zelkova": ("vase", 18, 0.80, "#4a7536", BARK, "Zelkova"),
    "Gleditsia": ("vase", 18, 0.80, "#7f9e4a", BARK, "Honeylocust"),
    "Tilia": ("oval", 22, 0.60, "#4f7a3a", BARK, "Linden"),
    "Quercus": ("round", 23, 0.85, "#3d6632", BARK, "Oak"),
    "Acer": ("round", 18, 0.75, "#55823b", BARK, "Maple"),
    "Platanus": ("round", 25, 0.80, "#5a8440", "#a9a58f", "Planetree"),
    "Fagus": ("round", 20, 0.80, "#3f6b33", "#8b8a84", "Beech"),
    "Aesculus": ("round", 19, 0.75, "#46733a", BARK, "Horsechestnut"),
    "Fraxinus": ("oval", 18, 0.70, "#55803c", BARK, "Ash"),
    "Liquidambar": ("oval", 19, 0.60, "#4c7a38", BARK, "Sweetgum"),
    "Styphnolobium": ("round", 15, 0.90, "#628c42", BARK, "Japanese pagoda tree"),
    "Sophora": ("round", 15, 0.90, "#628c42", BARK, "Japanese pagoda tree"),
    "Ginkgo": ("columnar", 18, 0.50, "#7f9e4a", BARK, "Ginkgo"),
    "Pyrus": ("columnar", 11, 0.55, "#416d35", BARK, "Callery pear"),
    "Betula": ("columnar", 14, 0.50, "#7f9e4a", "#d6d2c8", "Birch"),
    "Liriodendron": ("oval", 25, 0.50, "#5b8840", BARK, "Tulip tree"),
    "Salix": ("weeping", 15, 1.00, "#8aa35a", BARK, "Willow"),
    "Malus": ("small", 6, 1.00, "#5f8a42", BARK, "Crabapple"),
    "Prunus": ("small", 8, 1.00, "#55823f", "#4a3530", "Cherry"),
    "Amelanchier": ("small", 6, 0.80, "#648c44", BARK, "Serviceberry"),
    "Cornus": ("small", 6, 1.00, "#5a8a42", BARK, "Dogwood"),
    "Cercis": ("small", 6, 1.00, "#5c8a44", BARK, "Redbud"),
    "Magnolia": ("small", 8, 0.90, "#3f6b33", BARK, "Magnolia"),
    "Syringa": ("small", 7, 0.80, "#4f7d3b", BARK, "Tree lilac"),
    "Crataegus": ("small", 7, 0.90, "#55803c", BARK, "Hawthorn"),
    "Carpinus": ("oval", 11, 0.70, "#4a7838", BARK, "Hornbeam"),
    "Cercidiphyllum": ("oval", 14, 0.70, "#6a9448", BARK, "Katsura"),
    "Gymnocladus": ("oval", 18, 0.70, "#5c8641", BARK, "Kentucky coffeetree"),
    "Cladrastis": ("round", 12, 0.90, "#66904a", BARK, "Yellowwood"),
    "Koelreuteria": ("round", 9, 0.90, "#6c9448", BARK, "Goldenrain tree"),
    "Robinia": ("oval", 16, 0.60, "#6a9448", BARK, "Black locust"),
    "Catalpa": ("round", 15, 0.70, "#5d8a45", BARK, "Catalpa"),
    "Nyssa": ("oval", 14, 0.60, "#3f6b33", BARK, "Tupelo"),
    "Celtis": ("round", 16, 0.80, "#55823c", BARK, "Hackberry"),
    "Populus": ("columnar", 22, 0.45, "#6f9647", BARK, "Poplar"),
    "Carya": ("oval", 20, 0.60, "#4d7a3a", BARK, "Hickory"),
    "Juglans": ("round", 20, 0.85, "#55823c", BARK, "Walnut"),
    "Morus": ("round", 10, 1.00, "#5a8641", BARK, "Mulberry"),
    "Ailanthus": ("round", 16, 0.80, "#5c8a44", BARK, "Tree of heaven"),
    "Ostrya": ("oval", 10, 0.70, "#4a7838", BARK, "Hophornbeam"),
    "Phellodendron": ("round", 12, 1.00, "#5c8a44", BARK, "Corktree"),
    "Sassafras": ("round", 15, 0.70, "#5c8a44", BARK, "Sassafras"),
    # Conifers. "conifer" is the tiered cone of spruce, fir and hemlock (and the soft cones of the
    # deciduous conifers); "pine" is a bare trunk carrying flat pads of needles; "spire" is a narrow
    # dense column (arborvitae, juniper, false cypress); "clipped" is a dome that is cut to shape.
    "Metasequoia": ("conifer", 25, 0.35, "#4e7a3c", BARK, "Dawn redwood"),
    "Taxodium": ("conifer", 20, 0.35, "#5a8442", BARK, "Bald cypress"),
    "Pinus": ("pine", 18, 0.50, "#2f5236", BARK, "Pine"),
    "Picea": ("conifer", 18, 0.40, "#2c4f3a", BARK, "Spruce"),
    "Abies": ("conifer", 16, 0.40, "#2f5236", BARK, "Fir"),
    "Tsuga": ("conifer", 15, 0.45, "#2f5a3a", BARK, "Hemlock"),
    "Larix": ("conifer", 18, 0.40, "#4e7a3c", BARK, "Larch"),
    "Thuja": ("spire", 10, 0.35, "#32593a", BARK, "Arborvitae"),
    "Juniperus": ("spire", 8, 0.40, "#3a5f40", BARK, "Juniper"),
    "Cedrus": ("pine", 18, 0.60, "#3a5e45", BARK, "Cedar"),
    "Chamaecyparis": ("spire", 12, 0.35, "#35593c", BARK, "False cypress"),
    "Ilex": ("spire", 9, 0.55, "#2f5a3a", BARK, "Holly"),
    "Taxus": ("clipped", 6, 0.80, "#35603c", BARK, "Yew"),
    "Buxus": ("clipped", 3, 1.00, "#3a6a42", BARK, "Boxwood"),
}

# Species that differ from their genus, keyed on the lower-cased botanical name prefix:
# (form, mature height, spread ratio, common name or None to keep the genus name).
# Same kind of figures as GENERA: typical mature urban size from general arboricultural
# knowledge, used only where the genus row would be clearly off. A `height` or `diameter_crown`
# tag always wins over these.
SPECIES: dict[str, tuple] = {
    "acer palmatum": ("small", 5, 1.0, "Japanese maple"),
    "acer griseum": ("small", 7, 0.8, "Paperbark maple"),
    "acer ginnala": ("small", 6, 1.0, None),
    "acer tataricum": ("small", 6, 1.0, None),
    "acer campestre": ("round", 10, 0.9, "Hedge maple"),
    "acer saccharinum": ("round", 22, 0.8, "Silver maple"),
    "ulmus parvifolia": ("round", 14, 0.9, "Lacebark elm"),
    "quercus palustris": ("oval", 21, 0.65, "Pin oak"),
    "quercus phellos": ("oval", 20, 0.65, "Willow oak"),
    "salix alba": ("weeping", 18, 0.9, "White willow"),
    "salix babylonica": ("weeping", 16, 1.1, "Weeping willow"),
    "salix sepulcralis": ("weeping", 18, 1.1, "Weeping willow"),
    "fagus sylvatica": ("round", 20, 0.85, "European beech"),
    "prunus serotina": ("oval", 18, 0.6, "Black cherry"),
    "prunus x yedoensis": ("small", 9, 1.2, "Yoshino cherry"),
}

# Lowest foliage as a share of tree height, by form.
CROWN_BASE = {
    "vase": 0.42, "round": 0.30, "oval": 0.28, "columnar": 0.25, "weeping": 0.12, "small": 0.30,
    "conifer": 0.12, "pine": 0.25, "spire": 0.05, "clipped": 0.03,
}

# Trunk diameter (cm) to assume when no girth is mapped.
DENOTATION_CM = {"park": 45, "garden": 45, "avenue": 45, "landmark": 45, "street": 25, "urban": 25}
DEFAULT_CM = 35

# An unnamed needle-leaved tree that is at least this wide for its height is a clipped
# evergreen (a dome), not a spruce: a spruce is a third to a half as wide as it is tall.
CLIPPED_RATIO = 0.65


@dataclass(frozen=True)
class TreeShape:
    form: str  # one of FORMS
    height: float  # total, metres
    crown_width: float  # widest spread of the foliage, metres
    crown_base: float  # height of the lowest foliage, metres
    trunk_radius: float  # at the ground, metres
    leaf: str  # "#rrggbb"
    bark: str  # "#rrggbb"
    name: str | None  # common name for people, e.g. "American elm"
    measured: bool  # True when height came from a height tag rather than an estimate


def _unit(seed: int, k: int) -> float:
    """A deterministic value in [0, 1) for (seed, k): integer hashing, no global state."""
    x = (seed * 0x9E3779B1 + k * 0x85EBCA6B + 0x165667B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x / 2**32


def _vary(seed: int, k: int, amount: float) -> float:
    """A multiplier in 1 +- amount."""
    return 1 + amount * (2 * _unit(seed, k) - 1)


def _number(text) -> float | None:
    """First number in a tag value such as "12 m" or "40'" (feet are converted)."""
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(text or ""))
    if not match:
        return None
    value = float(match.group().replace(",", "."))
    return value * 0.3048 if re.search(r"ft|feet|'", str(text)) else value


def describe(tags: dict, seed: int = 0) -> TreeShape:
    botanical = (tags.get("species") or tags.get("taxon") or "").strip().lower()
    # The species says more than the genus, so it wins where a fix has corrected one and not the other.
    named = botanical.split(" ")[0].capitalize() if botanical else ""
    genus = named if named in GENERA else ((tags.get("genus") or "").strip() or named).capitalize()
    row = GENERA.get(genus)
    exception = next((v for k, v in sorted(SPECIES.items(), key=lambda kv: -len(kv[0])) if botanical.startswith(k)), None)

    unnamed_conifer = False
    if row is None and exception is None:
        if tags.get("leaf_type") == "needleleaved":
            form, mature, spread, leaf, bark, common = "conifer", 16, 0.4, _CONIFER_LEAF, BARK, None
            unnamed_conifer = True
        else:
            form, mature, spread, leaf, bark, common = "round", 18, 0.75, _BROADLEAF_LEAF, BARK, None
    else:
        form, mature, spread, leaf, bark, common = row or ("round", 18, 0.75, _BROADLEAF_LEAF, BARK, None)
        if exception:
            form, mature, spread = exception[:3]
            common = exception[3] or common
    # A weeping cultivar of any genus (inventory names such as "Weeping European beech" or a
    # botanical name with 'Pendula') is drawn as a weeping tree, keeping the genus's size and colour.
    cultivar = f"{botanical} {str(tags.get('species:en') or '').lower()}"
    # 'Pendula' counts as a cultivar name only: as a species epithet (Betula pendula, the
    # silver birch) it does not mean a weeping tree.
    words = botanical.replace("'", " ").replace('"', " ").split()
    if "weeping" in cultivar or "pendula" in words[2:]:
        form = "weeping"
    if form == "weeping":
        spread = max(spread, 1.05)  # a mature weeping tree is at least as wide as it is tall
    name = tags.get("species:en") or common

    stated = _number(tags.get("height"))
    measured = stated is not None and 1.5 <= stated <= 60
    crown = _number(tags.get("diameter_crown"))
    has_crown = crown is not None and 1 <= crown <= 40
    # A tree known only by its crown width (one seen in a photograph) has no trunk or height to go by.
    # The default trunk would make it tall and narrow, so size it from the crown instead.
    from_crown = has_crown and not measured and _number(tags.get("circumference")) is None

    # An evergreen with no species whose measured height and crown make it nearly as wide as
    # it is tall is a clipped dome, such as the evergreens along the Public Garden's walks.
    if unnamed_conifer and measured and has_crown and crown / stated >= CLIPPED_RATIO:
        form, leaf = "clipped", _CLIPPED_LEAF

    # Trunk diameter at breast height, cm.
    girth = _number(tags.get("circumference"))
    if from_crown:
        d = crown * 3.5  # the inverse of the crown cap below (0.28 d + 1.5 is about the crown width)
    elif girth is not None and 0.05 <= girth <= 8:
        d = girth / math.pi * 100
    else:
        d = DENOTATION_CM.get(str(tags.get("denotation", "")).lower(), DEFAULT_CM) * _vary(seed, 1, 0.2)

    # Height: mapped, else a saturating curve of trunk diameter towards the mature height.
    if measured:
        height = stated
    elif from_crown:
        height = max(2.0, min(mature, crown / spread))  # the crown is the evidence: invert the spread ratio
    else:
        height = 1.3 + (mature - 1.3) * (1 - math.exp(-d / (1.9 * mature)))
        height = max(2.0, height * _vary(seed, 2, 0.06))

    # Crown: mapped, else height times the genus spread; a thin young trunk cannot carry a wide crown.
    if not has_crown:
        # A weeping crown is carried on long drooping branches, so the trunk cap is looser for it.
        cap = 0.45 * d + 2.0 if form == "weeping" else 0.28 * d + 1.5
        crown = max(1.2, min(height * spread * _vary(seed, 3, 0.08), cap))

    # Park and street trees are limbed up, except conifers (branches to the ground).
    base = CROWN_BASE[form] * height
    if form == "weeping":
        base = min(1.75, 0.45 * height)  # curtains hang to head height whatever the tree's size
    elif form not in EVERGREEN_FORMS:
        base = max(base, min(2.2, 0.45 * height))

    # Root flare makes the base wider than the diameter at breast height.
    radius = min(1.2, max(0.05, 0.6 * d / 100))
    return TreeShape(form, round(height, 2), round(crown, 2), round(base, 2), round(radius, 3), leaf, bark, name, measured)


# --- geometry ---------------------------------------------------------------------------------

_PHI = (1 + 5**0.5) / 2
_ICO = np.array([(-1, _PHI, 0), (1, _PHI, 0), (-1, -_PHI, 0), (1, -_PHI, 0), (0, -1, _PHI), (0, 1, _PHI),
                 (0, -1, -_PHI), (0, 1, -_PHI), (_PHI, 0, -1), (_PHI, 0, 1), (-_PHI, 0, -1), (-_PHI, 0, 1)], float)
_ICO /= np.linalg.norm(_ICO, axis=1)[:, None]
_ICO_FACES = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                       (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                       (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
# A cheaper 6-vertex, 8-face template, for a `coarse` lobe.
_OCT = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
_OCT_FACES = np.array([(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)])


def _subdivide(unit: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every face in four, putting the new vertices on the unit sphere."""
    verts, cache, out = [tuple(v) for v in unit], {}, []

    def middle(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = (np.array(verts[a]) + np.array(verts[b])) / 2
            verts.append(tuple(m / np.linalg.norm(m)))
            cache[key] = len(verts) - 1
        return cache[key]

    for a, b, c in faces:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, float), np.array(out)


# 42 vertices, 80 faces: for a `fine` lobe.
_ICO1, _ICO1_FACES = _subdivide(_ICO, _ICO_FACES)


def _poled_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """An icosahedron with a vertex at each pole (the template above has none), for crowns that close at the top."""
    z = 1 / 5**0.5
    ring = math.sqrt(1 - z * z)
    upper = [(ring * math.cos(2 * math.pi * k / 5), ring * math.sin(2 * math.pi * k / 5), z) for k in range(5)]
    lower = [(ring * math.cos(2 * math.pi * k / 5 + math.pi / 5), ring * math.sin(2 * math.pi * k / 5 + math.pi / 5), -z) for k in range(5)]
    verts = np.array([(0, 0, 1), *upper, *lower, (0, 0, -1)], float)
    faces = []
    for k in range(5):
        j = (k + 1) % 5
        faces += [(0, 1 + k, 1 + j), (1 + k, 6 + k, 1 + j), (6 + k, 6 + j, 1 + j), (11, 6 + j, 6 + k)]
    return verts, np.array(faces)


# 12 and 42 vertices; 20 and 80 faces.
_POLED, _POLED_FACES = _poled_icosahedron()
_POLED1, _POLED1_FACES = _subdivide(_POLED, _POLED_FACES)


@dataclass
class _Part:
    kind: str  # "canopy", "trunk", "limb", "cone" or "skirt"
    positions: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    centre: np.ndarray | None = None  # canopy lobes: the point every face must face away from
    radii: np.ndarray | None = None  # canopy lobes: the ellipsoid it was drawn from (to hide faces inside others)
    colours: np.ndarray | None = None
    light: np.ndarray | None = None  # per vertex 0..1, how sunlit to paint it (0 deep shade, 1 sunlit tip)
    bulge: np.ndarray | None = None  # per vertex: how far a lobe was pushed out (+) or in (-); out is lit, in is shaded
    skin: bool = False  # the crown's main body, which the lobes of a crown are seated on


def _lobe(centre, radii, seed: int, coarse: bool = False, fine: bool = False, bump: float = 0.12):
    """An ellipsoid-ish blob: (positions, normals, faces). 12 vertices / 20 faces (6 / 8 if coarse, 42 / 80 if fine)."""
    unit, faces = (_OCT, _OCT_FACES) if coarse else (_ICO1, _ICO1_FACES) if fine else (_ICO, _ICO_FACES)
    unit = unit.copy()
    unit[:, 2] /= unit[:, 2].max()  # so the radius in z is the true top and bottom of the lobe
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    jitter = 1 + rng.uniform(-bump, bump, len(unit))  # irregular outline
    angle = rng.uniform(0, 2 * math.pi)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    radii = np.asarray(radii, float)
    positions = np.asarray(centre, float) + (unit * radii * jitter[:, None]) @ rotation.T
    normals = (unit / radii) @ rotation.T
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # Whatever the template's winding, make every face point away from the centre.
    a, b, c = (positions[faces[:, i]] for i in range(3))
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3 - centre) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return positions, normals, faces


def _stem(points, radii, sides: int, phase: float = 0.0, upright: bool = False):
    """A tapering open tube through `points` (radius `radii` at each): (positions, normals, faces).

    Rings run counter-clockwise seen from the far end, so every face points outward. An upright tube's
    rings lie flat, whatever way it leans, so its ends are at exactly the heights given."""
    pts = np.asarray(points, float)
    count = len(pts)
    tangent = np.zeros_like(pts)
    tangent[1:-1] = pts[2:] - pts[:-2]
    tangent[0], tangent[-1] = pts[1] - pts[0], pts[-1] - pts[-2]
    if upright:
        tangent[:] = (0.0, 0.0, 1.0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    mean = tangent.mean(axis=0)
    ref = np.array([1.0, 0, 0]) if abs(mean[0]) / np.linalg.norm(mean) < 0.9 else np.array([0, 1.0, 0])
    theta = phase + 2 * math.pi * np.arange(sides) / sides
    rings, radial_all = [], []
    for point, w, radius in zip(pts, tangent, radii):
        u = ref - ref.dot(w) * w
        u /= np.linalg.norm(u)
        v = np.cross(w, u)  # u x v = w
        radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
        rings.append(point + radius * radial)
        radial_all.append(radial)
    i = np.arange(sides)
    nxt = (i + 1) % sides
    faces = []
    for ring in range(count - 1):
        lo, hi = ring * sides, (ring + 1) * sides
        faces.append(np.column_stack([lo + i, lo + nxt, hi + nxt]))
        faces.append(np.column_stack([lo + i, hi + nxt, hi + i]))
    return np.concatenate(rings), np.concatenate(radial_all), np.concatenate(faces)


def _tube(p0, p1, r0: float, r1: float, sides: int, phase: float = 0.0):
    """A tapering open tube from p0 (radius r0) to p1 (radius r1): (positions, normals, faces)."""
    return _stem([p0, p1], [r0, r1], sides, phase)


def _lathe(profile, height: float, radius: float, base: float, sides: int, seed: int, jitter: float = 0.06):
    """A body of revolution: rings at (z fraction, radius fraction) from the bottom up, the last at a point.

    Returns (positions, normals, faces); the outline is a little irregular."""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    span = height - base
    theta = 2 * math.pi * np.arange(sides) / sides + rng.uniform(0, 2 * math.pi)
    rings, apex = [], None
    for zf, rf in profile:
        z = base + zf * span
        if rf <= 0:
            apex = np.array([0.0, 0.0, z])
            break
        r = radius * rf * (1 + rng.uniform(-jitter, jitter, sides))
        rings.append(np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(sides, z)]))
    positions = np.concatenate(rings + ([apex[None]] if apex is not None else []))
    i = np.arange(sides)
    nxt = (i + 1) % sides
    faces = []
    for k in range(len(rings) - 1):
        lo, hi = k * sides, (k + 1) * sides
        faces.append(np.column_stack([lo + i, lo + nxt, hi + nxt]))
        faces.append(np.column_stack([lo + i, hi + nxt, hi + i]))
    if apex is not None:
        lo = (len(rings) - 1) * sides
        faces.append(np.column_stack([lo + i, lo + nxt, np.full(sides, len(positions) - 1)]))
    centre = np.array([0.0, 0.0, base + 0.5 * span])
    out = (positions - centre) / np.array([radius, radius, max(span / 2, 1e-6)])
    return positions, out / np.linalg.norm(out, axis=1)[:, None], np.concatenate(faces)


def _compact(part: _Part) -> None:
    """Drop the vertices that no face uses."""
    used, inverse = np.unique(part.faces, return_inverse=True)
    part.positions, part.normals = part.positions[used], part.normals[used]
    part.light = None if part.light is None else part.light[used]
    part.bulge = None if part.bulge is None else part.bulge[used]
    part.faces = inverse.reshape(-1, 3)


def _hide_inside(parts: list[_Part], scale: float = 0.66) -> None:
    """Drop the faces of canopy lobes that lie wholly inside another lobe (nobody can see them), and the
    vertices only they used. This is what pays for having many lobes.

    A lobe's faces are never nearer its centre than about 0.7 of its radius, so a face whose corners all sit
    inside 0.66 of another lobe's radii is inside that lobe, and no hole opens in the crown."""
    lobes = [p for p in parts if p.kind == "canopy" and p.radii is not None]
    for index, part in enumerate(lobes):
        hidden = np.zeros(len(part.faces), bool)
        for other_index, other in enumerate(lobes):
            if other_index != index:
                d = (part.positions - other.centre) / (scale * other.radii)
                inside = np.einsum("ij,ij->i", d, d) < 1
                hidden |= inside[part.faces].all(axis=1)
        part.faces = part.faces[~hidden]
        _compact(part)
    parts[:] = [p for p in parts if len(p.faces)]


def _lean(seed: int, height: float) -> np.ndarray:
    """How far the top of the trunk leans from the vertical, in metres east and north."""
    angle = _unit(seed, 12) * 2 * math.pi
    amount = 0.02 * height * _unit(seed, 13)
    return np.array([amount * math.cos(angle), amount * math.sin(angle)])


def _trunk(shape: TreeShape, top: float, seed: int, lean: np.ndarray, sides: int | None = None) -> _Part:
    """The trunk: a flare at the base, then a taper, leaning a little."""
    r0 = shape.trunk_radius
    sides = sides or (5 if r0 < 0.25 else 6 if r0 < 0.5 else 8)
    top = max(top, 0.3)
    if top < 1.0:
        zs, rs = [0.0, top], [r0, 0.5 * r0]
    else:
        zs, rs = [0.0, min(0.45, 0.4 * top), top], [r0, 0.78 * r0, 0.5 * r0]
    points = [(*(lean * (z / top) ** 1.6), z) for z in zs]
    p, n, f = _stem(points, rs, sides, _unit(seed, 14) * math.pi, upright=True)
    return _Part("trunk", p, n, f)


def _canopy(centre, radii, seed: int, coarse: bool = False, fine: bool = False, bump: float = 0.10) -> _Part:
    centre, radii = np.asarray(centre, float), np.asarray(radii, float)
    p, n, f = _lobe(centre, radii, seed, coarse, fine, bump)
    return _Part("canopy", p, n, f, centre, radii)


def _skin(shape: TreeShape, seed: int, fine: bool, scale: float = 1.0, bottom: float | None = None) -> _Part:
    """A whole crown as one lumpy skin: a sphere stretched to the form, pushed out by broad bumps
    (sub-crowns) and in by a few dents, every vertex nudged, a little oval from above. Closed at the top
    and bottom. Returns a canopy part whose `bulge` says how far each vertex was pushed out or in."""
    height, R = shape.height, scale * shape.crown_width / 2
    base = shape.crown_base if bottom is None else bottom
    ch, form = height - base, shape.form
    unit, faces = (_POLED1, _POLED1_FACES) if fine else (_POLED, _POLED_FACES)
    rng = np.random.default_rng((seed * 1103515245 + 12345) & 0xFFFFFFFF)
    count = len(unit)
    unit = unit + rng.normal(0, 0.10 if fine else 0.14, (count, 3))  # no two vertices sit where a geodesic ball puts them
    unit /= np.linalg.norm(unit, axis=1)[:, None]

    def directions(n: int, lowest: float, highest: float) -> np.ndarray:
        z = rng.uniform(lowest, highest, n)
        angle = rng.uniform(0, 2 * math.pi, n)
        return np.column_stack([np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z])

    lumps = 4 + (shape.crown_width >= 10) + (shape.crown_width >= 14)
    bumps, dents = directions(lumps, -0.2, 0.95), directions(3, -0.9, 0.0)
    amp, sigma = rng.uniform(0.08, 0.20, lumps), rng.uniform(0.50, 0.90, lumps)
    away = np.arccos(np.clip(unit @ bumps.T, -1, 1))
    bulge = (amp * np.exp(-(away**2) / (2 * sigma**2))).sum(axis=1)
    bulge -= (0.12 * np.exp(-(np.arccos(np.clip(unit @ dents.T, -1, 1)) ** 2) / (2 * 0.7**2))).sum(axis=1)
    bulge += rng.uniform(-0.06, 0.06, count)
    bulge = np.clip(bulge, -0.2, 0.36)  # overlapping bumps never make a spike
    field = 1 + bulge
    t = (unit[:, 2] + 1) / 2
    wide = 0.55 + 0.55 * t**0.9 if form == "vase" else np.ones(count)
    rho = R * wide * field
    low = 0.46 * ch  # the crown is a little fuller above its middle than below it
    z_out = np.where(unit[:, 2] > 0, (ch - low) * unit[:, 2], low * unit[:, 2]) * (1 + 0.6 * bulge)
    xy = unit[:, :2] * rho[:, None]
    oval, turn = 0.10 * rng.random(), rng.uniform(0, math.pi)  # wider one way than the other
    cos, sin = math.cos(turn), math.sin(turn)
    along = xy[:, 0] * cos + xy[:, 1] * sin
    across = -xy[:, 0] * sin + xy[:, 1] * cos
    along, across = along * (1 + oval), across * (1 - oval)
    xy = np.column_stack([along * cos - across * sin, along * sin + across * cos])
    z_abs = np.maximum(base + low + z_out, base)  # no foliage below the crown's base
    top, middle = z_abs.max(), base + low
    if top > height:  # and none above the tree's height: squeeze the upper half to fit rather than flatten it
        up = z_abs > middle
        z_abs[up] = middle + (z_abs[up] - middle) * (height - middle) / (top - middle)
    positions = np.column_stack([xy, z_abs])
    normals = np.column_stack([unit[:, 0] / R, unit[:, 1] / R, unit[:, 2] / (0.5 * ch)])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centre = np.array([0.0, 0.0, base + low])
    a, b, c = (positions[faces[:, i]] for i in range(3))
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3 - centre) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    part = _Part("canopy", positions, normals, faces, centre)
    part.bulge = bulge
    part.skin = True
    return part


# --- broadleaf crowns: round, oval, vase, columnar, small ---------------------------------------------


# How tall a lobe is compared with its width, by form (columnar crowns are made of upright lobes).
_LOBE_FLAT = {"round": 0.78, "oval": 0.95, "vase": 0.72, "columnar": 1.30, "small": 0.82}


def _reach(part: _Part, axis: np.ndarray) -> float:
    """How far from the part's centre its surface is along `axis`: a ray cast at its faces (it is star-shaped)."""
    a, b, c = (part.positions[part.faces[:, i]] for i in range(3))
    e1, e2 = b - a, c - a
    h = np.cross(axis, e2)
    det = np.einsum("ij,ij->i", e1, h)
    safe = np.where(np.abs(det) > 1e-12, det, 1.0)
    s = part.centre - a
    u = np.einsum("ij,ij->i", s, h) / safe
    q = np.cross(s, e1)
    v = (q @ axis) / safe
    t = np.einsum("ij,ij->i", e2, q) / safe
    hit = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if hit.any():
        return float(t[hit].max())
    return float(np.linalg.norm(part.positions - part.centre, axis=1).min())


def _lobes(shape: TreeShape, seed: int, count: int, skin: _Part) -> list[_Part]:
    """Clumps of leaves seated on the skin of a crown, which make its outline lumpy. Each is the outer part of a
    lobe whose centre is on the skin, so it stands out from it like a bump; none is a loose shard. They are
    put at the heights and angles the golden ratio spreads them to, so they never line up, and the last
    one caps the top."""
    R, form = shape.crown_width / 2, shape.form
    flat = _LOBE_FLAT.get(form, 0.8)
    start = _unit(seed, 11) * 2 * math.pi
    height_offset = _unit(seed, 15)
    parts = []
    for k in range(count):
        top = k == count - 1
        up = 0.97 if top else -0.05 + 0.75 * ((0.618034 * k + height_offset) % 1.0)  # how high on the skin: its z component
        if form == "vase":
            up = max(up, 0.15)  # a vase is leafy at the top and bare low down
        angle = start + k * 2.399963 + (_unit(seed, 20 + k) - 0.5) * 0.5
        side = math.sqrt(max(1 - up * up, 0.0))
        axis = np.array([side * math.cos(angle), side * math.sin(angle), up])
        centre = skin.centre + axis * _reach(skin, axis) * 0.96  # on the skin, a little inside
        rx = (0.27 + 0.07 * (1 - abs(up))) * R * _vary(seed, 30 + k, 0.14)
        rz = rx * flat
        part = _canopy(centre, (rx, rx, rz), seed * 31 + k)
        # Keep the outer side of the lobe: the part behind the plane through its centre lies inside the skin.
        corners = [part.positions[part.faces[:, i]] for i in range(3)]
        part.faces = part.faces[(sum(corners) / 3 - centre) @ axis >= -0.20 * rx]
        _compact(part)
        part.bulge = np.full(len(part.positions), 0.08 + 0.10 * _unit(seed, 40 + k))
        parts.append(part)
    return parts


def _broadleaf_parts(shape: TreeShape, seed: int, young: bool) -> list[_Part]:
    height, base = shape.height, shape.crown_base
    ch, crown = height - base, shape.crown_width
    lean = _lean(seed, height)
    if young:  # a thin trunk and one blob
        parts = [_trunk(shape, base + 0.2 * ch, seed, lean)]
        parts.append(_canopy((*lean, base + 0.5 * ch), (crown / 2, crown / 2, max(0.5 * ch, 0.05)), seed * 31))
        return parts
    fork = base - 0.02 * ch  # just under the crown, so the limbs show from underneath
    parts = [_trunk(shape, fork, seed, lean)]
    shift = np.array([*lean, 0.0])
    skin = _skin(shape, seed, fine=crown >= 10 or (shape.form == "columnar" and crown >= 4), scale=0.86)
    parts.append(skin)
    parts += _lobes(shape, seed, 2 if crown < 5 else 3 if crown < 8 else 4 if crown < 12 else 6, skin)
    for part in parts[1:]:
        part.positions = part.positions + shift
        part.centre = part.centre + shift
    # Limbs from the fork out and up to the underside of the crown.
    count = 0 if crown < 5 else 3 if crown < 8 else 4
    r0 = max(0.04, 0.42 * shape.trunk_radius)
    start = _unit(seed, 11) * 2 * math.pi
    for k in range(count):
        angle = start + k * 2 * math.pi / count + (_unit(seed, 20 + k) - 0.5) * 0.6
        reach = 0.42 * crown / 2
        end = np.array([reach * math.cos(angle), reach * math.sin(angle), base + 0.30 * ch]) + shift
        p, n, f = _tube((*lean, fork), end, r0, max(0.025, 0.55 * r0), 3)
        parts.append(_Part("limb", p, n, f))
    return parts


# --- weeping crowns ----------------------------------------------------------------------------------

_HEM_LOW, _HEM_HIGH = 1.2, 3.0  # the ragged hem of a weeping curtain, metres above the ground
_STRAND_GAP = 0.08  # metres between the outward and the inward copy of a strand (see _fringe)


def _fringe(top_z: float, radius: float, count: int, hem_low: float, hem_high: float, seed: int, offset: float):
    """A hanging curtain made of `count` pointed strands round a circle, each reaching its own length down.

    Returns (positions, normals, faces, light), the strands facing out and then the same ones facing in
    (a walker standing under the tree sees the inside); light is 1 at the top edge, 0 at the tips."""
    rng = np.random.default_rng((seed * 7919 + int(radius * 100)) & 0xFFFFFFFF)
    slot = 2 * math.pi / count
    centre = (np.arange(count) + offset + rng.uniform(-0.25, 0.25, count)) * slot
    half = 0.5 * slot * rng.uniform(1.0, 1.3, count)
    top_a, top_b = centre - half, centre + half
    hem = rng.uniform(hem_low, hem_high, count)
    tip_r = radius * rng.uniform(0.90, 1.0, count)
    top_z_each = top_z + rng.uniform(-0.04, 0.04, count) * top_z
    left = np.column_stack([radius * np.cos(top_a), radius * np.sin(top_a), top_z_each])
    right = np.column_stack([radius * np.cos(top_b), radius * np.sin(top_b), top_z_each])
    tip = np.column_stack([tip_r * np.cos(centre), tip_r * np.sin(centre), hem])
    positions = np.empty((3 * count, 3))
    positions[0::3], positions[1::3], positions[2::3] = left, right, tip
    radial = np.zeros((3 * count, 3))
    radial[0::3] = np.column_stack([np.cos(top_a), np.sin(top_a), np.zeros(count)])
    radial[1::3] = np.column_stack([np.cos(top_b), np.sin(top_b), np.zeros(count)])
    radial[2::3] = np.column_stack([np.cos(centre), np.sin(centre), np.zeros(count)])
    faces = np.arange(3 * count).reshape(count, 3)
    # Face outward: swap two vertices wherever the winding comes out facing the axis.
    a, b, c = (positions[faces[:, i]] for i in range(3))
    outward = (a + b + c) * np.array([1.0, 1.0, 0.0])
    facing = np.einsum("ij,ij->i", np.cross(b - a, c - a), outward) < 0
    faces[facing] = faces[facing][:, ::-1]
    light = np.zeros(3 * count)
    light[0::3] = light[1::3] = 1.0
    stripe = rng.uniform(-0.18, 0.18, count)  # strands are lighter or darker than their neighbours
    light = np.clip(light * 0.7 + 0.12 + np.repeat(stripe, 3), 0, 1)
    light[2::3] *= 0.4
    n = len(positions)
    inward = faces[:, ::-1] + n
    nearer = positions - _STRAND_GAP * radial
    return (np.concatenate([positions, nearer]), np.concatenate([radial, -radial]),
            np.concatenate([faces, inward]), np.concatenate([light, light]))


def _weeping_parts(shape: TreeShape, seed: int, young: bool) -> list[_Part]:
    """A mature weeping tree: a thick trunk forking into limbs, a broad lumpy dome, and long strands hanging from it
    to head height in two staggered rings, so a walker can stand under it."""
    height, R = shape.height, shape.crown_width / 2
    lean = _lean(seed, height)
    if young:
        return [_trunk(shape, 0.3 * height, seed, lean), _canopy((*lean, 0.6 * height), (R, R, 0.4 * height), seed * 31)]
    start = _unit(seed, 11) * 2 * math.pi
    fork = 0.40 * height
    parts = [_trunk(shape, fork, seed, lean)]
    shift = np.array([*lean, 0.0])
    # The dome: a broad lumpy skin, a ring of lobes on its crown and a ring of "shoulders" that the strands hang from.
    # A small weeping tree (a weeping cherry) gets fewer of each.
    parts.append(_skin(shape, seed, fine=R > 5, scale=0.84, bottom=0.52 * height))
    lobes = []
    upper = 3 + (R > 4) + (R > 7)
    for k in range(upper):
        angle = start + k * 2 * math.pi / upper + (_unit(seed, 20 + k) - 0.5) * 0.5
        rk = 0.46 * R * _vary(seed, 25 + k, 0.12)
        rzk = 0.17 * height * _vary(seed, 35 + k, 0.10)
        o = 0.52 * R * _vary(seed, 45 + k, 0.1)
        lobes.append(((o * math.cos(angle), o * math.sin(angle), height - rzk * (1.05 + 0.5 * _unit(seed, 55 + k))), (rk, rk, rzk)))
    lower = 3 + (R > 4) + (R > 7)
    for k in range(lower):
        angle = start + (k + 0.5) * 2 * math.pi / lower + (_unit(seed, 60 + k) - 0.5) * 0.4
        rk = 0.36 * R * _vary(seed, 65 + k, 0.12)
        rzk = 0.14 * height
        o = 0.74 * R * _vary(seed, 75 + k, 0.06)
        lobes.append(((o * math.cos(angle), o * math.sin(angle), 0.62 * height), (rk, rk, rzk)))
    for k, (centre, radii) in enumerate(lobes):
        part = _canopy(centre, radii, seed * 31 + k)
        part.bulge = np.full(len(part.positions), 0.06 + 0.08 * _unit(seed, 40 + k))
        parts.append(part)
    for part in parts[1:]:
        if part.kind == "canopy":
            part.positions = part.positions + shift
            part.centre = part.centre + shift
    # Four limbs from the fork out and up into the dome, curving.
    r0 = max(0.05, 0.5 * shape.trunk_radius)
    limbs = 3 if R < 4 else 4
    for k in range(limbs):
        angle = start + 0.7 + k * 2 * math.pi / limbs + (_unit(seed, 85 + k) - 0.5) * 0.5
        d = np.array([math.cos(angle), math.sin(angle), 0.0])
        z_end = 0.70 * height
        start_point = np.array([*lean, fork])
        end = start_point + 0.52 * R * d + np.array([0.0, 0.0, z_end - fork])
        middle = start_point + 0.20 * R * d + np.array([0.0, 0.0, 0.55 * (z_end - fork)])  # up first, then outward
        p, n, f = _stem([start_point, middle, end], [r0, 0.7 * r0, 0.4 * r0], 4)
        parts.append(_Part("limb", p, n, f))
    # The strands: an outer ring on the rim and an inner ring staggered between, ragged at the hem.
    outer, inner = int(min(max(round(8 + 3.2 * R), 12), 44)), int(min(max(round(6 + 2.6 * R), 10), 36))
    for radius, count, top, low, high, offset in ((0.90 * R, outer, 0.64, _HEM_LOW, 2.6, 0.0), (0.72 * R, inner, 0.68, 1.6, _HEM_HIGH, 0.5)):
        p, n, f, light = _fringe(top * height, radius, count, low, high, seed, offset)
        parts.append(_Part("skirt", p + shift, n, f, light=light))
    return parts


# --- evergreens ------------------------------------------------------------------------------------


def _conifer_parts(shape: TreeShape, seed: int) -> list[_Part]:
    """A tiered cone, as spruce, fir and hemlock grow: each tier a cone with a ragged, drooping rim."""
    height, base, r = shape.height, shape.crown_base, shape.crown_width / 2
    ch = height - base
    lean = _lean(seed, height)
    tiers = int(min(max(round(3 + ch / 2.6), 3), 8))
    sides = 8 if r > 1.2 else 6
    rng = np.random.default_rng((seed * 40503 + 7) & 0xFFFFFFFF)
    parts = [_trunk(shape, base + 0.3 * ch, seed, lean, sides=5)]
    spacing = 0.88 * ch / tiers
    tier_height = ch * (0.12 * tiers + 0.88) / tiers
    for k in range(tiers):
        f = k / tiers
        rim_r = r * (1 - f) ** 0.85
        z_rim = base + spacing * k
        theta = 2 * math.pi * np.arange(sides) / sides + rng.uniform(0, math.pi / sides)
        radius = rim_r * rng.uniform(0.86, 1.12, sides)
        droop = tier_height * 0.16 * rng.uniform(0.4, 1.4, sides)
        rim = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z_rim - droop])
        apex = np.array([0.0, 0.0, min(z_rim + tier_height, height) if k < tiers - 1 else height])
        positions = np.concatenate([rim, [apex]])
        i = np.arange(sides)
        faces = np.column_stack([i, (i + 1) % sides, np.full(sides, sides)])
        slope = rim_r / max(apex[2] - z_rim, 1e-6)
        n_side = np.column_stack([np.cos(theta), np.sin(theta), np.full(sides, slope)])
        n_side /= np.linalg.norm(n_side, axis=1)[:, None]
        normals = np.concatenate([n_side, [[0, 0, 1]]])
        light = np.concatenate([0.78 + rng.uniform(-0.14, 0.12, sides), [0.12]])
        if k == 0:  # the underside of the lowest tier, for anyone standing beside the tree
            positions = np.concatenate([positions, [[0, 0, z_rim - droop.mean() * 0.5]]])
            cap = np.column_stack([np.full(sides, sides + 1), (i + 1) % sides, i])
            faces = np.concatenate([faces, cap])
            normals = np.concatenate([normals, [[0, 0, -1]]])
            light = np.concatenate([light, [0.05]])
        parts.append(_Part("cone", positions + np.array([*lean * (z_rim / height) ** 1.6, 0.0]), normals, faces, light=light))
    return parts


def _pine_parts(shape: TreeShape, seed: int, young: bool) -> list[_Part]:
    """A bare trunk with flat pads of needles stacked up it, each carried on a limb: pines and cedars."""
    height, base, r = shape.height, shape.crown_base, shape.crown_width / 2
    ch = height - base
    lean = _lean(seed, height)
    if young:
        return _broadleaf_parts(shape, seed, True)
    pads = 5 + (r > 2.5) + (r > 4.0)
    parts = [_trunk(shape, height * 0.92, seed, lean)]
    start = _unit(seed, 11) * 2 * math.pi
    shift = np.array([*lean, 0.0])
    for k in range(pads):
        f = (k + 0.5 + 0.5 * (_unit(seed, 20 + k) - 0.5)) / pads
        rk = r * (1.0 - 0.55 * f) * _vary(seed, 30 + k, 0.12)
        rz = 0.42 * rk
        z = min(base + 0.12 * ch + 0.80 * ch * f, height - rz)
        angle = start + k * 2.399963
        offset = r * 0.30 * (1 - f) * (0.3 + 0.7 * _unit(seed, 40 + k))
        centre = np.array([offset * math.cos(angle), offset * math.sin(angle), z]) + shift
        parts.append(_canopy(centre, (rk, rk, rz), seed * 31 + k))
        end = np.array([0.7 * offset * math.cos(angle), 0.7 * offset * math.sin(angle), z - 0.4 * rz]) + shift
        p, n, f_ = _tube((*lean, max(z - 0.2 * ch, 0.5)), end, max(0.04, 0.3 * shape.trunk_radius), 0.03, 4)
        parts.append(_Part("limb", p, n, f_))
    return parts


def _spire_parts(shape: TreeShape, seed: int) -> list[_Part]:
    """A dense narrow column with a pointed top: arborvitae, juniper, false cypress."""
    height, base, r = shape.height, shape.crown_base, shape.crown_width / 2
    sides = 5 if r < 0.7 else 6 if r < 1.6 else 7
    profile = [(0.0, 0.55), (0.10, 0.90), (0.35, 1.0), (0.62, 0.78), (0.85, 0.42), (1.0, 0.0)]
    if height < 4:
        profile = [(0.0, 0.7), (0.30, 1.0), (0.70, 0.62), (1.0, 0.0)]
    p, n, f = _lathe(profile, height, r, base, sides, seed, 0.08)
    centre = np.array([0.0, 0.0, base + 0.5 * (height - base)])
    return [_trunk(shape, base + 0.2, seed, np.zeros(2), sides=5), _Part("canopy", p, n, f, centre)]


def _clipped_parts(shape: TreeShape, seed: int) -> list[_Part]:
    """A dome cut to shape, dense to the ground: yew and the clipped evergreens along park walks."""
    height, base, r = shape.height, shape.crown_base, shape.crown_width / 2
    sides = 8 if r < 1.0 else 10
    profile = [(0.0, 0.94), (0.12, 1.0), (0.38, 0.92), (0.64, 0.72), (0.86, 0.44), (0.97, 0.2), (1.0, 0.0)]
    p, n, f = _lathe(profile, height, r, base, sides, seed, 0.05)
    centre = np.array([0.0, 0.0, base + 0.5 * (height - base)])
    return [_Part("canopy", p, n, f, centre, None)]


# --- colour ---------------------------------------------------------------------------------------------


def _pack(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB (n, 3), 0..1, as RGBA bytes."""
    out = np.empty((len(rgb), 4), np.uint8)
    out[:, :3] = np.round(np.clip(rgb, 0, 1) * 255)
    out[:, 3] = 255
    return out


def _palette(shape: TreeShape) -> tuple[np.ndarray, np.ndarray]:
    """Linear shade and sunlit colours for the foliage: darker and bluer deep inside, yellower on top."""
    base = np.array(srgb_to_linear(shape.leaf))
    if shape.form in ("conifer", "pine", "spire", "clipped"):
        return base * (0.45, 0.55, 0.65), base * (1.7, 1.55, 1.2)
    return base * (0.40, 0.50, 0.58), base * (1.35, 1.30, 0.85)


def _finish(shape: TreeShape, parts: list[_Part], seed: int) -> list[_Part]:
    """Keep everything inside the stated height and spread, then colour every vertex."""
    height, base, r = shape.height, shape.crown_base, shape.crown_width / 2
    weeping = shape.form == "weeping" and any(p.kind == "skirt" for p in parts)
    rng = np.random.default_rng((seed * 2654435761 + 0x5EED) & 0xFFFFFFFF)
    tint = np.array([_vary(seed, 100, 0.06), _vary(seed, 101, 0.05), _vary(seed, 102, 0.07)])
    shadow, sunlit = _palette(shape)
    bark = np.array(srgb_to_linear(shape.bark))
    low = _HEM_LOW if weeping else base
    span = max(height - low, 1e-6)
    for part in parts:
        if part.kind != "trunk":
            part.positions[:, 2] = np.clip(part.positions[:, 2], 0.0, height)
            rho = np.hypot(part.positions[:, 0], part.positions[:, 1])
            part.positions[:, :2] *= np.minimum(1.0, (1.1 if weeping else 1.25) * r / np.maximum(rho, 1e-9))[:, None]
        z = part.positions[:, 2]
        if part.kind in ("trunk", "limb"):
            shade = np.clip(0.72 + 0.28 * z / max(z.max(), 1e-6), 0.72, 1.0) * (0.95 + 0.1 * rng.random(len(z)))
            part.colours = _pack(bark * shade[:, None] * tint)
            continue
        t = np.clip((z - low) / span, 0, 1)
        if part.light is None:
            # Fake the light inside a crown: darker low down and facing down, brighter on top, with noise.
            light = 0.20 + 0.50 * t + 0.20 * part.normals[:, 2] + (rng.random(len(z)) - 0.5) * 0.30
            if part.bulge is not None:
                light = light + 1.4 * part.bulge
        else:
            light = part.light + 0.30 * (t - 0.5) * (0.4 if part.kind == "skirt" else 0.6)
        light = np.clip(light, 0, 1)
        part.colours = _pack((shadow + (sunlit - shadow) * light[:, None]) * tint)
    return parts


def _assemble(shape: TreeShape, seed: int = 0) -> list[_Part]:
    """All parts of a tree standing at the origin, clamped to its stated height and spread."""
    young = shape.trunk_radius < 0.08
    form = shape.form
    if form == "weeping":
        parts = _weeping_parts(shape, seed, young)
    elif form == "conifer":
        parts = _conifer_parts(shape, seed)
    elif form == "pine":
        parts = _pine_parts(shape, seed, young)
    elif form == "spire":
        parts = _spire_parts(shape, seed)
    elif form == "clipped":
        parts = _clipped_parts(shape, seed)
    else:
        parts = _broadleaf_parts(shape, seed, young)
    _hide_inside(parts)
    return _finish(shape, parts, seed)


BREAST_HEIGHT = 1.3  # metres: where a trunk's size is read (and where a walker meets it)


def trunk_radius(parts: list[_Part], at: float = BREAST_HEIGHT) -> float:
    """Radius in metres of the trunk as drawn, `at` metres above the ground, flare of the roots left out.

    Read off the trunk's own rings (each a flat ring of vertices), so it is what a walker can see and touch,
    whatever the form of the tree. Above the top of the trunk the last ring's radius is returned. A tree
    with no trunk drawn (a clipped evergreen is a dome of foliage to the ground) has none: 0."""
    for part in parts:
        if part.kind != "trunk":
            continue
        z = np.round(part.positions[:, 2], 4)
        levels = np.unique(z)
        radii = []
        for level in levels:
            ring = part.positions[z == level][:, :2]
            radii.append(float(np.hypot(*(ring - ring.mean(axis=0)).T).mean()))
        return float(np.interp(at, levels, radii))
    return 0.0


def add_tree(builder: MeshBuilder, shape: TreeShape, x: float, y: float, seed: int = 0, feature: int = 0) -> float:
    """A tree of `shape` standing at (x, y), relative to the tile origin.

    Returns the radius of its trunk at breast height (`trunk_radius`), for the tile's data file."""
    parts = _assemble(shape, seed)
    for part in parts:
        builder.add(part.positions + np.array([x, y, 0.0]), part.normals, part.faces, colour=part.colours, feature=feature)
    return trunk_radius(parts)

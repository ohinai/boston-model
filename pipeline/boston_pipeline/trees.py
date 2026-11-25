"""Trees whose size and shape follow their species and trunk diameter.

`describe` turns OpenStreetMap-style tags into a TreeShape (what kind of tree, how big);
`add_tree` turns a TreeShape into low-poly geometry. Metres, z up, ground at z = 0.
Every canopy lobe is an indexed mesh with shared vertices. Tiles are written without
normals, so the normals passed along here only record which way is out.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .mesh import MeshBuilder, colour_bytes

FORMS = ("round", "oval", "vase", "weeping", "small", "columnar", "conifer")

BARK = "#5b4a3c"
_CONIFER_LEAF, _BROADLEAF_LEAF = "#2f5236", "#55803c"

# genus -> (form, mature height m, spread ratio = crown width / height, leaf, bark, common name).
# Typical mature urban trees of the US Northeast: representative figures, not measurements.
GENERA: dict[str, tuple] = {
    "Ulmus": ("vase", 26, 0.80, "#3f6b33", BARK, "Elm"),
    "Zelkova": ("vase", 18, 0.80, "#4a7536", BARK, "Zelkova"),
    "Gleditsia": ("vase", 18, 0.80, "#7f9e4a", BARK, "Honeylocust"),
    "Tilia": ("oval", 22, 0.60, "#4f7a3a", BARK, "Linden"),
    "Quercus": ("round", 23, 0.85, "#3d6632", BARK, "Oak"),
    "Acer": ("round", 18, 0.75, "#55823b", BARK, "Maple"),
    "Platanus": ("round", 25, 0.80, "#5a8440", "#a9a58f", "Planetree"),
    "Fagus": ("round", 20, 0.80, "#3f6b33", "#8b8a84", "Beech"),
    "Aesculus": ("oval", 19, 0.75, "#46733a", BARK, "Horsechestnut"),
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
    "Metasequoia": ("conifer", 25, 0.35, "#4e7a3c", BARK, "Dawn redwood"),
    "Taxodium": ("conifer", 20, 0.35, "#5a8442", BARK, "Bald cypress"),
    "Pinus": ("conifer", 18, 0.45, "#2f5236", BARK, "Pine"),
    "Picea": ("conifer", 18, 0.40, "#2c4f3a", BARK, "Spruce"),
    "Abies": ("conifer", 16, 0.40, "#2f5236", BARK, "Fir"),
    "Tsuga": ("conifer", 15, 0.45, "#2f5a3a", BARK, "Hemlock"),
    "Larix": ("conifer", 18, 0.40, "#4e7a3c", BARK, "Larch"),
    "Thuja": ("conifer", 10, 0.35, "#32593a", BARK, "Arborvitae"),
    "Juniperus": ("conifer", 8, 0.40, "#3a5f40", BARK, "Juniper"),
    "Cedrus": ("conifer", 18, 0.55, "#3a5e45", BARK, "Cedar"),
    "Chamaecyparis": ("conifer", 12, 0.35, "#35593c", BARK, "False cypress"),
    "Taxus": ("conifer", 6, 0.80, "#2d4f33", BARK, "Yew"),
}

# Species that differ from their genus, keyed on the lower-cased botanical name prefix:
# (form, mature height, spread ratio, common name or None to keep the genus name).
SPECIES: dict[str, tuple] = {
    "acer palmatum": ("small", 5, 1.0, "Japanese maple"),
    "acer griseum": ("small", 7, 0.8, "Paperbark maple"),
    "acer ginnala": ("small", 6, 1.0, None),
    "acer tataricum": ("small", 6, 1.0, None),
    "acer campestre": ("round", 10, 0.9, "Hedge maple"),
    "acer saccharinum": ("round", 22, 0.8, "Silver maple"),
    "ulmus parvifolia": ("round", 14, 0.9, "Lacebark elm"),
    "quercus palustris": ("oval", 21, 0.65, "Pin oak"),
    "salix alba": ("weeping", 18, 0.9, "White willow"),
    "fagus sylvatica": ("round", 20, 0.85, "European beech"),
    "prunus serotina": ("oval", 18, 0.6, "Black cherry"),
}

# Lowest foliage as a share of tree height, by form.
CROWN_BASE = {"vase": 0.42, "round": 0.30, "oval": 0.28, "columnar": 0.25, "weeping": 0.12, "small": 0.30, "conifer": 0.12}

# Trunk diameter (cm) to assume when no girth is mapped.
DENOTATION_CM = {"park": 45, "garden": 45, "avenue": 45, "landmark": 45, "street": 25, "urban": 25}
DEFAULT_CM = 35


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

    if row is None and exception is None:
        if tags.get("leaf_type") == "needleleaved":
            form, mature, spread, leaf, bark, common = "conifer", 16, 0.4, _CONIFER_LEAF, BARK, None
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
    elif form != "conifer":
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
# A cheaper 6-vertex, 8-face shape for narrow hanging curtains.
_OCT = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
_OCT_FACES = np.array([(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)])


@dataclass
class _Part:
    kind: str  # "canopy", "trunk", "limb" or "cone"
    positions: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    centre: np.ndarray | None = None  # canopy lobes: the point every face must face away from
    colours: np.ndarray | None = None
    tint: np.ndarray | None = None  # per-vertex brightness multiplier (weeping skirt: one value per side)


def _lobe(centre, radii, seed: int, coarse: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An ellipsoid-ish blob: (positions, normals, faces). 12 vertices / 20 faces (6 / 8 if coarse)."""
    unit, faces = (_OCT, _OCT_FACES) if coarse else (_ICO, _ICO_FACES)
    unit = unit.copy()
    unit[:, 2] /= unit[:, 2].max()  # so the radius in z is the true top and bottom of the lobe
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    jitter = 1 + rng.uniform(-0.12, 0.12, len(unit))  # irregular outline
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


def _tube(p0, p1, r0: float, r1: float, sides: int, phase: float = 0.0):
    """A tapering open tube from p0 (radius r0) to p1 (radius r1): (positions, normals, faces)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    w = (p1 - p0) / np.linalg.norm(p1 - p0)
    ref = np.array([1.0, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - ref.dot(w) * w
    u /= np.linalg.norm(u)
    v = np.cross(w, u)  # u x v = w, so rings run counter-clockwise seen from p1
    theta = phase + 2 * math.pi * np.arange(sides) / sides
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    positions = np.concatenate([p0 + r0 * radial, p1 + r1 * radial])
    nxt = (np.arange(sides) + 1) % sides
    i = np.arange(sides)
    faces = np.concatenate([np.column_stack([i, nxt, sides + nxt]), np.column_stack([i, sides + nxt, sides + i])])
    return positions, np.concatenate([radial, radial]), faces


def _cone(z0: float, z1: float, radius: float, sides: int = 8, phase: float = 0.0):
    """A closed cone: ring at z0, apex at z1, flat base cap."""
    theta = phase + 2 * math.pi * np.arange(sides) / sides
    ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(sides, z0)])
    positions = np.concatenate([ring, [[0, 0, z1]], [[0, 0, z0]]])
    i, nxt = np.arange(sides), (np.arange(sides) + 1) % sides
    faces = np.concatenate([
        np.column_stack([i, nxt, np.full(sides, sides)]),  # sides, facing outward and up
        np.column_stack([np.full(sides, sides + 1), nxt, i]),  # base, facing down
    ])
    slope = radius / max(z1 - z0, 1e-6)
    side_n = np.column_stack([np.cos(theta), np.sin(theta), np.full(sides, slope)])
    side_n /= np.linalg.norm(side_n, axis=1)[:, None]
    normals = np.concatenate([side_n, [[0, 0, 1]], [[0, 0, -1]]])
    return positions, normals, faces


_SKIRT_SIDES = 10
_HEM_LOW, _HEM_HIGH = 1.2, 3.0  # the ragged hem of a weeping curtain, metres above the ground


def _skirt(top_z: float, top_r: float, bottom_r: float, seed: int):
    """A hanging curtain: an open tapering tube with a ragged hem, drawn facing out and facing in.

    Returns (positions, normals, faces, tint). The inward copy has its own vertices (so its
    normals can point inward) at the same positions; tint is one value per side."""
    n = _SKIRT_SIDES
    theta = 2 * math.pi * (np.arange(n) + 0.3 * np.array([_unit(seed, 70 + k) - 0.5 for k in range(n)])) / n
    grow = np.array([_vary(seed, 80 + k, 0.04) for k in range(n)])
    hem = np.array([_HEM_LOW + (_HEM_HIGH - _HEM_LOW) * _unit(seed, 60 + k) for k in range(n)])
    cos, sin = np.cos(theta), np.sin(theta)
    bottom = np.column_stack([bottom_r * grow * cos, bottom_r * grow * sin, hem])
    top = np.column_stack([top_r * grow * cos, top_r * grow * sin, np.full(n, top_z)])
    radial = np.column_stack([cos, sin, np.zeros(n)])
    i, nxt = np.arange(n), (np.arange(n) + 1) % n
    out = np.concatenate([np.column_stack([i, nxt, n + nxt]), np.column_stack([i, n + nxt, n + i])])
    positions = np.concatenate([bottom, top])
    normals = np.concatenate([radial, radial])
    tint = np.array([_vary(seed, 90 + k, 0.06) for k in range(n)])
    tint = np.concatenate([tint, tint])
    inward = out[:, ::-1] + 2 * n
    return (np.concatenate([positions, positions]), np.concatenate([normals, -normals]),
            np.concatenate([out, inward]), np.concatenate([tint, tint]))


def _weeping_parts(shape: TreeShape, seed: int) -> list[_Part]:
    """A mature weeping tree: trunk to half height, limbs, a broad flattened dome, a hanging skirt."""
    height, r = shape.height, shape.crown_width / 2
    start = _unit(seed, 11) * 2 * math.pi
    parts: list[_Part] = []
    top = 0.5 * height
    trunk_top = shape.trunk_radius * 0.55
    p, n, f = _tube((0, 0, 0), (0, 0, top), shape.trunk_radius, trunk_top, 6)
    parts.append(_Part("trunk", p, n, f))

    # The dome: one wide main lobe and three overlapping ones pushed outward.
    rz = 0.32 * height
    lobes = [((0, 0, height - rz), (0.9 * r, 0.9 * r, rz), False)]
    for k, coarse in enumerate((False, False, True)):
        angle = start + k * 2 * math.pi / 3 + (_unit(seed, 20 + k) - 0.5) * 0.7
        rk = 0.55 * r * _vary(seed, 25 + k, 0.1)
        lobes.append(((0.36 * r * math.cos(angle), 0.36 * r * math.sin(angle), 0.70 * height),
                      (rk, rk, 0.22 * height), coarse))
    for k, (centre, radii, coarse) in enumerate(lobes):
        c = np.array(centre, float)
        p, n, f = _lobe(c, radii, seed * 31 + k, coarse)
        parts.append(_Part("canopy", p, n, f, c))

    # Three thin limbs from the top of the trunk up into the dome.
    limb_r = 0.6 * trunk_top
    for k in range(3):
        angle = start + 0.9 + k * 2 * math.pi / 3
        end = (0.45 * r * math.cos(angle), 0.45 * r * math.sin(angle), 0.78 * height)
        p, n, f = _tube((0, 0, top), end, limb_r, max(0.03, 0.4 * limb_r), 4)
        parts.append(_Part("limb", p, n, f))

    p, n, f, tint = _skirt(height - rz - 0.05 * height, 0.94 * r, 0.80 * r, seed)
    parts.append(_Part("skirt", p, n, f, tint=tint))
    return parts


def _lobes_for(shape: TreeShape, seed: int) -> list[tuple]:
    """Canopy lobes as (cx, cy, cz, rx, ry, rz, coarse) for the tree's form."""
    height, base = shape.height, shape.crown_base
    r, ch = shape.crown_width / 2, height - base
    pick = int(_unit(seed, 10) * 1000)
    start = _unit(seed, 11) * 2 * math.pi

    def lobe(angle, offset, z, rx, rz, coarse=False):
        return (offset * math.cos(angle), offset * math.sin(angle), base + z * ch, rx, rx, rz * ch, coarse)

    if shape.trunk_radius < 0.08:  # a young tree: one blob
        return [lobe(0, 0, 0.5, r, 0.5)]
    form = shape.form
    if form == "round":
        n = 2 + pick % 2
        lobes = [lobe(0, 0, 0.5, 0.8 * r, 0.5)]
        for k in range(n):
            angle = start + k * 2 * math.pi / n + (_unit(seed, 20 + k) - 0.5) * 0.6
            lobes.append(lobe(angle, 0.55 * r, (0.30, 0.45, 0.35)[k], 0.5 * r, 0.30))
        return lobes
    if form == "oval":
        lobes = [lobe(0, 0, 0.5, 0.72 * r, 0.5), lobe(start, 0.4 * r, 0.24, 0.5 * r, 0.26)]
        if pick % 2:
            lobes.append(lobe(start + math.pi, 0.35 * r, 0.62, 0.45 * r, 0.26))
        return lobes
    if form == "columnar":
        return [lobe(0, 0, 0.42, 0.8 * r, 0.42), lobe(start, 0.1 * r, 0.84, 0.5 * r, 0.16)]
    if form == "vase":  # widest near the top: three lobes around, one over the centre
        lobes = [lobe(start + k * 2 * math.pi / 3, 0.55 * r, 0.58, 0.55 * r, 0.38) for k in range(3)]
        return lobes + [lobe(0, 0, 0.64, 0.55 * r, 0.34)]
    if form == "weeping":
        lobes = [lobe(0, 0, 0.6, 0.78 * r, 0.4), lobe(start, 0.5 * r, 0.4, 0.5 * r, 0.28),
                 lobe(start + math.pi + 0.4, 0.5 * r, 0.4, 0.5 * r, 0.28)]
        z_high = base + 0.45 * ch
        z_low = min(1.8, 0.6 * base)  # curtains hang almost to head height
        count = 5 + pick % 2
        for k in range(count):
            angle = start + k * 2 * math.pi / count + (_unit(seed, 30 + k) - 0.5) * 0.4
            lobes.append((0.8 * r * math.cos(angle), 0.8 * r * math.sin(angle), (z_high + z_low) / 2,
                          0.2 * r, 0.2 * r, (z_high - z_low) / 2, True))
        return lobes
    # small: low and wide, about as broad as it is tall
    lobes = [lobe(0, 0, 0.55, 0.7 * r, 0.45)]
    for k in range(1 + pick % 2):
        lobes.append(lobe(start + k * 2 * math.pi / 3 + (math.pi if k == 0 else 0), 0.5 * r, 0.4, 0.55 * r, 0.35))
    return lobes


def _assemble(shape: TreeShape, seed: int = 0) -> list[_Part]:
    """All parts of a tree standing at the origin, clamped to its stated height and spread."""
    height, base, r = shape.height, shape.crown_base, shape.crown_width / 2
    ch = height - base
    parts: list[_Part] = []
    young = shape.trunk_radius < 0.08

    weeping = shape.form == "weeping" and not young
    top = 0.6 * height if shape.form == "vase" and not young else base + (0.2 if young or shape.form == "conifer" else 0.15) * ch
    if weeping:
        top = 0.5 * height
    trunk_top = (shape.trunk_radius * 0.55)
    if weeping:
        parts = _weeping_parts(shape, seed)
    else:
        p, n, f = _tube((0, 0, 0), (0, 0, top), shape.trunk_radius, trunk_top, 6)
        parts.append(_Part("trunk", p, n, f))

    if weeping:
        pass
    elif shape.form == "conifer" and not young:
        # Three stacked cones, each narrower than the one below and overlapping it.
        starts, radii = (0.0, 0.28, 0.55), (1.0, 0.74, 0.48)
        for k in range(3):
            z0 = base + starts[k] * ch
            z1 = height if k == 2 else z0 + (0.5, 0.45)[k] * ch
            p, n, f = _cone(z0, z1, r * radii[k], 8, _unit(seed, 40 + k) * math.pi / 4)
            parts.append(_Part("cone", p, n, f))
    else:
        lobes = _lobes_for(shape, seed)
        for k, (cx, cy, cz, rx, ry, rz, coarse) in enumerate(lobes):
            centre = np.array([cx, cy, cz])
            p, n, f = _lobe(centre, (rx, ry, max(rz, 0.05)), seed * 31 + k, coarse)
            parts.append(_Part("canopy", p, n, f, centre))
        if shape.form == "vase" and not young:  # three limbs from the fork up to the outer lobes
            limb_r = 0.6 * trunk_top
            for cx, cy, cz, rx, ry, rz, _ in lobes[:3]:
                end = (0.8 * cx, 0.8 * cy, cz - 0.6 * rz)
                p, n, f = _tube((0, 0, 0.45 * height), end, limb_r, max(0.03, 0.4 * limb_r), 4)
                parts.append(_Part("limb", p, n, f))

    # Keep everything inside the stated height and spread, then colour by height in the crown.
    for index, part in enumerate(parts):
        if part.kind != "trunk":
            part.positions[:, 2] = np.clip(part.positions[:, 2], 0.0, height)
            rho = np.hypot(part.positions[:, 0], part.positions[:, 1])
            part.positions[:, :2] *= np.minimum(1.0, (1.1 if weeping else 1.25) * r / np.maximum(rho, 1e-9))[:, None]
        z = part.positions[:, 2]
        if part.kind in ("trunk", "limb"):
            shades = np.clip(0.8 + 0.2 * z / max(top, 1e-6), 0.8, 1.0)
            part.colours = np.array([colour_bytes(shape.bark, s) for s in shades])
        elif weeping:
            # Light dome, darker hanging foliage: x1.08 at the top to x0.70 at the lowest hem.
            t = np.clip((z - _HEM_LOW) / max(height - _HEM_LOW, 1e-6), 0, 1)
            tint = part.tint if part.tint is not None else 1 + 0.05 * (2 * _unit(seed, 50 + index) - 1)
            part.colours = np.array([colour_bytes(shape.leaf, (0.70 + 0.38 * s) * m) for s, m in zip(t, np.broadcast_to(tint, t.shape))])
        else:
            # Fake the shadow inside a canopy: darker low down, brighter at the top.
            t = np.clip((z - base) / max(ch, 1e-6), 0, 1)
            lobe_shade = 1 + 0.05 * (2 * _unit(seed, 50 + index) - 1)  # per-lobe variation
            part.colours = np.array([colour_bytes(shape.leaf, (0.72 + 0.38 * s) * lobe_shade) for s in t])
    return parts


def add_tree(builder: MeshBuilder, shape: TreeShape, x: float, y: float, seed: int = 0, feature: int = 0) -> None:
    """A tree of `shape` standing at (x, y), relative to the tile origin."""
    for part in _assemble(shape, seed):
        builder.add(part.positions + np.array([x, y, 0.0]), part.normals, part.faces, colour=part.colours, feature=feature)

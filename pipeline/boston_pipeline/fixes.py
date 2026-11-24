"""Fixes kept in this repository, applied to source data before it is interpreted.

Everything that corrects the model lives under fixes/ as plain text, in any
sub-folder and under any file name:

  *.toml     change the tags of existing features, or delete them, by id
  *.geojson  add features, optionally replacing existing ones

Both use OpenStreetMap's tag vocabulary, so a fix reads the same as the data
it corrects and features.py interprets both in one place. fixes/README.md has
the format with examples.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .features import element_id

PATCH_KEYS = {"why", "at", "set", "unset", "delete"}
ID_PATTERN = r"[nwr]\d+|[a-z]+:\d+"  # OpenStreetMap ids, or <source>:<number> such as bprd:1416
RESERVED ={"why", "replaces"}  # GeoJSON properties that are not tags


class FixError(ValueError):
    """A fix file is malformed. The message names the file and what to change."""


@dataclass
class Fixes:
    patches: dict[str, dict] = field(default_factory=dict)  # feature id -> patch
    added: list[dict] = field(default_factory=list)  # ready-made source elements
    replaced: dict[str, str] = field(default_factory=dict)  # feature id -> file replacing it


@dataclass
class Applied:
    elements: list[dict]
    patched: int
    removed: int
    added: int
    unmatched: list[str]  # ids named by fixes that this data does not contain


def load(folder: Path) -> Fixes:
    """Read every fix file under `folder`. Raises FixError on the first bad one."""
    fixes = Fixes()
    if not folder.is_dir():
        return fixes
    for path in sorted(folder.rglob("*.toml")):
        _load_patches(path, path.relative_to(folder).as_posix(), fixes)
    for path in sorted(folder.rglob("*.geojson")):
        _load_features(path, path.relative_to(folder).as_posix(), fixes)
    return fixes


def _load_patches(path: Path, name: str, fixes: Fixes) -> None:
    try:
        entries = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise FixError(f"fixes/{name}: not valid TOML: {error}") from error
    for ident, entry in entries.items():
        where = f'fixes/{name}: ["{ident}"]'
        if not re.fullmatch(ID_PATTERN, ident):
            raise FixError(f"{where} is not a feature id; ids look like w123456, n123456, r123456 or bprd:1416")
        if not isinstance(entry, dict):
            raise FixError(f"{where} must be a table of why/set/unset/delete")
        if unknown := set(entry) - PATCH_KEYS:
            raise FixError(f"{where} has unknown keys {sorted(unknown)}; allowed: {sorted(PATCH_KEYS)}")
        if not str(entry.get("why", "")).strip():
            raise FixError(f"{where} needs a `why` saying what was wrong and where the right value comes from")
        if not (entry.get("set") or entry.get("unset") or entry.get("delete")):
            raise FixError(f"{where} changes nothing; give it `set`, `unset` or `delete = true`")
        if ident in fixes.patches:
            raise FixError(f"{where} is already changed in fixes/{fixes.patches[ident]['file']}; keep one")
        fixes.patches[ident] = {
            "set": {str(k): str(v) for k, v in (entry.get("set") or {}).items()},
            "unset": [str(k) for k in entry.get("unset") or []],
            "delete": bool(entry.get("delete")),
            "file": name,
        }


def _points(coordinates) -> list[dict]:
    return [{"lon": float(c[0]), "lat": float(c[1])} for c in coordinates]


def _elements(geometry: dict, ident: str, tags: dict, where: str) -> list[dict]:
    """Turn one GeoJSON geometry into source elements shaped like Overpass output."""
    kind, coordinates = geometry.get("type"), geometry.get("coordinates")
    common = {"id": 0, "tags": tags, "patched": sorted(tags)}
    if kind == "Point":
        return [{"type": "node", "ident": ident, "lon": float(coordinates[0]), "lat": float(coordinates[1]), **common}]
    if kind == "LineString":
        return [{"type": "way", "ident": ident, "geometry": _points(coordinates), **common}]
    if kind == "Polygon" and len(coordinates) == 1:
        return [{"type": "way", "ident": ident, "geometry": _points(coordinates[0]), **common}]
    if kind in ("Polygon", "MultiPolygon"):
        polygons = [coordinates] if kind == "Polygon" else coordinates
        members = [
            {"type": "way", "role": "inner" if index else "outer", "geometry": _points(ring)}
            for polygon in polygons
            for index, ring in enumerate(polygon)
        ]
        return [{"type": "relation", "ident": ident, "members": members, **common, "tags": {**tags, "type": "multipolygon"}}]
    if kind in ("MultiPoint", "MultiLineString"):
        single = {"type": kind[5:], "coordinates": None}
        return [
            element
            for index, part in enumerate(coordinates)
            for element in _elements({**single, "coordinates": part}, f"{ident}.{index}", tags, where)
        ]
    raise FixError(f"{where} has geometry type {kind!r}; use Point, LineString, Polygon or their Multi forms")


def _load_features(path: Path, name: str, fixes: Fixes) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FixError(f"fixes/{name}: not valid JSON: {error}") from error
    if data.get("type") != "FeatureCollection":
        raise FixError(f'fixes/{name}: must be a GeoJSON FeatureCollection ("type": "FeatureCollection")')
    stem = name.removesuffix(".geojson")
    for index, feature in enumerate(data.get("features", [])):
        where = f"fixes/{name}: feature {index}"
        properties = dict(feature.get("properties") or {})
        if not str(properties.get("why", "")).strip():
            raise FixError(f"{where} needs a `why` property saying what it adds and where it comes from")
        replaces = properties.get("replaces") or []
        replaces = [replaces] if isinstance(replaces, str) else list(replaces)
        tags = {str(k): str(v) for k, v in properties.items() if k not in RESERVED}
        if not tags:
            raise FixError(f"{where} needs at least one tag saying what it is, for example natural=tree")
        if not feature.get("geometry"):
            raise FixError(f"{where} has no geometry")
        for ident in replaces:
            if not re.fullmatch(ID_PATTERN, str(ident)):
                raise FixError(f"{where} replaces {ident!r}, which is not a feature id like w123456 or bprd:1416")
            fixes.replaced[str(ident)] = name
        fixes.added.extend(_elements(feature["geometry"], f"fix:{stem}#{index}", tags, where))


def apply(elements: list[dict], fixes: Fixes) -> Applied:
    """Source elements with fixes applied. The input list is left untouched."""
    present = {element_id(element) for element in elements}
    remove = {ident for ident, patch in fixes.patches.items() if patch["delete"]} | set(fixes.replaced)
    out, patched = [], 0
    for element in elements:
        ident = element_id(element)
        if ident in remove:
            continue
        patch = fixes.patches.get(ident)
        if patch:
            tags = {k: v for k, v in (element.get("tags") or {}).items() if k not in patch["unset"]}
            tags.update(patch["set"])
            element = {**element, "tags": tags, "patched": sorted({*patch["set"], *patch["unset"]})}
            patched += 1
        out.append(element)
    out.extend(fixes.added)
    return Applied(
        elements=out,
        patched=patched,
        removed=len(remove & present),
        added=len(fixes.added),
        unmatched=sorted((set(fixes.patches) | set(fixes.replaced)) - present),
    )

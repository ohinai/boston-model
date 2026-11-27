"""Where a module adds its own kinds of object without editing furniture.py, features.py or tiles.py.

Each module named in MODULES may define any of these, and is skipped if it does not exist (yet):

- `SHAPES`: `{kind: shape}`, in the form of `furniture._KINDS`. `furniture.add_object` asks here when a kind is not one of its own.
- `interpret(tags) -> (kind, variant, size) | None`: what a point with these tags is, asked by `features._object` when none of its own
  rules matched. `kind` must be in some module's `SHAPES`.
- `objects(scene) -> list[Object]`: objects made from the rest of the scene (the goals of a pitch, the light towers of a stadium),
  which `tiles.build_context` adds to the scene's own, each placed in the tile it stands in.

The modules import `furniture` and `features`, so they are loaded on first use, not at import time.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

MODULES = ("sports", "playgrounds", "stadium")


@lru_cache(maxsize=None)
def loaded() -> tuple:
    found = []
    for name in MODULES:
        full = f"{__package__}.{name}"
        try:
            found.append(importlib.import_module(full))
        except ModuleNotFoundError as error:
            if error.name != full:  # a module that exists but fails to import is a bug to see, not to hide
                raise
    return tuple(found)


def shape(kind: str):
    """The shape function for a kind that one of the modules defines, else None."""
    for module in loaded():
        found = getattr(module, "SHAPES", {}).get(kind)
        if found is not None:
            return found
    return None


def kinds() -> tuple[str, ...]:
    return tuple(kind for module in loaded() for kind in getattr(module, "SHAPES", {}))


def interpret(tags: dict) -> tuple | None:
    for module in loaded():
        rule = getattr(module, "interpret", None)
        found = rule(tags) if rule else None
        if found:
            return found
    return None


def objects(scene) -> list:
    made: list = []
    for module in loaded():
        make = getattr(module, "objects", None)
        if make:
            made.extend(make(scene))
    return made

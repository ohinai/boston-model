"""Answer "what is at this spot, and where did it come from?".

This is the first thing to run when working on a note: it lists the source
features at and around a point, with their ids and tags, and says which
ground layer the point ends up in.
"""

from __future__ import annotations

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from . import world
from .features import Scene

LAYER_OF_SURFACE = {
    "road": "roads", "path": "paths", "rail": "rail", "water": "water", "bed": "beds", "green": "green",
}
# Highest priority first; mirrors the ground partition in tiles.py.
GROUND_PRIORITY = ["building", "roads", "rail", "paths", "water", "beds", "green"]


def inspect(scene: Scene, x: float, y: float, radius: float = 10.0, sea: list[Polygon] | None = None) -> dict:
    """Everything within `radius` metres of world position (x, y), nearest first."""
    point = ShapelyPoint(x, y)
    found: list[dict] = []

    def add(layer: str, ident: str, distance: float, tags: dict, **extra) -> None:
        if distance <= radius:
            found.append({"layer": layer, "id": ident, "distance": round(max(0.0, distance), 2), "tags": tags, **extra})

    for building in scene.buildings:
        add(
            "building", building.id, building.polygon.distance(point), building.tags,
            height=round(building.height, 2), height_source=building.height_source,
            **({"part_of": building.part_of} if building.part_of else {}),
        )
    for way in scene.ways:
        add(LAYER_OF_SURFACE[way.surface], way.id, way.line.distance(point) - way.width / 2, way.tags, width=round(way.width, 2))
    for surface, areas in scene.areas.items():
        for area in areas:
            add(LAYER_OF_SURFACE[surface], area.id, area.polygon.distance(point), area.tags)
    for barrier in scene.barriers:
        # A kerb round water is placed on the land side of its shore; the line is where it stands.
        add(barrier.kind, barrier.id, barrier.line.distance(point) - barrier.width / 2, barrier.tags, **({"width": round(barrier.width, 2)} if barrier.width else {}))
    for thing in scene.objects:
        add(thing.kind, thing.id, point.distance(ShapelyPoint(thing.x, thing.y)), thing.tags)
    for layer, points in (("tree", scene.trees), ("signal", scene.signals), ("place", scene.pois)):
        for item in points:
            add(layer, item.id, point.distance(ShapelyPoint(item.x, item.y)), item.tags)
    for polygon in sea or []:
        add("water", "coastline", polygon.distance(point), {"natural": "coastline", "note": "sea, derived from coastline ways"})

    found.sort(key=lambda entry: (entry["distance"], entry["layer"], entry["id"]))
    covering = {entry["layer"] for entry in found if entry["distance"] == 0}
    lon, lat = (float(v) for v in world.to_lonlat(x, y))
    return {
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "x": round(x, 2),
        "y": round(y, 2),
        "tile": list(world.tile_of(x, y)),
        "drawn_as": next((layer for layer in GROUND_PRIORITY if layer in covering), "ground"),
        "here": [entry for entry in found if entry["distance"] == 0],
        "nearby": [entry for entry in found if entry["distance"] > 0],
    }


# Tags that say what a thing is come first; the rest follow alphabetically.
KEY_TAGS = [
    "name", "building", "building:part", "highway", "railway", "leisure", "landuse", "natural", "water",
    "waterway", "man_made", "amenity", "shop", "tourism", "historic", "garden:type", "sport", "surface",
    "height", "building:levels", "width", "lanes", "bridge", "tunnel", "layer", "leaf_type", "genus", "species",
]


def _line(entry: dict, show_distance: bool) -> str:
    ordered = [key for key in KEY_TAGS if key in entry["tags"]]
    ordered += sorted(key for key in entry["tags"] if key not in KEY_TAGS)
    tags = " ".join(f"{key}={entry['tags'][key]}" for key in ordered[:7])
    if len(ordered) > 7:
        tags += f" (+{len(ordered) - 7} more)"
    extra = ""
    if "height" in entry:
        extra = f"  [{entry['height']} m tall, from {entry['height_source']}"
        extra += f", part of {entry['part_of']}]" if "part_of" in entry else "]"
    elif "width" in entry:
        extra = f"  [{entry['width']} m wide]"
    distance = f"{entry['distance']:6.1f} m  " if show_distance else "  "
    return f"{distance}{entry['layer']:<9} {entry['id']:<14} {tags}{extra}"


def format_report(report: dict, radius: float) -> str:
    lines = [
        f"{report['lat']:.6f}, {report['lon']:.6f}   world x {report['x']} m, y {report['y']} m   "
        f"tile {report['tile'][0]}_{report['tile'][1]}",
        f"Drawn here as: {report['drawn_as']}",
        "",
        "Covering this point" if report["here"] else "Nothing mapped covers this point (plain paved ground).",
        *(_line(entry, False) for entry in report["here"]),
        "",
        f"Within {radius:g} m" if report["nearby"] else f"Nothing else within {radius:g} m.",
        *(_line(entry, True) for entry in report["nearby"][:40]),
    ]
    if len(report["nearby"]) > 40:
        lines.append(f"  ... and {len(report['nearby']) - 40} more; use a smaller --radius or --json")
    return "\n".join(lines)

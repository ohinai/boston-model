"""Command line entry point: `boston-pipeline fetch|build|inspect|info`."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tomllib
from pathlib import Path

from shapely.geometry import box

from . import brookline, cityheights, fixes, inventory, lookup, osm, world
from .features import Scene, classify
from .tiles import build, sea_polygons

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "web" / "public" / "tiles"
FIXES = ROOT / "fixes"


def _areas() -> dict:
    return tomllib.loads((ROOT / "areas.toml").read_text(encoding="utf-8"))


def _area(name: str) -> dict:
    areas = _areas()
    if name not in areas:
        raise SystemExit(f"unknown area '{name}'; areas.toml defines: {', '.join(areas)}")
    return areas[name]


def _raw_path(name: str) -> Path:
    return ROOT / "data" / "raw" / name / "osm.json"


def _trees_path(name: str) -> Path:
    return ROOT / "data" / "raw" / name / "trees.json"


EXTRA_SOURCES = ("brookline", "city_heights")  # what an area's `extra_sources` in areas.toml may list


def _extra_sources(area: dict) -> list[str]:
    """The optional data sources an area asks for in areas.toml (`extra_sources = ["brookline"]`); none by default."""
    names = list(area.get("extra_sources", []))
    if unknown := [n for n in names if n not in EXTRA_SOURCES]:
        raise SystemExit(f"areas.toml: unknown extra_sources {unknown}; known: {', '.join(EXTRA_SOURCES)}")
    return names


def _scene(name: str) -> tuple[Scene, dict, fixes.Applied]:
    """Source data for an area with this repository's fixes applied, interpreted."""
    raw = _raw_path(name)
    if not raw.exists():
        raise SystemExit(f"{raw} not found; run `boston-pipeline fetch {name}` first")
    payload = osm.load(raw)
    elements = list(payload["elements"])
    trees = _trees_path(name)
    if trees.exists():
        elements += inventory.elements(json.loads(trees.read_text(encoding="utf-8")))
    else:
        print(f"{trees} not found; using OpenStreetMap trees only (run `boston-pipeline fetch {name}` for the city inventory)")
    # The Town of Brookline's layers, for an area that asks for them.
    town = False
    if "brookline" in _extra_sources(_area(name)):
        trees_file, buildings_file = (_raw_path(name).parent / f for f in (brookline.TREES_FILE, brookline.BUILDINGS_FILE))
        if trees_file.exists() and buildings_file.exists():
            elements += brookline.tree_elements(json.loads(trees_file.read_text(encoding="utf-8")))
            town_footprints = brookline.footprints(json.loads(buildings_file.read_text(encoding="utf-8")))
            town = True
        else:
            print(f"{trees_file.parent} has no Brookline files; run `boston-pipeline fetch {name}` to download the Town of Brookline's layers")
    # The City of Boston's building survey, for an area that asks for it.
    city = None
    if "city_heights" in _extra_sources(_area(name)):
        city_file = _raw_path(name).parent / cityheights.FILE
        if city_file.exists():
            city = cityheights.footprints(json.loads(city_file.read_text(encoding="utf-8")))
        else:
            print(f"{city_file} not found; run `boston-pipeline fetch {name}` to download the City of Boston's building heights")
    try:
        applied = fixes.apply(elements, fixes.load(FIXES))
    except fixes.FixError as error:
        raise SystemExit(f"error in a fix file:\n  {error}") from error
    elements = applied.elements
    if town:
        # After the fixes, so that a fix's `height` or `building:levels` is never overruled.
        elements, payload["brookline_heights"] = brookline.apply_heights(elements, town_footprints[0])
        payload["brookline_heights"]["set_aside"] = town_footprints[1]
    if city is not None:
        # After the fixes, so that a fix's `height` or `building:levels` is never overruled; before the Town's, which covers other buildings.
        elements, payload["city_heights"] = cityheights.apply_heights(elements, city[0])
        payload["city_heights"]["set_aside"] = city[1]
    scene = classify(elements)
    # The tree counts travel with the source payload, for the build summary.
    payload["tree_counts"] = inventory.reconcile(scene, other=(brookline.TREE_PREFIX,) if town else ())
    if town:
        payload["brookline_tree_counts"] = inventory.reconcile(scene, brookline.TREE_PREFIX, other=("bprd:",), drop_in_parks=False)
    payload["trees_in_buildings"] = inventory.drop_trees_in_buildings(scene)  # after every inventory is in
    return scene, payload, applied


def cmd_fetch(args) -> None:
    tiles = world.tiles_covering(tuple(_area(args.area)["bbox"]))
    # A margin so features straddling the outermost tiles arrive whole.
    bounds = world.lonlat_bounds(tiles, margin=60.0)
    osm.fetch_area(bounds, _raw_path(args.area), force=args.force)
    inventory.fetch(bounds, _trees_path(args.area), force=args.force)
    if "brookline" in _extra_sources(_area(args.area)):
        brookline.fetch(bounds, _raw_path(args.area).parent, force=args.force)
    if "city_heights" in _extra_sources(_area(args.area)):
        cityheights.fetch(bounds, _raw_path(args.area).parent, force=args.force)


def cmd_build(args) -> None:
    area = _area(args.area)
    scene, payload, applied = _scene(args.area)
    tiles = world.tiles_covering(tuple(area["bbox"]))
    out_dir = Path(args.out).resolve()
    if (out_dir / "tiles").exists():
        shutil.rmtree(out_dir / "tiles")
    print(f"building {len(tiles)} tiles into {out_dir} ...")
    stats = build(scene, tiles, out_dir, area, payload.get("osm3s", {}).get("timestamp_osm_base"))
    sources = stats["height_sources"]
    total = max(1, stats["buildings"])
    print(
        f"done: {stats['tiles']} tiles, {stats['triangles']:,} triangles, {stats['megabytes']} MB\n"
        f"  {stats['buildings']:,} buildings, {stats['trees']:,} trees, {stats['objects']:,} benches, lamps and monuments,"
        f" {stats['bridges']} raised footbridges, {stats.get('barriers', 0):,} fences and walls,"
        f" {stats['streets']:,} street and path segments\n"
        "  building heights: "
        + ", ".join(
            f"{sources.get(k, 0) * 100 // total}% from {k}" for k in ("height", "levels", "default", "override", "town", "city") if k not in ("town", "city") or sources.get(k)
        )
        + f"\n  fixes applied: {applied.patched} changed, {applied.removed} removed, {applied.added} added"
    )
    if survey := payload.get("city_heights"):
        print(
            f"  City of Boston survey: {survey['given']:,} building heights from the roof-break parts ({survey['over_levels']:,} replace a count of levels;"
            f" {survey['kept_levels']:,} keep their levels because the survey disagrees by more than a third; {survey['parts']:,} usable parts, set aside: {survey['set_aside']})"
        )
    if heights := payload.get("brookline_heights"):
        print(
            f"  Brookline: {heights['given']:,} building heights from the Town's footprints"
            f" ({heights['from_outbuilding']:,} outbuildings; {heights['footprints']:,} usable footprints, set aside: {heights['set_aside']})"
        )
    counts = payload["tree_counts"]
    if counts["inventory"]:
        print(
            f"  trees: {counts['inventory']:,} from the city inventory, {counts['kept_osm']:,} from OpenStreetMap only"
            f" ({counts['merged']:,} duplicates merged, {counts['dropped']:,} dropped)"
            + (f", {counts['added']:,} added by fixes" if counts.get("added") else "")
        )
    if town := payload.get("brookline_tree_counts"):
        print(f"  Brookline trees: {town['inventory']:,} from the Town's inventory, {town['merged']:,} merged with OpenStreetMap trees (the rest of OpenStreetMap's are kept)")
    if payload.get("trees_in_buildings"):
        print(f"  {payload['trees_in_buildings']:,} trees dropped because they stand inside a building's outline")
    if applied.unmatched:
        print(
            f"  {len(applied.unmatched)} fixes name features this area does not contain "
            f"(outside the area, or the id has changed): {', '.join(applied.unmatched[:12])}"
        )


def _parse_position(words: list[str]) -> tuple[float, float]:
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", " ".join(words))]
    if len(numbers) != 2:
        raise SystemExit("give a position as latitude and longitude, for example: inspect 42.35606,-71.06262")
    lat, lon = numbers
    if lat < 0 < lon:  # given as lon, lat
        lat, lon = lon, lat
    return lat, lon


def cmd_inspect(args) -> None:
    lat, lon = _parse_position(args.position)
    names = [args.area] if args.area else [
        name for name, area in _areas().items() if area["bbox"][0] <= lon <= area["bbox"][2] and area["bbox"][1] <= lat <= area["bbox"][3]
    ]
    if not names:
        raise SystemExit(f"{lat}, {lon} is outside every area in areas.toml")
    scene, _, _ = _scene(names[0])
    tiles = world.tiles_covering(tuple(_area(names[0])["bbox"]))
    extent = box(
        min(t[0] for t in tiles) * world.TILE_SIZE,
        min(t[1] for t in tiles) * world.TILE_SIZE,
        (max(t[0] for t in tiles) + 1) * world.TILE_SIZE,
        (max(t[1] for t in tiles) + 1) * world.TILE_SIZE,
    )
    x, y = (float(v) for v in world.to_local(lon, lat))
    report = lookup.inspect(scene, x, y, args.radius, sea_polygons(scene.coastlines, extent))
    print(json.dumps(report, indent=1) if args.json else lookup.format_report(report, args.radius))


def cmd_audit(args) -> None:
    from . import audit
    from .tiles import prepare

    if bool(args.park) == bool(args.around):
        raise SystemExit("give exactly one of --park NAME or --around LAT,LON")
    area = _area(args.area)
    scene, _, _ = _scene(args.area)
    if args.park:
        geometry = audit.park_geometry(scene, args.park)
    else:
        lat, lon = _parse_position([args.around])
        geometry = audit.square_around(lat, lon, args.size)
    # Which footbridges are raised is decided while preparing tiles, so ask the same code.
    ctx = prepare(scene, world.tiles_covering(tuple(area["bbox"])))
    decks = [pair for pairs in ctx.decks.values() for pair in pairs]
    audit.run(
        args.area, scene, geometry, decks, ROOT / "data" / "raw" / args.area, Path(args.out), args.cell, args.pixels, args.force,
        canopy=not args.no_canopy,
    )


def cmd_ground(args) -> None:
    from . import audit, ground

    _area(args.area)
    scene, _, _ = _scene(args.area)
    geometry = audit.park_geometry(scene, args.park)
    ground.run(
        args.area, args.park, geometry, ROOT / "data" / "raw" / args.area, Path(args.out), ROOT / "web" / "dist",
        limit=args.limit, force=args.force, do_capture=not args.no_capture,
    )


def cmd_info(_args) -> None:
    lon, lat = (float(v) for v in world.to_lonlat(0.0, 0.0))
    print(f"world CRS      {world.WORLD_CRS}")
    print(f"origin         E {world.ORIGIN_E:.0f}  N {world.ORIGIN_N:.0f}  (lat {lat:.6f}, lon {lon:.6f})")
    print(f"tile size      {world.TILE_SIZE:.0f} m")
    for name, area in _areas().items():
        tiles = world.tiles_covering(tuple(area["bbox"]))
        print(f"area {name:<10}{len(tiles)} tiles, x {tiles[0][0]}..{tiles[-1][0]}, y {tiles[0][1]}..{tiles[-1][1]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="boston-pipeline", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download source data for an area")
    fetch.add_argument("area")
    fetch.add_argument("--force", action="store_true", help="re-download even if cached")
    fetch.set_defaults(run=cmd_fetch)

    build_cmd = commands.add_parser("build", help="build 3D tiles for an area")
    build_cmd.add_argument("area")
    build_cmd.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: web/public/tiles)")
    build_cmd.set_defaults(run=cmd_build)

    inspect = commands.add_parser(
        "inspect", help="list the source features at a position, with ids and tags (start here when fixing a note)"
    )
    inspect.add_argument("position", nargs="+", help="latitude and longitude, e.g. 42.35606,-71.06262")
    inspect.add_argument("--radius", type=float, default=10.0, help="how far around the point to look, in metres")
    inspect.add_argument("--area", help="area name (default: the area containing the point)")
    inspect.add_argument("--json", action="store_true", help="print machine-readable JSON")
    inspect.set_defaults(run=cmd_inspect)

    audit = commands.add_parser(
        "audit", help="draw the model over an aerial photograph, cell by cell, to spot where it is wrong"
    )
    audit.add_argument("area")
    audit.add_argument("--park", help="audit every cell overlapping the park with this name")
    audit.add_argument("--around", help="audit a square centred on LAT,LON")
    audit.add_argument("--size", type=float, default=240.0, help="side of the --around square, in metres")
    audit.add_argument("--cell", type=float, default=120.0, help="side of a cell, in metres")
    audit.add_argument("--pixels", type=int, default=1000, help="width and height of each picture")
    audit.add_argument("--out", default=ROOT / "data" / "audit", help="output directory (default: data/audit)")
    audit.add_argument("--force", action="store_true", help="download the photographs again")
    audit.add_argument("--no-canopy", action="store_true", help="skip the canopy picture (proposed trees, measured crowns)")
    audit.set_defaults(run=cmd_audit)

    ground = commands.add_parser(
        "ground", help="lay ground-level photographs of a park beside the model's views from the same spots"
    )
    ground.add_argument("area")
    ground.add_argument("--park", required=True, help="name of the park, as in the data")
    ground.add_argument("--limit", type=int, default=40, help="most photographs to use (default: 40)")
    ground.add_argument("--out", default=ROOT / "data" / "ground", help="output directory (default: data/ground)")
    ground.add_argument("--force", action="store_true", help="query and download again, and capture every view again")
    ground.add_argument("--no-capture", action="store_true", help="write the photographs and sheets without the model's pictures")
    ground.set_defaults(run=cmd_ground)

    commands.add_parser("info", help="print the world frame and areas").set_defaults(run=cmd_info)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

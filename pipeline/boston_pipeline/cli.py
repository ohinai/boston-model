"""Command line entry point: `boston-pipeline fetch|build|inspect|info`."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tomllib
from pathlib import Path

from shapely.geometry import box

from . import fixes, lookup, osm, world
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


def _scene(name: str) -> tuple[Scene, dict, fixes.Applied]:
    """Source data for an area with this repository's fixes applied, interpreted."""
    raw = _raw_path(name)
    if not raw.exists():
        raise SystemExit(f"{raw} not found; run `boston-pipeline fetch {name}` first")
    payload = osm.load(raw)
    try:
        applied = fixes.apply(payload["elements"], fixes.load(FIXES))
    except fixes.FixError as error:
        raise SystemExit(f"error in a fix file:\n  {error}") from error
    return classify(applied.elements), payload, applied


def cmd_fetch(args) -> None:
    tiles = world.tiles_covering(tuple(_area(args.area)["bbox"]))
    # A margin so features straddling the outermost tiles arrive whole.
    osm.fetch(world.lonlat_bounds(tiles, margin=60.0), _raw_path(args.area), force=args.force)


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
        f"  {stats['buildings']:,} buildings, {stats['trees']:,} trees, {stats['streets']:,} street and path segments\n"
        "  building heights: "
        + ", ".join(f"{sources.get(k, 0) * 100 // total}% from {k}" for k in ("height", "levels", "default", "override"))
        + f"\n  fixes applied: {applied.patched} changed, {applied.removed} removed, {applied.added} added"
    )
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

    commands.add_parser("info", help="print the world frame and areas").set_defaults(run=cmd_info)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

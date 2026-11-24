"""Command line entry point: `boston-pipeline fetch|build|info`."""

from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path

from . import osm, world
from .features import classify
from .tiles import build

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "web" / "public" / "tiles"


def _area(name: str) -> dict:
    areas = tomllib.loads((ROOT / "areas.toml").read_text(encoding="utf-8"))
    if name not in areas:
        raise SystemExit(f"unknown area '{name}'; areas.toml defines: {', '.join(areas)}")
    return areas[name]


def _raw_path(name: str) -> Path:
    return ROOT / "data" / "raw" / name / "osm.json"


def _overrides() -> dict:
    path = ROOT / "overrides" / "buildings.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cmd_fetch(args) -> None:
    tiles = world.tiles_covering(tuple(_area(args.area)["bbox"]))
    # A margin so features straddling the outermost tiles arrive whole.
    osm.fetch(world.lonlat_bounds(tiles, margin=60.0), _raw_path(args.area), force=args.force)


def cmd_build(args) -> None:
    area = _area(args.area)
    raw = _raw_path(args.area)
    if not raw.exists():
        raise SystemExit(f"{raw} not found; run `boston-pipeline fetch {args.area}` first")
    payload = osm.load(raw)
    tiles = world.tiles_covering(tuple(area["bbox"]))
    print(f"classifying {len(payload['elements'])} OSM elements ...")
    scene = classify(payload["elements"], _overrides())
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
    )


def cmd_info(_args) -> None:
    lon, lat = (float(v) for v in world.to_lonlat(0.0, 0.0))
    print(f"world CRS      {world.WORLD_CRS}")
    print(f"origin         E {world.ORIGIN_E:.0f}  N {world.ORIGIN_N:.0f}  (lat {lat:.6f}, lon {lon:.6f})")
    print(f"tile size      {world.TILE_SIZE:.0f} m")
    for name in tomllib.loads((ROOT / "areas.toml").read_text(encoding="utf-8")):
        tiles = world.tiles_covering(tuple(_area(name)["bbox"]))
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
    commands.add_parser("info", help="print the world frame and areas").set_defaults(run=cmd_info)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

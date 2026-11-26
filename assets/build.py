"""Build the assets from their Blender scripts.

    python assets/build.py                 # every assets/src/*.py
    python assets/build.py park-lamp       # one asset: assets/src/park_lamp.py -> assets/park-lamp.glb

Needs Blender (3.6 or later; tested with 5.2). It is found through the BLENDER environment variable, then the PATH,
then the usual install folders. Only people changing an asset need it: the built .glb files are committed.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_blender() -> str:
    chosen = os.environ.get("BLENDER")
    if chosen:
        if not Path(chosen).exists():
            raise SystemExit(f"BLENDER is set to {chosen!r}, which does not exist")
        return chosen
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    patterns = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Blender Foundation", "Blender *", "blender.exe"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    raise SystemExit("Blender not found: install it or set BLENDER to its path")


def main(names: list[str]) -> None:
    scripts = sorted(p for p in (HERE / "src").glob("*.py") if not p.name.startswith("_"))
    if names:
        wanted = {n.replace("-", "_") for n in names}
        scripts = [p for p in scripts if p.stem in wanted]
        if len(scripts) != len(wanted):
            raise SystemExit(f"no script in assets/src for: {', '.join(sorted(wanted - {p.stem for p in scripts}))}")
    blender = find_blender()
    for script in scripts:
        out = HERE / f"{script.stem.replace('_', '-')}.glb"
        print(f"{script.name} -> {out.name}")
        result = subprocess.run(
            [blender, "--background", "--factory-startup", "--python", str(script), "--", str(out)],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("asset:"):
                print("  " + line)
        if result.returncode != 0 or not out.exists():
            print(result.stdout[-1500:], result.stderr[-1500:], sep="\n")
            raise SystemExit(f"{script.name} failed")


if __name__ == "__main__":
    main(sys.argv[1:])

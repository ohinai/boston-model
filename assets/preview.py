"""A contact sheet of an asset: front, right, back and three-quarter views, with its vertex colours.

    python assets/preview.py park-lamp                 # assets/park-lamp.glb -> park-lamp-preview.png beside the model
    python assets/preview.py park-lamp --out sheet.png # or somewhere else

Needs Blender (see build.py) and Pillow (the pipeline's dev extra). Look at the result to judge a model without
opening Blender: the view called "front" is the side the asset faces.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from build import HERE, find_blender


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    name = argv[0]
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else HERE / f"{name}-preview.png"
    model = Path(name) if name.endswith(".glb") else HERE / f"{name}.glb"
    if not model.exists():
        raise SystemExit(f"{model} does not exist: build it first (python assets/build.py {name})")
    from PIL import Image  # imported here so build.py works without Pillow

    with tempfile.TemporaryDirectory() as scratch:
        prefix = str(Path(scratch) / "view")
        result = subprocess.run(
            [find_blender(), "--background", "--factory-startup", "--python", str(HERE / "src" / "_preview.py"), "--", str(model), prefix],
            capture_output=True, text=True,
        )
        views = [Path(f"{prefix}_{v}.png") for v in ("front", "right", "back", "three_quarter")]
        if result.returncode != 0 or not all(v.exists() for v in views):
            print(result.stdout[-1500:], result.stderr[-1500:], sep="\n")
            raise SystemExit("the preview failed")
        images = [Image.open(v).convert("RGB") for v in views]
        sheet = Image.new("RGB", (sum(i.width for i in images), images[0].height))
        x = 0
        for image in images:
            sheet.paste(image, (x, 0))
            x += image.width
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])

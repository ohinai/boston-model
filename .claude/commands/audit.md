---
description: Check part of the model against aerial photographs and report what is wrong
argument-hint: <park name in quotes, or lat,lon> [cell names to check]
---

Audit the Boston Model against aerial photographs. The argument is: $ARGUMENTS

This is a checking job: look carefully, measure, and report only what the
photograph clearly shows. A wrong "fix" is worse than a missed one.

## 1. Make the audit sheets

```sh
boston-pipeline audit pilot --park "Boston Public Garden"      # or: --around LAT,LON --size 240
```

This writes one folder per 120 m cell under `data/audit/pilot/`, each with:

- `aerial.jpg`: the photograph (USGS NAIP, 7 July 2023, public domain, north up).
- `overlay.jpg`: the same photograph with the model drawn on it. Tree crowns
  are circles with a number (yellow: city inventory, orange: other), path edges
  cyan, planted beds magenta, water blue, raised footbridges white, buildings
  red, benches red squares, lamps white dots, monuments magenta diamonds.
  White circles are what a canopy detector found in the photograph's infrared
  band: a white circle labelled `C1`, `C2`... is a crown with no tree in the
  data (a proposed tree); a thin white circle around a numbered tree is that
  tree's crown as measured.
- `canopy.jpg`: the photograph with the detector's canopy outline in magenta,
  to see why it proposed what it did.
- `features.json`: what each number and outline is (ids, species, sizes), the
  cell's bounds in world metres and `metres_per_pixel`.

If cell names were given, check only those; otherwise check every cell written.

## 2. Check each cell

Read `aerial.jpg`, `overlay.jpg` and `features.json`. Zoom in before judging
anything small: crop a region with Pillow, enlarge it three or four times, save
it to a scratch folder and read that. Measure in pixels and multiply by
`metres_per_pixel`.

Things to know about the photograph:

- It was taken in summer, in full leaf. Before judging anything, work out
  which way shadows fall in the picture in front of you, from a lamp post or a
  tree standing alone: do not assume a direction. A dark patch on that side of
  a tree is its shadow, not a tree.
- It is from July 2023. A tree planted since will not be in it, and a young
  tree with a thin trunk may be too small to see. Only call a tree absent if
  its `trunk_cm` is 25 or more, or it has no trunk size and a modelled crown of
  6 m or more.
- Where crowns touch, you cannot tell trees apart from above. Say so; do not
  guess.
- Tree heights cannot be read from this photograph. Do not report them.

Look for, in this order:

1. **Alignment.** Do path outlines sit on the paths and the water outline on
   the water's edge? If everything is shifted the same way, report that once
   and nothing else for the cell.
2. **Proposed trees and measured crowns.** Give every `C` label a verdict:
   `tree` (a real crown, standing where the circle is), `not_a_tree` (a shadow,
   a patch of lawn, a shrub bed, an umbrella, part of a larger crown) or
   `unsure`. Give every thin white measured circle a verdict: `ok` (it hugs
   that tree's crown, within about a fifth), `wrong` (it does not) or `unsure`.
   The detector reads deep shade as canopy and very light foliage as lawn, so
   check each one against the photograph, enlarged.
3. **Other trees.** A circle with no crown under it (absent, or a few metres off: say
   which and where the crown is). A clear crown standing on its own with no
   circle (unmapped). A crown standing on its own whose width differs from its
   circle by more than a quarter.
4. **Paths.** A path whose real width differs from its outline by more than a
   quarter. A clear path with no outline. An outline where there is no path.
5. **Planted beds.** A bed with no outline, an outline with no bed, or a
   clearly wrong shape.
6. **Anything else** a walker would notice: a missing pond, a building or
   dock in the wrong place, a footbridge of the wrong width.

## 3. Write the findings

Write `findings.json` into the cell's folder:

```json
{
  "cell": "c_-5_-4",
  "alignment": "good",
  "summary": "One sentence on how well this cell matches.",
  "proposed": {"C1": "tree", "C2": "not_a_tree", "C3": "unsure"},
  "crowns": {"7": "ok", "12": "wrong"},
  "findings": [
    {"kind": "tree_absent", "tree": 12, "id": "bprd:123", "confidence": "high",
     "evidence": "Open lawn at the trunk; the nearest crown edge is 9 m away."},
    {"kind": "tree_moved", "tree": 3, "id": "n456", "px": [612, 340], "confidence": "medium",
     "evidence": "A crown of the same size sits 4 m to the east of the circle."},
    {"kind": "tree_unmapped", "px": [412, 655], "seen_m": 6.0, "confidence": "high",
     "evidence": "Round crown 50 px across on open lawn, with its shadow beside it."},
    {"kind": "crown_size", "tree": 7, "id": "bprd:789", "modelled_m": 5.2, "seen_m": 11.0,
     "confidence": "high", "evidence": "Crown stands alone and is 92 px across."},
    {"kind": "path_width", "id": "w123", "modelled_m": 3.2, "seen_m": 4.6, "confidence": "medium",
     "evidence": "Measured 38 px between the lawn edges at three places."},
    {"kind": "other", "px": [500, 500], "confidence": "high", "evidence": "What it is and why it matters."}
  ]
}
```

- `proposed` has a verdict for every `C` label in `features.json`, and
  `crowns` one for every tree that has a `seen_crown` there, keyed by the
  tree's number.
- `kind` is one of `tree_absent`, `tree_moved`, `tree_unmapped`, `crown_size`,
  `path_width`, `path_unmapped`, `path_absent`, `bed_unmapped`, `bed_absent`,
  `bed_shape`, `other`.
- `px` is `[column, row]` in the cell's picture. `tree` is the number on the
  overlay and `id` its id from `features.json`.
- `confidence` is `high` only when someone else looking at the same picture
  would certainly agree. Use `medium` or `low` freely, and leave out anything
  you could not check.
- A cell with nothing wrong gets an empty `findings` list.

## 4. Report

Summarise per cell: how many findings of each kind and confidence, and the two
or three that matter most. Do not write fix files, commit or comment on tickets
unless asked. Turning findings into fixes is a separate step, done after a
second look: see "Fixing a note" in AGENTS.md, and cite the photograph (USGS
NAIP aerial imagery, 7 July 2023) in each fix's `why`.

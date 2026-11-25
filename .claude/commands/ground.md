---
description: Compare ground-level photographs of a park with the model's view from the same spots
argument-hint: <park name in quotes> [photo ids to check]
---

Check how the Boston Model looks at eye level against real photographs. The
argument is: $ARGUMENTS

This is a looking job. The aim is a short, honest list of what a walker would
notice is wrong, not a list of every difference.

## 1. Make the sheets

```sh
boston-pipeline build pilot                 # the tiles must be current
cd web && npm run build && cd ..            # the capture uses the built viewer
boston-pipeline ground pilot --park "Boston Public Garden"
```

This writes one folder per photograph under `data/ground/pilot/`, each with:

- `photo.jpg`: an openly licensed photograph from Wikimedia Commons, taken
  inside the park.
- `model_<heading>.jpg`: the model seen from the photograph's position,
  standing on the ground, looking in that compass direction (degrees clockwise
  from north).
- `sheet.jpg`: the photograph above the model's views, each labelled with its
  heading.
- `info.json`: where the photograph was taken, when, by whom, its licence, and
  whether the camera recorded which way it faced.

If photo ids were given, check only those; otherwise check every sheet.

## 2. Check each sheet

Read `sheet.jpg` and `info.json`. Open `photo.jpg` and single model pictures at
full size when detail matters.

1. **Is the photograph usable?** It is if it shows the park at walking scale
   with enough surroundings to recognise the place: paths, lawns, trees, the
   lagoon, a bridge, a statue in its setting, buildings behind. It is not if it
   is a close-up (a flower, a plaque, a duck, a face), an interior, an aerial
   view, a night shot, or mostly people. Say which and move on.
2. **Which way was the camera facing?** Find the model picture that shows the
   same scene, using things that do not move: buildings on the skyline, the
   water, a bridge, a path's bend, a statue. If the position recorded for the
   photograph is clearly wrong (nothing matches in any direction), say so and
   move on. Positions are often tens of metres out.
3. **What would a walker notice is different?** Most important first, at most
   eight per photograph. Look for:
   - things in the photograph that the model lacks: fences and railings,
     gates, walls, hedges, statues, signs, flagpoles, kiosks, boats, steps;
   - things in the model that are not there;
   - the wrong look: the style of a bench, lamp post or bridge; the colour or
     material of a path; a statue that is the wrong shape; water that is wrong;
   - the wrong size or density: trees much taller, shorter, wider or sparser
     than they are; a path too narrow; buildings behind too low or too tall.

Not differences: the season (bare trees, snow, autumn colour, flowers in
bloom), the weather and light, people, vehicles, events and temporary things.

## 3. Write the findings

Write `findings.json` into the photograph's folder:

```json
{
  "photo": 12345678,
  "usable": true,
  "matching_view": 120,
  "summary": "One sentence on how well the model matches this view.",
  "findings": [
    {"kind": "missing", "what": "Black iron fence, about waist height, along both sides of the path",
     "where": "foreground, left and right", "confidence": "high",
     "evidence": "Clear in the photograph; the model shows open lawn to the path edge."},
    {"kind": "wrong_look", "what": "Bridge railings are ornate ironwork between stone piers, the model has plain rails",
     "where": "centre", "confidence": "high", "evidence": "..."},
    {"kind": "wrong_size", "what": "The willows by the water are about twice as tall as the model draws them",
     "where": "right bank", "confidence": "medium", "evidence": "Judged against the lamp posts beside them."}
  ]
}
```

- `usable` false needs only `photo`, `usable` and a `reason`.
- `matching_view` is the heading of the model picture that matches, or `null`.
- `kind` is one of `missing`, `extra`, `wrong_look`, `wrong_size`,
  `wrong_place`, `other`.
- `confidence` is `high` only when anyone comparing the two pictures would
  agree.

## 4. Report

Summarise across the sheets: how many photographs were usable, and the
differences that came up most often, with the photo ids that show them best.

Do not write fix files, commit or comment on tickets unless asked. Photographs
are visual reference only (see "Where correct information may come from" in
AGENTS.md): they show what to look into, and the fix itself needs a source the
project may use as data. `info.json` records each photograph's licence and
author. Most findings here point at something the pipeline does not draw yet
rather than at a single wrong fact: say so.

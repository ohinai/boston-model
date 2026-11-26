# Capturing a patch for a splat pilot

**Status: tabled (25 November 2026).** The generated model stays the way forward: splats cannot be corrected through
`fixes/`, a capture covers one tree on one day, and each patch needs tape-measured registration. Nothing here is
needed by the rest of the project. The kit below is ready if the question is picked up again; the viewer spike is on
branch `worktree-agent-a67b727cd71e2ced9`, unmerged.

A pilot to find out whether a Gaussian splat made from the project owner's own photographs can sit inside the viewer
as an **appearance layer** over the generated model. The splat is only how a patch looks. Where things are, how big
they are, collision and notes stay with the generated model and the data; see the notes on what the splat spike
(`tools/splats/`, on its own branch) found about registration, lighting and performance.

Source rules: these are the owner's own photographs, which `AGENTS.md` allows as a source. Keep people out of the
frame (wait, or come early), and keep the photographs out of git: they live under `data/capture/`, which is ignored.

## Which patch, and when

| Subject | Best time | Why |
|---|---|---|
| Norway spruce `bprd:56169`, 2 lamps and 3 benches within about 20 m | **Now (winter)**: it is evergreen | Dense needles reconstruct far better than bare twigs, and the model also draws it in full leaf |
| Dutch elm `bprd:900093528`, a row of four benches beside it | Leaf-on, May to September | A bare elm is mostly thin twigs, which splats render as noise, and it would not match the model's summer crown |

The sheets are in `data/capture/<tree id>-capture-sheet.png`: the aerial photograph with the tree, benches and
lamps marked, the two orbits to walk, and a list of distances to measure. **Overcast and calm** is best: no hard shadows
baked in, and leaves and needles that do not move. A weekday morning has the fewest people.

## Phone settings

- Use the **main 1x camera only**. No zoom, no ultra-wide, no portrait or night mode, no Live Photo, no filters.
- Highest-quality photo format (HEIC or JPEG are both fine). Location (GPS) tagging **on**.
- On each pass lock exposure and focus (tap and hold on the tree) so brightness does not drift between photographs.
- Clean the lens. Do not shoot into the sun or in rain. If you prefer video: 4K, 30 fps, walk slowly (frames are
  extracted later), but photographs are sharper.

## What to shoot (300 to 450 photographs in all)

Each photograph should share most of its view with the one before it (about 70% overlap): about a metre between steps.

1. **Outer orbit** (the magenta ring): walk all the way round, camera at chest height, facing the tree. 60 to 80
   photographs. Then a second lap with the camera tilted up about 30 degrees to take in the crown.
2. **Inner orbit** (the yellow ring): the same, closer, 40 to 50 photographs, including the trunk, the low branches and
   a few looking straight up.
3. **The walker's view**: the viewer is for walking, and a splat is only good near where the camera was, so walk the
   nearest path **in both directions** at eye height. Look ahead, then to each side. 60 to 100 photographs.
4. **The measuring points**: a few photographs of each end of every distance on the sheet, with the tape laid on the
   ground between them where you can.
5. If a ring is blocked by planting, a bed or water, skip that part and walk the clear sides; do not wade in.

Do not delete blurry or odd photographs on site: more is better than fewer, and I can drop bad ones.

## What to measure

Fill in the sheet: the tape distances between the labelled points (ground level, base to base), the height of lamp L1
to the top of its globe, and the trunk girth at chest height. The tape measurements give the splat its real size (a
reconstruction from photographs has no scale of its own). The model's own positions are only about a metre accurate,
so they place the splat roughly, and edges that show in both (a path, a kerb, a bed) refine it. Photograph the
filled-in sheet.

## Handing it over

Copy the **original files** (not screenshots or messages, which strip the metadata) into
`data/capture/<tree id>/photos/`, for example `data/capture/bprd-56169/photos/`, with the photograph of the sheet and a
line saying the date, time and weather. Nothing else is needed from you.

## Tools on this machine

Kept outside the repository, in `C:\Users\ohina\splat-tools\` (each in its own folder, checksums checked against the
publishers' on 25 November 2026):

| Tool | Version | Used for | Licence |
|---|---|---|---|
| COLMAP (`colmap-4.2.1\bin\colmap.exe`, CUDA build) | 4.2.1 | Camera positions from the photographs (structure from motion) | BSD (ETH Zurich and UNC Chapel Hill) |
| Brush (`brush-0.3.0\brush_app.exe`) | 0.3.0 | Training the splat from COLMAP's output | Apache 2.0 |

Smoke-tested: COLMAP extracts SIFT features from images, and both programs print their help. Training itself has not
been run yet, because it needs the photographs.

## What happens next

Reconstruct the camera positions from the photographs (structure from motion), train the splat on this machine's
RTX 4070, scale and place it using the tape measurements and the model's paths, clean it, and load it in the viewer with
the splat spike's registry. Then compare it with the generated patch from the same spots. The questions the pilot
answers: does it register to within a few centimetres, does it hold up when you walk off the line you photographed, and
does the baked-in light look right next to the rest of the model.

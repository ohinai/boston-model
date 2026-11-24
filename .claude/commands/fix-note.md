---
description: Fix a note left in the viewer (a GitHub issue labelled "note")
argument-hint: <issue number, or blank to list open notes>
---

Work on a note from the Boston Model viewer. The argument is: $ARGUMENTS

If no issue number was given, run `gh issue list --label note --state open` and
show the open notes with their numbers, place and first line. Recommend which to
take first (parks and gardens are the current priority), then stop and ask.

Otherwise, follow "Fixing a note" in AGENTS.md for that issue:

1. `gh issue view <number>` and read the note and its JSON position block.
2. Run `boston-pipeline inspect` at the `looking_at` position (or `standing` if
   there is none) to see the source features there. Build the area first if
   `data/raw/pilot/osm.json` or the tiles are missing.
3. Say what kind of problem this is (wrong data for one place, something drawn
   wrongly everywhere, something not drawn at all, or a viewer issue) and what
   you intend to change. If the note does not give enough to act on, or the fix
   would need information from a source the project may not use, say so and
   stop rather than guessing.
4. Make the change: a file under `fixes/` for one-place data problems, pipeline
   or viewer code otherwise, with a test for pipeline changes.
5. Rebuild, run `inspect` at the spot again, run `pytest pipeline`, and look at
   the note's link in the local viewer.
6. Summarise what changed and what you checked. Do not commit, push or comment
   on the issue unless asked.

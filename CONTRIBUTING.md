# Contributing

The goal is a Boston that feels real to walk through. Every correction goes
through this repository, so it is reviewed, kept, and reaches everyone on the
next build. There are four ways to help, from least to most involved.

## 1. Leave a note

Walk the [live model](https://ohinai.github.io/boston-model/). When something
looks wrong, stand where you can see it, look at it, and press **N**. Say what
is wrong or what should be there instead.

The note records exactly where you stood, which way you faced and what you were
looking at. **Save and open a ticket** takes you to GitHub with the ticket
filled in; you only press submit. You need a free GitHub account for that step.
Without one, **Save for later** keeps the note in your browser, and **Notes**
lets you download them to send to someone who can file them.

Open tickets show up in the model as blue markers, so you can see what others
have already reported. Your own unfiled notes are amber.

Good notes say how you know: "the fountain here was removed in 2023, I walk
past it daily" is something a fixer can act on.

## 2. Fix a note

Tickets filed from the viewer carry the label
[`note`](https://github.com/ohinai/boston-model/issues?q=is%3Aissue+is%3Aopen+label%3Anote).
Pick one and fix it.

This is designed to be done with an AI coding assistant.
[AGENTS.md](AGENTS.md) is written for the assistant: it explains the ticket's
position data, the `boston-pipeline inspect` command that shows what the data
holds at that spot, and how to write the fix. In Claude Code the repository
provides a command:

```
/fix-note 12
```

By hand, the same steps apply:

1. Run `boston-pipeline inspect <lat>,<lon>` with the position from the ticket.
2. Write a fix file under [`fixes/`](fixes). [fixes/README.md](fixes/README.md)
   has the two formats: one to change or remove a feature, one to add or
   redraw one.
3. Rebuild, look at the result in the viewer at the ticket's link, and open a
   pull request that says `Fixes #<ticket number>` and where your information
   came from.

Some notes cannot be fixed with a fix file because the pipeline does not draw
that kind of thing yet, or draws it wrongly everywhere. Those are code changes.

## 3. Check an area against aerial photographs

You do not have to wait for a note. `boston-pipeline audit` draws the model on
top of an aerial photograph, one 120 m cell at a time, and proposes trees the
data lacks:

```sh
pip install -e "./pipeline[audit]"     # Pillow; the dev extra has it too
boston-pipeline audit pilot --park "Boston Common"
```

Look at `overlay.jpg` in each folder under `data/audit/pilot/`. In Claude Code,
`/audit "Boston Common"` does the checking and writes a `findings.json` for each
cell. Either way, a finding is only a lead: look at it again, then write a fix
file that cites the photograph in `why`. The audit never changes the model by
itself. [AGENTS.md](AGENTS.md) has the details.

To check at eye level instead, `boston-pipeline ground pilot --park "Boston
Public Garden"` finds openly licensed photographs taken in the park and sets
each beside the model's view from the same spot. It needs the built viewer
(`npm run build` in `web/`) and a Chrome, Edge or Chromium on your machine. In
Claude Code, `/ground "Boston Public Garden"` does the comparing. The
photographs show what to look into; they are visual reference only, so a fix
still needs a source listed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## 4. Work on the code

[docs/ROADMAP.md](docs/ROADMAP.md) lists what is planned, with parks and
gardens first. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the
pieces fit. For anything larger than a small fix, open an issue first so the
approach can be agreed before you write it.

## Setup

You need Python 3.11 or newer and Node.js 22 or newer.

```sh
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS or Linux: source .venv/bin/activate
pip install -e "./pipeline[dev]"
boston-pipeline fetch pilot       # downloads the source map data once
boston-pipeline build pilot       # writes web/public/tiles

cd web
npm install
npm run dev                       # http://localhost:5173
```

After changing a fix file or the pipeline, run `boston-pipeline build pilot`
again and reload the viewer.

## Before opening a pull request

```sh
pytest pipeline          # pipeline tests
cd web && npm run build  # type-checks and builds the viewer
```

Then walk through the area your change affects and look at it. Include a
screenshot, and the viewer link to the spot, for anything visible.

## Ground rules

- **Fixes go in this repository.** Do not ask contributors to edit
  OpenStreetMap instead. It is where the starting data comes from, and you are
  welcome to improve it as well, but this project does not depend on that.
- **Every fix says where it comes from.** Your own observation and photos, or
  an open source listed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Never
  Google Maps, Street View or Google Earth: their terms forbid it.
- **Never commit built or downloaded data.** `data/` and `web/public/tiles/`
  are ignored on purpose.
- **Do not change the world frame.** The origin, projection and tile size in
  `world.py` are permanent; every tile and fix depends on them.
- **Tile layer names are a contract.** Clients rely on them. Adding a layer is
  fine; renaming or removing one needs discussion and a note in
  `docs/ARCHITECTURE.md`.
- **Keep clients thin.** If the browser viewer and a future Unreal project
  would both need something, it belongs in the pipeline's output.

By contributing you agree that your code is released under the MIT licence and
your fixes under the Open Database License, matching the project.

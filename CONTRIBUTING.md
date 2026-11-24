# Contributing

Thank you for helping build Boston. There are three ways to contribute, in
order of how much each one helps.

## 1. Improve OpenStreetMap

Almost everything in the model comes from OpenStreetMap, so the best fix for a
wrong or missing detail is usually to fix it there. It then improves every map
and model that uses OSM, and reaches this project on the next build.

In the viewer, walk to the spot and use **Fix this spot in OpenStreetMap**. The
tags the pipeline reads today:

| What | Tags |
|---|---|
| Building height | `height` in metres, or `building:levels` |
| Parts of a building at different heights | `building:part` with `height` and `min_height` |
| Wall and roof colour | `building:colour`, `roof:colour`, `building:material` |
| Street width | `lanes`, `width`, `parking:left`, `parking:right`, `parking:both` |
| Trees | a node tagged `natural=tree` |
| Shops and places | `shop`, `amenity`, `tourism`, `historic`, with a `name` |
| Traffic signals | a node tagged `highway=traffic_signals` |

Follow OpenStreetMap's own rules: map what you can verify on the ground or from
sources OSM permits, and never copy from Google Maps or Street View.

About half the buildings downtown have no height or storey count. Adding
`building:levels` to a block you know is a real contribution.

## 2. Add a correction here

Some things OpenStreetMap cannot hold, or holds correctly for maps but not for
a 3D model. Those go in [`overrides/`](overrides):

- [`overrides/buildings.toml`](overrides/buildings.toml) sets a building's
  height or colours, or hides it, by OSM id. The file explains the format.
- [`areas.toml`](areas.toml) lists the landmarks the viewer can jump to.

Say where each value comes from in the pull request. A measured or published
figure is needed, not an estimate from a photo.

## 3. Work on the code

[docs/ROADMAP.md](docs/ROADMAP.md) lists what is planned and ends with good
first tasks. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the
pieces fit. For anything larger than a small fix, open an issue first so the
approach can be agreed before you write it.

### Setup

```sh
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS or Linux: source .venv/bin/activate
pip install -e "./pipeline[dev]"
boston-pipeline fetch pilot
boston-pipeline build pilot

cd web
npm install
npm run dev
```

After changing the pipeline, run `boston-pipeline build pilot` again and reload
the viewer.

### Before opening a pull request

```sh
pytest pipeline          # pipeline tests
cd web && npm run build  # type-checks and builds the viewer
```

Then walk through the area your change affects and look at it. Include a
screenshot, and the viewer link to the spot, for anything visible.

### Ground rules

- **Never commit built or downloaded data.** `data/` and `web/public/tiles/`
  are ignored on purpose.
- **Do not change the world frame.** The origin, projection and tile size in
  `world.py` are permanent; every tile and override depends on them.
- **Tile layer names are a contract.** Clients rely on them. Adding a layer is
  fine; renaming or removing one needs discussion and a note in
  `docs/ARCHITECTURE.md`.
- **Keep clients thin.** If the browser viewer and a future Unreal project
  would both need something, it belongs in the pipeline's output.
- **Interpretation lives in `features.py`.** Defaults such as "a building with
  no height is 12 m" go there with a comment, not scattered through the code.

## Data and licences

Only use sources listed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md), or add
a new one there in the same pull request with its licence. The licence must
allow redistribution of derived work. Google Maps, Street View, Google Earth
and other proprietary imagery are not acceptable sources for anything in this
repository.

By contributing you agree that your code is released under the MIT licence and
your data corrections under the Open Database License, matching the project.

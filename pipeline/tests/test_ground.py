"""The ground command: parsing Commons records, choosing photographs, the capture server, the sheet."""

import json
import urllib.error
import urllib.request

import pytest
from shapely.geometry import box

from boston_pipeline import capture, ground, world

LON, LAT = -71.0700, 42.3541
X0, Y0 = (float(v) for v in world.to_local(LON, LAT))


def record(**over):
    ext = {
        "LicenseShortName": {"value": "CC BY 2.0"},
        "Artist": {"value": '<a href="//commons.wikimedia.org/wiki/User:Jo">Jo &amp; Co</a>'},
        "DateTimeOriginal": {"value": "2019-06-14 10:22:31"},
        "ImageDescription": {"value": "<p>The <b>lagoon</b> in June.</p>"},
        "ObjectName": {"value": "Lagoon in June"},
    }
    ext.update(over.pop("ext", {}))
    info = {
        "thumburl": "https://upload.example/thumb/1024px-x.jpg", "width": 4000, "height": 3000, "mime": "image/jpeg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:X.jpg", "extmetadata": ext,
        "metadata": [{"name": "GPSImgDirection", "value": "123/1"}],
    }
    info.update(over.pop("info", {}))
    page = {"pageid": 42, "title": "File:X.jpg", "coordinates": [{"lat": LAT, "lon": LON}], "imageinfo": [info]}
    page.update(over)
    return page


def photo(pageid, dx=0.0, dy=0.0, direction=None, stamp=20200101, **over):
    base = dict(
        pageid=pageid, title="t", page_url="u", thumb_url="http://x/y.jpg", mime="image/jpeg", width=1024, height=768,
        author="a", licence="CC BY 2.0", licence_class="data", date="2020", year=stamp // 10000, stamp=stamp,
        description="", lat=LAT, lon=LON, x=X0 + dx, y=Y0 + dy, direction=direction,
    )
    base.update(over)
    return ground.Photo(**base)


# --- parsing ----------------------------------------------------------------------------------

def test_parse_record():
    p = ground.parse_record(record())
    assert p.pageid == 42 and p.title == "Lagoon in June"
    assert p.author == "Jo & Co"
    assert p.description == "The lagoon in June."
    assert p.licence == "CC BY 2.0" and p.licence_class == "data"
    assert p.year == 2019 and p.stamp == 20190614
    assert p.direction == 123.0
    assert p.width == 4000 and p.page_url.endswith("File:X.jpg")
    assert (p.x, p.y) == pytest.approx((X0, Y0))


def test_parse_record_needs_coordinates_and_info():
    assert ground.parse_record(record(coordinates=[])) is None
    assert ground.parse_record(record(imageinfo=[])) is None


def test_description_is_cut():
    p = ground.parse_record(record(ext={"ImageDescription": {"value": "word " * 200}}))
    assert len(p.description) <= 300


@pytest.mark.parametrize("name, expected", [
    ("Public domain", "data"), ("CC0", "data"), ("CC BY 2.0", "data"), ("CC BY 4.0", "data"),
    ("CC BY-SA 4.0", "look"), ("CC BY-SA 2.5", "look"), ("GFDL", "look"), ("FAL", "look"),
    ("CC BY-NC 2.0", None), ("", None), (None, None),
])
def test_licence_classes(name, expected):
    assert ground.licence_class(name) == expected


@pytest.mark.parametrize("text, year", [
    ("2019-06-14 10:22:31", 2019),
    ('<time datetime="2021-05-02">2 May 2021</time>', 2021),
    ("circa 1905", 1905),
    ("unknown", None),
    ("", None),
])
def test_year_parsing(text, year):
    assert ground.parse_year(text) == year


def test_direction_forms():
    assert ground.parse_direction("123/1") == 123.0
    assert ground.parse_direction("1235/10") == pytest.approx(123.5)
    assert ground.parse_direction(87.25) == 87.25
    assert ground.parse_direction("360") == 0.0
    assert ground.parse_direction("1/0") is None
    assert ground.parse_direction("north") is None
    assert ground.parse_direction(None) is None


# --- filters and thinning ---------------------------------------------------------------------

PARK = box(X0 - 50, Y0 - 50, X0 + 50, Y0 + 50)


def pages_of(*records):
    return {str(r["pageid"]): r for r in records}


def test_filters():
    good = record(pageid=1)
    old = record(pageid=2, ext={"DateTimeOriginal": {"value": "2012-01-01"}})
    undated = record(pageid=3, ext={"DateTimeOriginal": {"value": ""}})
    small = record(pageid=4, info={"width": 700})
    png = record(pageid=5, info={"mime": "image/png"})
    nonfree = record(pageid=6, ext={"LicenseShortName": {"value": "CC BY-NC 2.0"}})
    unlicensed = record(pageid=7, ext={"LicenseShortName": {"value": ""}})
    outside = record(pageid=8, coordinates=[{"lat": LAT + 0.01, "lon": LON}])
    found = ground.eligible(pages_of(good, old, undated, small, png, nonfree, unlicensed, outside), PARK)
    assert [p.pageid for p in found] == [1]


def test_thin_one_per_square_prefers_direction_then_newest():
    a = photo(1, 1, 1, direction=None, stamp=20230101)
    b = photo(2, 2, 2, direction=90.0, stamp=20160101)
    c = photo(3, 3, 3, direction=90.0, stamp=20220101)
    far = photo(4, 100, 100)
    kept = ground.thin([a, b, c, far], limit=10)
    assert [p.pageid for p in kept] == [3, 4]


def test_thin_limit_is_deterministic_and_spread():
    crowd = [photo(100 + i, 0.5 * i, 0.0, stamp=20200101 + i) for i in range(30)]  # all but one share squares
    row = [photo(i, 120 * (i % 5), 120 * (i // 5)) for i in range(1, 11)]  # a spread of ten
    first = ground.thin(crowd + row, limit=8)
    again = ground.thin(list(reversed(crowd + row)), limit=8)
    assert [p.pageid for p in first] == [p.pageid for p in again]
    assert len(first) == 8
    cells = {(p.x // 100, p.y // 100) for p in first}
    assert len(cells) >= 8 - 1  # taking turns over coarse squares, not the first N


def test_thin_keeps_at_most_limit():
    spread = [photo(i, 40 * i, 0) for i in range(20)]
    assert len(ground.thin(spread, limit=5)) == 5
    assert len(ground.thin(spread, limit=50)) == 20


def test_view_headings():
    assert ground.view_headings(None) == [0, 60, 120, 180, 240, 300]
    assert ground.view_headings(90.0) == [55, 90, 125]
    assert ground.view_headings(10.0) == [335, 10, 45]
    assert ground.view_headings(350.4) == [315, 350, 25]
    assert ground.view_id(42, 5) == "42_005"
    assert ground.compass_word(45) == "NE" and ground.compass_word(350) == "N"


# --- the capture server -----------------------------------------------------------------------

@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    (root / "tiles").mkdir(parents=True)
    (root / "index.html").write_text("<html>viewer</html>", encoding="utf-8")
    (root / "tiles" / "tileset.json").write_text("{}", encoding="utf-8")
    (root / "tiles" / "a.glb").write_bytes(b"glTF")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    return root


def status_of(request):
    try:
        with urllib.request.urlopen(request) as reply:
            return reply.status, reply.read(), reply.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def test_capture_with_fake_browser(site, tmp_path, monkeypatch):
    seen = {}

    def fake_launch(url, width, height, profile_dir):
        base = url.split("/?")[0]
        assert url.endswith("/?capture=/__capture/views.json")
        with urllib.request.urlopen(base + "/__capture/views.json") as reply:
            seen["views"] = json.load(reply)
        seen["index"] = status_of(base + "/")
        seen["glb"] = status_of(base + "/tiles/a.glb")
        seen["list"] = status_of(base + "/tiles/")
        seen["outside"] = status_of(base + "/../secret.txt")
        seen["outside2"] = status_of(base + "/%2e%2e/secret.txt")
        for ident in ("a_000", "b_090"):
            urllib.request.urlopen(urllib.request.Request(f"{base}/__capture/shot/{ident}.jpg", data=b"\xff\xd8jpeg", method="POST"))
        for bad in ("../x", "a%20b", "x.png", ".jpg"):
            seen[bad] = status_of(urllib.request.Request(f"{base}/__capture/shot/{bad}", data=b"zz", method="POST"))
        body = json.dumps({"captured": 2, "failed": []}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{base}/__capture/shot/done", data=body, method="POST"))
        return None

    monkeypatch.setattr(capture, "_launch", fake_launch)
    views = [{"id": "a_000", "lat": 1, "lon": 2, "heading": 0, "pitch": 0}, {"id": "b_090", "lat": 1, "lon": 2, "heading": 90, "pitch": 0}]
    out = tmp_path / "out"
    result = capture.capture(views, out, site_dir=site, timeout=20)

    assert result["completed"] and result["captured"] == 2 and sorted(result["arrived"]) == ["a_000", "b_090"]
    assert (out / "a_000.jpg").read_bytes() == b"\xff\xd8jpeg"
    assert sorted(p.name for p in out.iterdir()) == ["a_000.jpg", "b_090.jpg"]
    assert seen["views"]["post"] == "/__capture/shot/" and seen["views"]["views"] == views
    assert seen["views"]["width"] == 1024 and seen["views"]["fov"] == 70
    assert seen["index"][0] == 200 and b"viewer" in seen["index"][1]
    assert seen["glb"][0] == 200 and seen["glb"][2]["Content-Type"] == "model/gltf-binary"
    assert seen["list"][0] == 404
    assert seen["outside"][0] in (403, 404) and b"secret" not in seen["outside"][1]
    assert seen["outside2"][0] in (403, 404) and b"secret" not in seen["outside2"][1]
    for bad in ("../x", "a%20b", "x.png", ".jpg"):
        assert seen[bad][0] in (400, 404)


def test_capture_times_out_without_done(site, tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "_launch", lambda *a: None)
    result = capture.capture([], tmp_path / "o", site_dir=site, timeout=1)
    assert result["completed"] is False and result["arrived"] == []


def test_capture_needs_the_built_viewer(tmp_path):
    with pytest.raises(RuntimeError, match="npm run build"):
        capture.capture([], tmp_path / "o", site_dir=tmp_path / "nothing")


def test_capture_rejects_bad_ids(site, tmp_path):
    with pytest.raises(ValueError):
        capture.capture([{"id": "../x", "lat": 0, "lon": 0, "heading": 0, "pitch": 0}], tmp_path / "o", site_dir=site)


def test_browser_arguments():
    args = capture.browser_args("http://x/", 1024, 768, capture.Path("p"))
    assert "--headless=new" in args and "--window-size=1024,768" in args and args[-1] == "http://x/"


# --- the sheet --------------------------------------------------------------------------------

def test_sheet_size_and_content():
    from PIL import Image

    p = photo(1, direction=90.0, title="Lagoon in June", date="2019-06-14")
    picture = Image.new("RGB", (1024, 683), (40, 120, 60))
    models = [(55, Image.new("RGB", (1024, 768), (90, 90, 200))), (90, Image.new("RGB", (1024, 768), (200, 90, 90))), (125, None)]
    sheet = ground.render_sheet(p, picture, models)
    tile = 768 * 512 // 1024
    assert sheet.size == (1536, 36 + 8 + 683 + 8 + (26 + tile + 8))
    assert len({sheet.getpixel((x, y)) for x in range(0, 1536, 7) for y in range(0, sheet.height, 7)}) > 5  # not blank
    assert sheet.getpixel((5, sheet.height - 3)) == (255, 255, 255)


def test_sheet_without_model_pictures():
    from PIL import Image

    sheet = ground.render_sheet(photo(1), Image.new("RGB", (1024, 1536), (10, 10, 10)), [(h, None) for h in ground.ALL_ROUND])
    assert sheet.width == 1536 and sheet.height == 36 + 8 + 768 + 8 + 60
    assert len({sheet.getpixel((x, y)) for x in range(0, 1536, 7) for y in range(0, sheet.height, 7)}) > 2


def test_search_centres_cover_the_box():
    centres = ground.search_centres(box(0, 0, 400, 300))
    assert len(centres) >= 6
    points = [tuple(float(v) for v in world.to_local(lon, lat)) for lat, lon in centres]
    assert all(-1 <= x <= 401 and -1 <= y <= 301 for x, y in points)

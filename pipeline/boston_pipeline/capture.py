"""Ask the built viewer for pictures of the model, from given spots and headings.

The viewer (`web/dist`) has a capture mode. This module serves the built site on a local port, opens it in
a headless Chromium-based browser at `/?capture=/__capture/views.json`, and receives the pictures the
page posts back:

    GET  /__capture/views.json         {"width", "height", "fov", "post", "views": [{id, lat, lon, heading, pitch[, altitude]}]}
    POST /__capture/shot/<id>.jpg      the JPEG bytes
    POST /__capture/shot/done          {"captured": n, "failed": [{"id", "reason"}], "error": optional}

The browser launcher (`_launch`) is kept apart so tests can replace it with a plain HTTP client.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEWS_PATH = "/__capture/views.json"
SHOT_PREFIX = "/__capture/shot/"
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
MAX_BODY = 64 * 1024 * 1024
BASE_TIMEOUT = 120.0
PER_VIEW_TIMEOUT = 20.0
PROGRESS_EVERY = 10

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json", ".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".webp": "image/webp",
    ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8", ".ico": "image/x-icon", ".bin": "application/octet-stream",
}


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


class _State:
    def __init__(self, site_dir: Path, out_dir: Path, views_payload: bytes) -> None:
        self.site_dir = site_dir.resolve()
        self.out_dir = out_dir
        self.views_payload = views_payload
        self.lock = threading.Lock()
        self.arrived: list[str] = []
        self.summary: dict | None = None
        self.done = threading.Event()


def _handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # quiet
            pass

        def _reply(self, status: int, body: bytes = b"", kind: str = "text/plain; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path == VIEWS_PATH:
                return self._reply(200, state.views_payload, "application/json")
            from urllib.parse import unquote

            relative = unquote(path).lstrip("/")
            if not relative or relative.endswith("/"):
                relative += "index.html"
            if "\x00" in relative:
                return self._reply(400, b"bad path")
            target = (state.site_dir / relative).resolve()
            try:
                target.relative_to(state.site_dir)
            except ValueError:
                return self._reply(403, b"outside the site")
            if not target.is_file():  # no directory listing
                return self._reply(404, b"not found")
            self._reply(200, target.read_bytes(), content_type(target))

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if not path.startswith(SHOT_PREFIX):
                return self._reply(404, b"not found")
            name = path[len(SHOT_PREFIX):]
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                return self._reply(400, b"bad length")
            body = self.rfile.read(length)
            if name == "done":
                try:
                    summary = json.loads(body.decode("utf-8") or "{}")
                    if not isinstance(summary, dict):
                        raise ValueError("not an object")
                except ValueError:
                    return self._reply(400, b"bad json")
                state.summary = summary
                state.done.set()
                return self._reply(200, b"ok")
            if not name.endswith(".jpg") or not ID_PATTERN.fullmatch(name[:-4]):
                return self._reply(400, b"bad id")
            ident = name[:-4]
            state.out_dir.mkdir(parents=True, exist_ok=True)
            (state.out_dir / f"{ident}.jpg").write_bytes(body)
            with state.lock:
                if ident not in state.arrived:
                    state.arrived.append(ident)
                count = len(state.arrived)
            if count % PROGRESS_EVERY == 0:
                print(f"  {count} pictures received")
            self._reply(200, b"ok")

    return Handler


# --- the browser ------------------------------------------------------------------------------

def find_browser() -> str:
    """Path of a Chromium-based browser: $BOSTON_BROWSER, else Chrome, Edge or Chromium in the usual places."""
    chosen = os.environ.get("BOSTON_BROWSER")
    if chosen:
        if Path(chosen).exists() or shutil.which(chosen):
            return chosen
        raise RuntimeError(f"BOSTON_BROWSER is set to {chosen!r}, which does not exist")
    places = []
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), r"C:\Program Files", r"C:\Program Files (x86)"):
        if base:
            places += [
                Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe",
                Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                Path(base) / "Chromium" / "Application" / "chrome.exe",
            ]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        places += [Path(local) / "Google" / "Chrome" / "Application" / "chrome.exe", Path(local) / "Chromium" / "Application" / "chrome.exe"]
    places += [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    for place in places:
        if place.exists():
            return str(place)
    for command in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(command)
        if found:
            return found
    raise RuntimeError(
        "no Chromium-based browser found (looked for Chrome, Edge and Chromium); install one or set BOSTON_BROWSER to its path"
    )


def browser_args(url: str, width: int, height: int, profile_dir: Path) -> list[str]:
    return [
        "--headless=new", f"--window-size={width},{height}", f"--user-data-dir={profile_dir}",
        "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--mute-audio",
        "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
        "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
        url,
    ]


def _launch(url: str, width: int, height: int, profile_dir: Path):
    """Start the headless browser; returns the process (anything with poll, terminate, kill and wait)."""
    command = [find_browser(), *browser_args(url, width, height, profile_dir)]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop(process) -> None:
    if process is None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


# --- running ----------------------------------------------------------------------------------

def check_site(site_dir: Path) -> None:
    missing = [p for p in ("index.html", "tiles/tileset.json") if not (site_dir / p).is_file()]
    if missing:
        raise RuntimeError(
            f"the built viewer is missing {', '.join(missing)} in {site_dir}; "
            "run `boston-pipeline build <area>` and then `npm run build` in web/ first"
        )


def capture(
    views: list[dict], out_dir: Path, *, site_dir: Path, width: int = 1024, height: int = 768, fov: float = 70,
    timeout: float | None = None, query: str = "",
) -> dict:
    """Render every view in the model, writing `<out_dir>/<id>.jpg`.

    `views` are dicts with id, lat, lon, heading, pitch, and optionally altitude (metres; above 2 is flying).
    `query` is added to the viewer's address, for the viewer's own switches such as `lod=0` (see web/src/city.ts).
    Returns the viewer's own summary (captured, failed, error) plus `arrived` (ids whose pictures
    were received) and `completed` (whether the viewer reported it was done before the timeout).
    """
    site_dir, out_dir = Path(site_dir), Path(out_dir)
    check_site(site_dir)
    for view in views:
        if not ID_PATTERN.fullmatch(str(view["id"])):
            raise ValueError(f"bad view id {view['id']!r}: use letters, digits, _ and -")
    out_dir.mkdir(parents=True, exist_ok=True)
    if timeout is None:
        timeout = BASE_TIMEOUT + PER_VIEW_TIMEOUT * len(views)
    payload = json.dumps({"width": width, "height": height, "fov": fov, "post": SHOT_PREFIX, "views": views}).encode("utf-8")
    state = _State(site_dir, out_dir, payload)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    # The browser is stopped while connections are still open; that is not an error worth printing.
    server.handle_error = lambda request, client_address: None
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    profile = Path(tempfile.mkdtemp(prefix="boston-capture-"))
    process = None
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/?capture={VIEWS_PATH}" + (f"&{query}" if query else "")
        print(f"capturing {len(views)} views of the model ...")
        process = _launch(url, width, height, profile)
        deadline = time.monotonic() + timeout
        exited_at = None
        while not state.done.wait(0.5):
            if time.monotonic() > deadline:
                print(f"  timed out after {timeout:.0f} s with {len(state.arrived)} pictures")
                break
            if process is not None and process.poll() is not None:  # the browser quit; give late posts a moment
                exited_at = exited_at or time.monotonic()
                if time.monotonic() - exited_at > 3:
                    print("  the browser exited before the viewer reported it was done")
                    break
    finally:
        _stop(process)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        shutil.rmtree(profile, ignore_errors=True)
    result = dict(state.summary or {})
    result["completed"] = state.done.is_set()
    result["arrived"] = list(state.arrived)
    return result

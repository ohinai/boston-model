"""Downloading the map: a server with an out-of-date copy is passed over."""

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from boston_pipeline import osm

BOX = (-71.1, 42.3, -71.0, 42.4)


def _stamp(days_old: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_old)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(days_old: float | None) -> dict:
    payload = {"elements": [{"type": "node", "id": 1, "lat": 42.35, "lon": -71.06, "tags": {"natural": "tree"}}]}
    if days_old is not None:
        payload["osm3s"] = {"timestamp_osm_base": _stamp(days_old)}
    return payload


class _Reply:
    headers: dict = {}

    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def _serve(monkeypatch, replies: dict[str, dict]) -> list[str]:
    """Answer each server with its reply; returns the list that records which were asked."""
    asked = []

    def fake_urlopen(request, timeout=None):
        asked.append(request.full_url)
        return _Reply(replies[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(osm, "ROUND_PAUSE", 0)
    return asked


def test_data_age_reads_the_servers_timestamp():
    now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    assert osm.data_age({"osm3s": {"timestamp_osm_base": "2026-11-25T00:00:00Z"}}, now) == pytest.approx(0.5)
    assert osm.data_age({"osm3s": {"timestamp_osm_base": "2026-07-15T15:22:01Z"}}, now) == pytest.approx(132.86, abs=0.01)
    assert osm.data_age({"osm3s": {}}, now) is None
    assert osm.data_age({}, now) is None
    assert osm.data_age({"osm3s": {"timestamp_osm_base": "yesterday"}}, now) is None


def test_fetch_passes_over_a_server_with_an_old_copy(tmp_path, monkeypatch, capsys):
    first, second, third = osm.ENDPOINTS
    asked = _serve(monkeypatch, {first: _payload(133), second: _payload(None), third: _payload(0.01)})
    dest = osm.fetch(BOX, tmp_path / "osm.json")

    assert asked == [first, second, third]
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert osm.data_age(saved) < 1 and saved["bbox_wsen"] == list(BOX)
    printed = capsys.readouterr().out
    assert "133 days old" in printed and "does not say how old" in printed


def test_fetch_fails_when_every_copy_is_old(tmp_path, monkeypatch):
    asked = _serve(monkeypatch, {url: _payload(osm.MAX_AGE_DAYS + 1) for url in osm.ENDPOINTS})
    dest = tmp_path / "osm.json"
    with pytest.raises(RuntimeError, match="days old"):
        osm.fetch(BOX, dest)
    assert len(asked) == osm.ROUNDS * len(osm.ENDPOINTS) and not dest.exists()


def test_fetch_takes_a_copy_within_the_limit(tmp_path, monkeypatch):
    asked = _serve(monkeypatch, {osm.ENDPOINTS[0]: _payload(osm.MAX_AGE_DAYS - 0.5)})
    osm.fetch(BOX, tmp_path / "osm.json")
    assert asked == [osm.ENDPOINTS[0]]

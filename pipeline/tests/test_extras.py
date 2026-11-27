"""The hook that lets sports.py, playgrounds.py and stadium.py add kinds of object (extras.py)."""

import sys
import types

import pytest

from boston_pipeline import extras, furniture
from boston_pipeline.features import _object
from boston_pipeline.mesh import MeshBuilder


@pytest.fixture
def fake_module(monkeypatch):
    """A module called `sports`, standing in for a lane's real one, with all three hooks."""
    module = types.ModuleType(f"{extras.__package__}.sports")
    calls = []

    def shape(parts, x, y, direction, opts):
        calls.append((x, y))  # nothing drawn: the test is about dispatch

    module.SHAPES = {"fake_goal": shape}
    module.interpret = lambda tags: ("fake_goal", "wide", 3.0) if tags.get("fake") == "goal" else None
    module.objects = lambda scene: [scene] if scene == "marker" else []
    monkeypatch.setitem(sys.modules, module.__name__, module)
    extras.loaded.cache_clear()
    yield module, calls
    extras.loaded.cache_clear()


def test_modules_that_do_not_exist_are_skipped():
    extras.loaded.cache_clear()
    assert extras.shape("no_such_kind") is None
    assert extras.interpret({"amenity": "nothing"}) is None
    assert extras.objects(object()) == []


def test_shape_interpret_and_objects_reach_the_module(fake_module):
    module, calls = fake_module
    assert extras.shape("fake_goal") is module.SHAPES["fake_goal"]
    assert "fake_goal" in extras.kinds()
    assert extras.interpret({"fake": "goal"}) == ("fake_goal", "wide", 3.0)
    assert extras.interpret({"fake": "other"}) is None
    assert extras.objects("marker") == ["marker"]


def test_a_point_with_an_unknown_kind_becomes_the_modules_object(fake_module):
    thing = _object("n1", {"fake": "goal"}, 4.0, 5.0)
    assert (thing.kind, thing.variant, thing.size) == ("fake_goal", "wide", 3.0)
    assert _object("n2", {"fake": "nothing"}, 4.0, 5.0) is None


def test_add_object_draws_a_kind_the_module_defines(fake_module):
    _module, calls = fake_module
    furniture.add_object(MeshBuilder("furniture", colours=True, features=True), "fake_goal", 1.0, 2.0)
    assert calls == [(1.0, 2.0)]


def test_add_object_still_rejects_an_unknown_kind():
    extras.loaded.cache_clear()
    with pytest.raises(ValueError, match="unknown furniture kind"):
        furniture.add_object(MeshBuilder("furniture", colours=True, features=True), "no_such_kind", 0.0, 0.0)

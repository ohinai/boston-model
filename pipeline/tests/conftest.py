"""Tests run on the procedural shapes unless they are about the hand-made assets.

Once a model exists in assets/, the pipeline draws it instead of the procedural shape, and tests that count a shape's
vertices or triangles would then be measuring the model. So every test sets BOSTON_NO_ASSETS, except the modules
named `test_asset*`, which are the ones that test the assets (and the loader) and see the real library.
"""

import pytest

from boston_pipeline import assets


@pytest.fixture(autouse=True)
def procedural_unless_testing_assets(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1].startswith("test_asset"):
        monkeypatch.delenv(assets.DISABLE_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(assets.DISABLE_VARIABLE, "1")

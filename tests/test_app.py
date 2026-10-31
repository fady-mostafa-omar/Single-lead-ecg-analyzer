"""The Gradio app builds (all tabs and callbacks) and its text helpers work."""
import os
import sys

import pytest

from holter import vendor

if not os.path.isdir(vendor.path("rhythm", "models")):
    pytest.skip("vendor/ is missing: run python scripts/get_models.py", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import app  # noqa: E402


def test_demo_builds():
    demo, _ = app.build_demo()
    assert demo is not None


def test_about_text_has_the_numbers():
    md = app.about_md()
    assert "MIT-BIH DS2" in md and "Chapman-Shaoxing" in md and "Not" not in md.split("### Accuracy")[0][:20]

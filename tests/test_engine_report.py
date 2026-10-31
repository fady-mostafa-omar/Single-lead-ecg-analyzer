"""End to end on the bundled MIT-BIH examples with the three projects' real models (run scripts/get_models.py first)."""
import json
import os

import numpy as np
import pytest

from holter import vendor

if not os.path.isdir(vendor.path("rhythm", "models")):
    pytest.skip("vendor/ is missing: run python scripts/get_models.py", allow_module_level=True)

from holter.derive import summarize  # noqa: E402
from holter.engine import Analyzer, check_signal  # noqa: E402
from holter.report import build_pdf  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    z = np.load(vendor.path("rhythm", "app", "examples", "examples.npz"))
    meta = {m["key"]: m for m in json.loads(str(z["meta"]))}
    return Analyzer(), z, meta


def test_polarity_rule():
    fs = 360
    t = np.arange(60 * fs) / fs
    beat = np.exp(-((t % 1.0) - 0.5) ** 2 / (2 * 0.01 ** 2))          # 60 bpm, upright R waves of 1 mV
    assert not check_signal(beat, fs)[0]
    assert check_signal(-beat, fs)[0]


def test_normal_recording(setup):
    A, z, meta = setup
    x = z["mit100"][: 5 * 60 * 360] / 1000.0
    r = A.analyze(x, 360, placement="chest")
    s = summarize(r)
    assert not r["inverted"] and not r["morph_run"]
    assert 70 < s["hr_mean"] < 80                                      # MIT-BIH 100: about 75 bpm
    assert not s["af_intervals"] and s["readable_share"] > 0.95
    assert s["counts"]["V"] <= 2


def test_mit200_is_not_flipped(setup):
    A, z, _ = setup
    assert not check_signal(z["mit200"] / 1000.0, 360)[0]              # upright normal beats, big V beats point down


def test_af_recording_and_pdf(setup, tmp_path):
    A, z, meta = setup
    x = z["mit210"][: 10 * 60 * 360] / 1000.0                          # AF throughout
    r = A.analyze(x, 360, placement="chest")
    s = summarize(r)
    assert s["af_burden"] > 0.8
    pdf = build_pdf(r, s, str(tmp_path / "r.pdf"), patient=dict(name="Test", age="70", sex="Male"), source="test")
    data = open(pdf, "rb").read()
    assert data[:4] == b"%PDF" and len(data) > 50_000


def test_short_and_limb(setup):
    A, z, _ = setup
    z3 = np.load(vendor.path("morphology", "app", "examples", "examples.npz"))
    m = json.loads(str(z3["meta"]))[0]                                 # a 10-s PTB-XL test ECG, limb lead II
    r = A.analyze(z3[m["key"]] / 1000.0, 100, placement="limb")
    s = summarize(r)
    assert r["morph_run"] and len(r["strip_ok"]) == 1
    assert len(s["groups"]) == 5
    with pytest.raises(ValueError):
        A.analyze(np.zeros(500), 100)                                  # 5 s: too short

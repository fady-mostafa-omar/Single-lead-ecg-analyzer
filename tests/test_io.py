import datetime as dt

import numpy as np
import pytest

from holter import io as hio


def test_csv_header_and_column_choice(tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("time,V1,II\n" + "".join(f"{i},{0.1 * i},{0.2 * i}\n" for i in range(50)))
    names, has_header = hio.csv_columns(str(p))
    assert has_header and names == ["time", "V1", "II"]
    assert hio.default_column(names) == "II"
    x = hio.read_csv(str(p), "II")
    assert len(x) == 50 and np.isclose(x[10], 2.0)


def test_csv_without_header(tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("".join(f"{0.01 * i}\n" for i in range(20)))
    names, has_header = hio.csv_columns(str(p))
    assert not has_header and names == ["Column 1"]
    assert len(hio.read_csv(str(p), "Column 1")) == 20


def test_parse_start():
    assert hio.parse_start("") == (None, False)
    t, known = hio.parse_start("08:30")
    assert t.time() == dt.time(8, 30) and not known
    t, known = hio.parse_start("2026-10-28 09:14:05")
    assert t == dt.datetime(2026, 10, 28, 9, 14, 5) and known
    with pytest.raises(ValueError):
        hio.parse_start("half past eight")


def test_wfdb_roundtrip_units(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    fs = 250
    x = np.sin(np.arange(10 * fs) / fs * 2 * np.pi)          # in mV
    wfdb.wrsamp("rec", fs=fs, units=["uV"], sig_name=["ECG"], p_signal=(x * 1000)[:, None], fmt=["16"],
                write_dir=str(tmp_path), base_time=dt.time(7, 5, 0))
    rec = hio.wfdb_record([str(tmp_path / "rec.hea"), str(tmp_path / "rec.dat")])
    names, f = hio.wfdb_channels(rec)
    assert names == ["ECG"] and f == fs
    sig, f, start, date_known = hio.read_wfdb(rec, "ECG")
    assert np.allclose(sig, x, atol=2e-3)                      # µV converted back to mV
    assert start.time() == dt.time(7, 5) and not date_known

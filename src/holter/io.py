"""Reading a single-lead recording: CSV (one column per lead) or PhysioNet WFDB (.hea + .dat)."""
import datetime as dt
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

PREFERRED = ("II", "MLII", "ML II", "LEAD II", "ECG II", "ECG", "ECG1", "CH1")


def default_column(names):
    for n in names:
        if str(n).strip().upper().replace("_", " ") in PREFERRED:
            return n
    return names[0]


def csv_columns(path):
    head = pd.read_csv(path, nrows=5, header=None)
    if pd.to_numeric(head.iloc[0], errors="coerce").isna().all():
        return [str(v) for v in head.iloc[0]], True
    return [f"Column {i + 1}" for i in range(head.shape[1])], False


def read_csv(path, column):
    names, has_header = csv_columns(path)
    df = pd.read_csv(path, header=0 if has_header else None)
    idx = names.index(column) if column in names else 0
    return pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna().to_numpy(float)


def wfdb_record(paths):
    """paths: the uploaded files (at least one .hea and its signal file). Returns the record name in a temp folder."""
    tmp = tempfile.mkdtemp(prefix="wfdb_")
    for p in paths:
        shutil.copy(p, os.path.join(tmp, os.path.basename(p)))
    heads = [p for p in os.listdir(tmp) if p.lower().endswith(".hea")]
    if not heads:
        raise ValueError("No .hea header file among the uploaded files.")
    return os.path.join(tmp, heads[0][:-4])


def wfdb_channels(rec_path):
    import wfdb
    h = wfdb.rdheader(rec_path)
    return list(h.sig_name), float(h.fs)


def read_wfdb(rec_path, channel):
    """Returns (signal in mV, fs, start time or None, True when the start date is known)."""
    import wfdb
    names, _ = wfdb_channels(rec_path)
    r = wfdb.rdrecord(rec_path, channel_names=[channel if channel in names else names[0]])
    x = r.p_signal[:, 0].astype(float)
    unit = str(r.units[0]).strip().lower()
    if unit in ("uv", "µv", "μv"):
        x = x / 1000.0
    elif unit == "v":
        x = x * 1000.0
    start, date_known = None, False
    if getattr(r, "base_time", None) is not None:
        date_known = getattr(r, "base_date", None) is not None
        start = dt.datetime.combine(r.base_date if date_known else dt.date(2000, 1, 1), r.base_time)
    return np.nan_to_num(x), float(r.fs), start, date_known


def parse_start(text):
    """'HH:MM', 'HH:MM:SS' or 'YYYY-MM-DD HH:MM[:SS]' -> (datetime, date known). Empty -> (None, False)."""
    text = (text or "").strip()
    if not text:
        return None, False
    for fmt, has_date in (("%Y-%m-%d %H:%M:%S", True), ("%Y-%m-%d %H:%M", True), ("%H:%M:%S", False), ("%H:%M", False)):
        try:
            t = dt.datetime.strptime(text, fmt)
            return (t if has_date else dt.datetime.combine(dt.date(2000, 1, 1), t.time())), has_date
        except ValueError:
            pass
    raise ValueError(f"Start time '{text}' not understood: use HH:MM, HH:MM:SS or YYYY-MM-DD HH:MM.")

"""Holter measures on hand-made engine outputs (no models needed)."""
import numpy as np

from holter import derive


def fake_result(dur=600.0, beat_t=None, labels=None, af=(), unreadable=(), placement="chest"):
    n_strip = int(dur // 10)
    ok = np.ones(n_strip, bool)
    for a, b in unreadable:
        ok[int(a // 10):int(np.ceil(b / 10))] = False
    if beat_t is None:
        beat_t = np.arange(0.5, dur, 1.0)                       # 60 bpm
    beat_t = np.asarray(beat_t, float)
    labels = np.array(labels if labels is not None else ["N"] * len(beat_t))
    prob = np.zeros((len(beat_t), 5), np.float32)
    for i, c in enumerate(labels):
        prob[i, "NSVFQ".index(c)] = 1
    ws = np.arange(0, dur - 30 + 1e-9, 15.0)
    wc = np.array(["A" if any(a <= s and s + 30 <= b for a, b in af) else "N" for s in ws])
    wp = np.zeros((len(ws), 4), np.float32)
    for i, c in enumerate(wc):
        wp[i, "NAO~".index(c)] = 1
    return dict(fs=250, dur=dur, start_time=None, notes=[], placement=placement, morph_run=placement == "limb",
                inverted=False, scale=1.0, beat_t=beat_t, beat_label=labels, beat_prob=prob, peaks_t=beat_t,
                win_start=ws, win_end=np.r_[ws[1:], dur], win_prob=wp, win_class=wc, win_class_names=list("NAO~"),
                strip_est=np.where(ok, 25.0, -5.0), strip_ok=ok, strip_p5=np.zeros((n_strip, 5)),
                strip_p23=np.zeros((n_strip, 3)), thr5=dict(NORM=.5, MI=.2, STTC=.2, CD=.2, HYP=.1), thr23={},
                names23=["NORM", "IMI", "CLBBB"], reliability={}, gate_threshold=7.1,
                signal=np.zeros(int(dur * 250), np.float32), signal_fs=250, seconds=1.0)


def test_helpers():
    assert derive.runs_of([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert derive._count_pattern("nVnVnVnn", "nV", 3) == 1
    assert derive._count_pattern("nVnVn", "nV", 3) == 0
    assert derive._merge([(0, 10), (10.5, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert derive._join([(0, 1), (3, 4), (20, 20)]) == [(0, 4)]


def test_regular_rhythm():
    s = derive.summarize(fake_result())
    assert s["n_beats"] == 600 and s["counts"]["V"] == 0
    assert abs(s["hr_mean"] - 60) < 0.5
    assert not s["pauses"] and not s["af_intervals"] and s["readable_share"] == 1.0
    assert any("No atrial fibrillation" in f for f in s["findings"])


def test_pause_couplet_and_unreadable():
    t = list(np.arange(0.5, 300, 1.0)) + list(np.arange(302.5, 600, 1.0))   # one 3-s gap = pause
    lab = ["N"] * len(t)
    lab[100], lab[101] = "V", "V"                                           # couplet
    s = derive.summarize(fake_result(beat_t=t, labels=lab, unreadable=[(500, 520)]))
    assert len(s["pauses"]) == 1 and abs(s["pauses"][0]["dur"] - 3.0) < 1e-6
    assert s["patterns"]["V_couplets"] == 1 and s["counts"]["V"] == 2
    assert s["readable_s"] == 580 and s["n_beats"] < len(t)                # beats in 500-520 s left out
    assert any(e[0] == "pause" for e in s["events"])


def test_af_burden_and_s_beats_inside_af():
    t = np.arange(0.5, 600, 0.8)
    lab = np.array(["N"] * len(t))
    lab[np.where((t > 200) & (t < 400))[0][::3]] = "S"
    lab[np.where(t < 100)[0][::20]] = "S"
    s = derive.summarize(fake_result(beat_t=t, labels=lab, af=[(195, 405)]))
    assert len(s["af_intervals"]) == 1
    assert 0.3 < s["af_burden"] < 0.4                                       # about 210 s of 600 s
    (a, b), = s["af_intervals"]
    outside = (t < a) | (t >= b)
    assert s["counts"]["S"] == int((lab[outside] == "S").sum())            # S inside AF not counted
    assert s["s_in_af"] > 0


def test_morphology_off_for_chest_and_text():
    s = derive.summarize(fake_result(placement="chest"))
    assert not s["morph_run"] and all(g["share"] == 0 for g in s["groups"])
    assert any("not performed" in f for f in s["findings"])


def test_period_table_merges_short_last_block():
    s = derive.summarize(fake_result(dur=1230.0, beat_t=np.arange(0.5, 1230, 1.0)))
    assert len(s["hours"]) == 2                                             # 0-600, 600-1230 (last 30 s joined)
    assert s["hours"]["beats"].sum() == s["n_beats"]

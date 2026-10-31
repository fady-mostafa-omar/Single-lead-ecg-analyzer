"""Holter measures computed from the engine output: readable time, heart rate, pauses, ectopy and its patterns,
atrial fibrillation episodes, morphology alerts, hourly table and the event strips for the report."""
import datetime as dt

import numpy as np
import pandas as pd

STRIP = 10
PAUSE_S = 2.0
BRADY, TACHY = 50, 100
MIN_READABLE_MIN = 40            # seconds of readable signal for a minute to count in the heart-rate trend
PERSISTENT, INTERMITTENT = 0.50, 0.10
GROUP = {"NORM": "Normal ECG", "MI": "Myocardial infarction pattern", "STTC": "ST/T change",
         "CD": "Conduction disturbance", "HYP": "Hypertrophy (voltage)"}
PHRASE = {"MI": "a myocardial infarction pattern", "STTC": "an ST/T change", "CD": "a conduction disturbance",
          "HYP": "hypertrophy (voltage criteria)"}
UNCLASSIFIED_CAVEAT = 0.05       # share of unclassified beats above which the beat counts get a warning
ECTOPY_CAVEAT = 0.10             # share of extra beats inside AF episodes above which the AF call gets a warning
SUBNAME = {"IMI": "Inferior myocardial infarction", "AMI": "Anterior myocardial infarction",
           "LMI": "Lateral myocardial infarction", "PMI": "Posterior myocardial infarction", "STTC": "ST/T change",
           "NST_": "Non-specific ST change", "ISC_": "Non-specific ischaemia", "ISCA": "Ischaemia, anterior leads",
           "ISCI": "Ischaemia, inferior leads", "LAFB/LPFB": "Left fascicular block", "IRBBB": "Incomplete RBBB",
           "_AVB": "AV block", "IVCD": "Intraventricular conduction delay", "CRBBB": "Complete right bundle branch block",
           "CLBBB": "Complete left bundle branch block", "ILBBB": "Incomplete LBBB", "WPW": "WPW pattern",
           "LVH": "Left ventricular hypertrophy", "LAO/LAE": "Left atrial enlargement",
           "RAO/RAE": "Right atrial enlargement", "RVH": "Right ventricular hypertrophy", "SEHYP": "Septal hypertrophy"}


def clock(res, t):
    """Time label: clock time when the start time is known, otherwise time from the start."""
    if res.get("start_time") is not None:
        return (res["start_time"] + dt.timedelta(seconds=float(t))).strftime("%H:%M:%S")
    t = int(round(t))
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def runs_of(mask):
    """Start and end indices (inclusive) of runs of True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def summarize(res):
    dur, ok = res["dur"], res["strip_ok"]
    n_strip = len(ok)
    readable_s = float(ok.sum() * STRIP)
    strip_of = lambda t: np.clip((np.asarray(t) // STRIP).astype(int), 0, max(n_strip - 1, 0))
    in_ok = lambda t: (np.asarray(t) < n_strip * STRIP) & ok[strip_of(t)] if n_strip else np.zeros(len(np.atleast_1d(t)), bool)

    # beats that fall in readable strips
    bt, bl = res["beat_t"], res["beat_label"]
    valid = in_ok(bt) if len(bt) else np.zeros(0, bool)
    vt, vl = bt[valid], bl[valid].copy()

    # rhythm windows: each window owns the time up to the next start; counted when >= half of it is readable
    ws, we, wc, wp = res["win_start"], res["win_end"], res["win_class"], res["win_prob"]
    w_read = np.array([_readable_fraction(ok, s, min(s + 30, dur)) for s in ws]) if len(ws) else np.zeros(0)
    af_win = (wc == "A") & (w_read >= 0.5)
    af_int = _merge([(float(s), float(e)) for s, e, a in zip(ws, we, af_win) if a])
    af_time = sum(e - s for s, e in af_int)
    other_time = float(sum(e - s for s, e, c, r in zip(ws, we, wc, w_read) if c == "O" and r >= 0.5))

    # inside AF every beat is irregular, so "supraventricular extra beat" (premature) has no meaning there
    in_af = np.zeros(len(vt), bool)
    for a, b in af_int:
        in_af |= (vt >= a) & (vt < b)
    s_in_af = int(((vl == "S") & in_af).sum())
    vl[(vl == "S") & in_af] = "N"
    af_ectopy = float((vl[in_af] == "V").mean()) if in_af.any() else 0.0     # V share inside AF (false-alarm warning)
    counts = {c: int((vl == c).sum()) for c in ("N", "S", "V")}
    counts["U"] = int(np.isin(vl, ["F", "Q"]).sum())          # fusion / unclassifiable: the beat model misses fusion beats
    n_beats = int(valid.sum())

    # RR between consecutive readable beats with no unreadable strip in between
    rr_t, rr = np.zeros(0), np.zeros(0)
    if len(vt) > 1:
        a, b = vt[:-1], vt[1:]
        same = np.array([ok[strip_of(x):strip_of(y) + 1].all() for x, y in zip(a, b)])
        rr_all = b - a
        keep = same & (rr_all > 0.25) & (rr_all < 6.0)
        rr_t, rr = b[keep], rr_all[keep]
    hr_ok = (rr >= 0.25) & (rr <= 3.0)
    hr_mean = float(60 / rr[hr_ok].mean()) if hr_ok.any() else float("nan")

    # per-minute heart rate (minutes with >= 40 s readable)
    n_min = int(np.ceil(dur / 60))
    minute = np.arange(n_min)
    read_min = np.array([ok[m * 6:(m + 1) * 6].sum() * STRIP for m in minute])
    hr_min_series = np.full(n_min, np.nan)
    mi = (rr_t // 60).astype(int)
    for m in range(n_min):
        sel = (mi == m) & hr_ok
        if read_min[m] >= MIN_READABLE_MIN and sel.sum() >= 10:
            hr_min_series[m] = 60 / rr[sel].mean()
    fin = np.isfinite(hr_min_series)
    hr_lo = (float(np.nanmin(hr_min_series)), int(np.nanargmin(hr_min_series))) if fin.any() else (float("nan"), 0)
    hr_hi = (float(np.nanmax(hr_min_series)), int(np.nanargmax(hr_min_series))) if fin.any() else (float("nan"), 0)
    brady = runs_of(hr_min_series < BRADY)
    tachy = runs_of(hr_min_series > TACHY)
    rate_episodes = dict(brady=_join(brady), tachy=_join(tachy))

    # pauses
    pz = np.where(rr >= PAUSE_S)[0]
    pauses = [dict(t=float(rr_t[i] - rr[i]), dur=float(rr[i])) for i in pz]

    # ectopy patterns along the readable beat sequence (sequences break at unreadable gaps)
    seq_break = np.r_[True, ~np.isin(vt[1:], rr_t)] if len(vt) else np.zeros(0, bool)
    seg_id = np.cumsum(seq_break)
    patterns = dict(V_couplets=0, V_runs=[], S_couplets=0, S_runs=[], bigeminy=0, trigeminy=0)
    for s in np.unique(seg_id):
        idx = np.where(seg_id == s)[0]
        lab = vl[idx]
        for c in ("V", "S"):
            for a, b in runs_of(lab == c):
                n = b - a + 1
                if n == 2:
                    patterns[f"{c}_couplets"] += 1
                elif n >= 3:
                    t0, t1 = vt[idx[a]], vt[idx[b]]
                    rate = 60 * (n - 1) / max(t1 - t0, 1e-6)
                    patterns[f"{c}_runs"].append(dict(t=float(t0), beats=int(n), rate=float(rate)))
        txt = "".join("V" if x == "V" else "n" for x in lab)
        patterns["bigeminy"] += _count_pattern(txt, "nV", 3)
        patterns["trigeminy"] += _count_pattern(txt, "nnV", 3)

    # morphology: share of readable strips flagged (limb placement only)
    p5, p23 = res["strip_p5"], res["strip_p23"]
    groups = []
    run = res.get("morph_run", True)
    for j, c in enumerate(["NORM", "MI", "STTC", "CD", "HYP"]):
        f = (p5[ok, j] >= res["thr5"][c]) if run and ok.any() else np.zeros(0, bool)
        share = float(f.mean()) if len(f) else 0.0
        groups.append(dict(code=c, name=GROUP[c], share=share, level=_level(share) if c != "NORM" else "",
                           best=int(np.where(ok)[0][np.argmax(p5[ok, j])]) if ok.any() else 0))
    named = []
    for j, s in enumerate(res["names23"]):
        r = res["reliability"].get(s)
        if not run or s == "NORM" or s not in res["thr23"] or not r or r["tier"] == "low":
            continue
        f = (p23[ok, j] >= res["thr23"][s]) if ok.any() else np.zeros(0, bool)
        share = float(f.mean()) if len(f) else 0.0
        if share >= INTERMITTENT:
            named.append(dict(code=s, name=SUBNAME.get(s, s), share=share, level=_level(share), tier=r["tier"],
                              auc=r["auc"], best=int(np.where(ok)[0][np.argmax(p23[ok, j])])))
    named.sort(key=lambda d: -d["share"])

    # unreadable periods
    bad = [(int(a) * STRIP, int(b + 1) * STRIP) for a, b in runs_of(~ok)]

    # period table: hours for recordings longer than 3 h, 10-min blocks otherwise
    block = 3600 if dur > 3 * 3600 else 600
    edges = list(np.arange(0, dur, block)) + [dur]
    if len(edges) > 2 and edges[-1] - edges[-2] < 60:          # a last block under 1 min joins the one before
        edges.pop(-2)
    hours = []
    for a, b in zip(edges[:-1], edges[1:]):
        a = int(a)
        hm = hr_min_series[a // 60:int(np.ceil(b / 60))]
        sel = (vt >= a) & (vt < b)
        fin_h = np.isfinite(hm).any()
        hours.append(dict(start=clock(res, a), readable_min=float(ok[a // STRIP:int(np.ceil(b / STRIP))].sum() * STRIP / 60),
                          hr_min=float(np.nanmin(hm)) if fin_h else np.nan,
                          hr_mean=float(np.nanmean(hm)) if fin_h else np.nan,
                          hr_max=float(np.nanmax(hm)) if fin_h else np.nan,
                          beats=int(sel.sum()), V=int((vl[sel] == "V").sum()), S=int((vl[sel] == "S").sum()),
                          af_min=sum(max(0.0, min(e, b) - max(s_, a)) for s_, e in af_int) / 60,
                          pauses=sum(1 for p in pauses if a <= p["t"] < b)))

    s = dict(dur=dur, readable_s=readable_s, readable_share=readable_s / max(dur, 1e-9), n_beats=n_beats, counts=counts,
             hr_mean=hr_mean, hr_min=hr_lo, hr_max=hr_hi, hr_minute=hr_min_series, brady=brady, tachy=tachy, rate_episodes=rate_episodes,
             pauses=pauses, patterns=patterns, af_intervals=af_int, af_time=af_time,
             af_burden=af_time / max(readable_s, 1e-9), af_ectopy=af_ectopy, s_in_af=s_in_af, n_ok=int(ok.sum()), other_time=other_time, morph_run=run,
             groups=groups, named=named,
             unreadable=bad, hours=pd.DataFrame(hours), block=block, beat_t=vt, beat_label=vl, beat_idx=np.where(valid)[0], rr_t=rr_t, rr=rr)
    s["events"] = select_events(res, s)
    s["findings"] = findings_text(res, s)
    return s


def _join(runs, gap=2, min_len=2):
    """Minute runs joined across gaps of up to `gap` minutes; only episodes of `min_len` minutes or more are kept."""
    out = []
    for a, b in runs:
        if out and a - out[-1][1] - 1 <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def _count_pattern(txt, unit, min_rep):
    """Number of episodes where `unit` repeats at least min_rep times in a row."""
    n, i, L = 0, 0, len(unit)
    while i <= len(txt) - L * min_rep:
        k = 0
        while txt[i + k * L:i + (k + 1) * L] == unit:
            k += 1
        if k >= min_rep:
            n += 1
            i += k * L
        else:
            i += 1
    return n


def _readable_fraction(ok, a, b):
    i0, i1 = int(a // STRIP), int(np.ceil(b / STRIP))
    sl = ok[i0:i1]
    return float(sl.mean()) if len(sl) else 0.0


def _merge(iv, gap=1.0):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _level(share):
    return "persistent" if share >= PERSISTENT else ("intermittent" if share >= INTERMITTENT else "")


def select_events(res, s, L=7.5):
    """Event strips for the report: (kind, title, start s). Each strip is 7.5 s and must be readable."""
    dur, ok = res["dur"], res["strip_ok"]
    ev = []

    def add(kind, title, t_center):
        t0 = float(np.clip(t_center - L / 2, 0, max(dur - L, 0)))
        if _readable_fraction(ok, t0, t0 + L) < 0.99:
            # move to the nearest fully readable 7.5-s span within +-20 s
            for d in np.arange(0.5, 20.5, 0.5):
                for t in (t0 - d, t0 + d):
                    if 0 <= t <= dur - L and _readable_fraction(ok, t, t + L) >= 0.99:
                        t0 = float(t)
                        break
                else:
                    continue
                break
        if not any(abs(t0 - e[2]) < 2 for e in ev):
            ev.append((kind, title, t0))

    if np.isfinite(s["hr_max"][0]):
        add("hr_max", f"Maximum heart rate {s['hr_max'][0]:.0f} bpm (1-min mean)", s["hr_max"][1] * 60 + 30)
    if np.isfinite(s["hr_min"][0]):
        add("hr_min", f"Minimum heart rate {s['hr_min'][0]:.0f} bpm (1-min mean)", s["hr_min"][1] * 60 + 30)
    if s["pauses"]:
        p = max(s["pauses"], key=lambda p: p["dur"])
        add("pause", f"Longest pause {p['dur']:.1f} s", p["t"] + p["dur"] / 2)
    if s["af_intervals"]:
        a, b = max(s["af_intervals"], key=lambda iv: iv[1] - iv[0])
        ws, pa = res["win_start"], res["win_prob"][:, list(res["win_class_names"]).index("A")]
        inside = np.where((ws >= a) & (ws + 30 <= b + 1e-6))[0]
        tc = ws[inside[np.argmax(pa[inside])]] + 15 if len(inside) else (a + b) / 2
        add("af", f"Atrial fibrillation, longest episode ({(b - a) / 60:.1f} min from {clock(res, a)})", tc)
    for c, name in (("V", "ventricular"), ("S", "supraventricular")):
        r = s["patterns"][f"{c}_runs"]
        if r:
            x = max(r, key=lambda d: d["beats"])
            add(f"{c}_run", f"Longest {name} run: {x['beats']} beats at {x['rate']:.0f} bpm", x["t"] + 1)
    bt, bl, bp = s["beat_t"], s["beat_label"], res["beat_prob"][s["beat_idx"]]
    for c, name, j in (("V", "Ventricular extra beat (V)", 2), ("S", "Supraventricular extra beat (S)", 1)):
        sel = np.where(bl == c)[0]
        sel = [i for i in sel if _readable_fraction(ok, bt[i] - 3.75, bt[i] + 3.75) >= 0.99]
        if sel:
            i = max(sel, key=lambda i: bp[i, j])
            add(c, f"{name}, most typical example", bt[i])
    for g in s["groups"][1:]:
        if g["level"] == "persistent":
            add("morph", f"Morphology alert: {g['name']} ({100 * g['share']:.0f}% of readable strips)", g["best"] * STRIP + 5)
    for n in s["named"][:3]:
        add("named", f"Named finding: {n['name']} ({100 * n['share']:.0f}% of readable strips)", n["best"] * STRIP + 5)
    if not ev and dur >= L:
        add("sample", "Representative strip", min(60.0, dur / 2))
    return sorted(ev, key=lambda e: e[2])


def share_text(share, n_ok):
    """'in 35% of readable strips', or 'in 1 of 3 readable 10-s strips' for short recordings."""
    if n_ok < 6:
        return f"in {round(share * n_ok)} of {n_ok} readable 10-s strip{'s' if n_ok > 1 else ''}"
    return f"in {100 * share:.0f}% of readable strips"


def findings_text(res, s):
    """Plain-language automatic findings, most important first. Never a diagnosis: points to clinical review."""
    out = []
    if s["af_intervals"]:
        longest = max(b - a for a, b in s["af_intervals"])
        out.append(f"Atrial fibrillation detected: {100 * s['af_burden']:.1f}% of the readable time, "
                   f"{len(s['af_intervals'])} episode(s), longest {longest / 60:.1f} min.")
        if s["af_ectopy"] >= ECTOPY_CAVEAT:
            out.append(f"Caution: {100 * s['af_ectopy']:.0f}% of the beats inside the AF episodes are ventricular extra "
                       "beats. Frequent extra beats can be mistaken for AF by the rhythm model; check the AF strips.")
    else:
        out.append("No atrial fibrillation detected.")
    c, n = s["counts"], max(s["n_beats"], 1)
    hrs = s["readable_s"] / 3600
    vr = s["patterns"]["V_runs"]
    v_txt = (f"Ventricular extra beats: {c['V']:,} ({100 * c['V'] / n:.1f}% of beats, {c['V'] / max(hrs, 1e-9):.0f} per hour)"
             + (f", {s['patterns']['V_couplets']} couplet(s)" if s["patterns"]["V_couplets"] else "")
             + (f", {len(vr)} run(s) of 3 or more beats (longest {max(r['beats'] for r in vr)} beats)" if vr else "") + ".")
    out.append(v_txt if c["V"] else "No ventricular extra beats detected.")
    if c["S"]:
        out.append(f"Supraventricular extra beats: {c['S']:,} ({100 * c['S'] / n:.1f}% of beats). Low reliability: the "
                   "beat model detects these poorly (see Methods)"
                   + (". Beats during AF are not counted." if s["s_in_af"] else "."))
    if s["pauses"]:
        p = max(s["pauses"], key=lambda p: p["dur"])
        out.append(f"Pauses of {PAUSE_S:.1f} s or more: {len(s['pauses'])} (longest {p['dur']:.1f} s at {clock(res, p['t'])}).")
    if np.isfinite(s["hr_mean"]) and not np.isfinite(s["hr_min"][0]):
        out.append(f"Heart rate: mean {s['hr_mean']:.0f} bpm (too short for a minute-by-minute trend).")
    elif np.isfinite(s["hr_mean"]):
        out.append(f"Heart rate: mean {s['hr_mean']:.0f} bpm, minimum {s['hr_min'][0]:.0f} bpm at "
                   f"{clock(res, s['hr_min'][1] * 60)}, maximum {s['hr_max'][0]:.0f} bpm at {clock(res, s['hr_max'][1] * 60)} "
                   "(1-min means).")
    if c["U"]:
        out.append(f"Unclassified beats: {c['U']:,} (fusion or unclear shape; not counted as extra beats).")
    if c["U"] / n >= UNCLASSIFIED_CAVEAT:
        out.append(f"Caution: {100 * c['U'] / n:.0f}% of the beats could not be classified. Beat shapes in this "
                   "recording differ from the training data, so the extra-beat counts may be unreliable.")
    if not s["morph_run"]:
        out.append("ECG shape screening: not performed. Its models were trained on resting ECGs with limb electrodes "
                   "and are not valid for chest (Holter) electrode placement.")
    else:
        alerts = [g for g in s["groups"][1:] if g["level"]]
        for g in alerts:
            lvl = f" ({g['level']})" if s["n_ok"] >= 6 else ""
            out.append(f"ECG shape screening: possible {PHRASE[g['code']]} {share_text(g['share'], s['n_ok'])}{lvl}; "
                       "confirm with a 12-lead ECG.")
        for n_ in s["named"][:3]:
            out.append(f"Named finding: {n_['name']} {share_text(n_['share'], s['n_ok'])} "
                       f"({n_['tier']} single-lead reliability, AUC {n_['auc']:.2f}); confirm with a 12-lead ECG.")
        if not alerts:
            out.append("ECG shape screening: no persistent or intermittent alert (a single lead does not rule out all "
                       "structural findings).")
    if s["readable_share"] >= 0.995:
        out.append("Signal quality: the whole recording was readable.")
    else:
        out.append(f"Signal quality: {100 * s['readable_share']:.0f}% of the recording was readable; the rest "
                   f"({(s['dur'] - s['readable_s']) / 60:.1f} min) was left out of every count.")
    return out

"""Single-Lead ECG Analyzer: a Holter-style analysis of one ECG lead (up to 24 h), built on three projects:
heartbeats (Project 1), atrial fibrillation (Project 2), signal quality and ECG shape (Project 3).
Gives a short on-screen summary and a professional PDF report. Research tool, not a medical device."""
import datetime as dt
import inspect
import json
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from holter import io as hio  # noqa: E402
from holter import validation, vendor  # noqa: E402
from holter.derive import GROUP, SUBNAME, clock, summarize  # noqa: E402
from holter.engine import MAX_HOURS, MIN_SECONDS, Analyzer  # noqa: E402
from holter.report import RHYTHM_COLOR, RHYTHM_NAME, build_pdf, event_log  # noqa: E402

GRID_MAJOR, GRID_MINOR, PAPER = "rgba(220,60,60,0.55)", "rgba(220,60,60,0.18)", "#fff6f6"
FONT = dict(family="Inter, sans-serif", size=12)
BEAT_COLOR = {"N": "#7a8494", "S": "#1f5fbf", "V": "#c0392b", "U": "#7b4fa0"}
C_OK, C_BAD, C_WARN, C_GREY = "#2e9e5b", "#dc2626", "#d68910", "#9ca3af"
BASE = dt.datetime(2000, 1, 1)               # time axis origin when the clock time is unknown
PLACEMENT = {"Chest electrodes (Holter, patch, AD8232 on the chest)": "chest",
             "Limb electrodes (wrists and ankle, resting lead II)": "limb"}
EXAMPLES = {"mit202": "MIT-BIH 202 · 30 min · paroxysmal atrial fibrillation",
            "mit210": "MIT-BIH 210 · 30 min · atrial fibrillation throughout",
            "mit100": "MIT-BIH 100 · 30 min · normal rhythm",
            "mit200": "MIT-BIH 200 · 30 min · many ventricular extra beats (known AF false alarm)",
            "afdb04043": "AF Database 04043 · 1 h · repeated AF episodes"}

ICON = {
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "pause": '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
    "shape": '<path d="M2 12h3l3-8 4 16 3-8h7"/>',
    "signal": '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>',
}
CSS = """
.title {text-align:center; font-size:2.1rem; font-weight:800; margin:8px 0 2px; letter-spacing:-0.02em;}
.subtitle {text-align:center; opacity:.75; margin:0 0 14px;}
.sec {font-weight:700; font-size:1.05rem; margin:20px 0 8px; padding-bottom:6px;
      border-bottom:2px solid var(--border-color-primary);}
.step {font-weight:700; font-size:1rem; margin:4px 0 2px;}
.kpi-row {display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 12px;}
.kpi {flex:1 1 180px; display:flex; gap:12px; align-items:center; border:1px solid var(--border-color-primary);
      border-radius:14px; padding:12px 14px; background:var(--background-fill-secondary);}
.kpi .ic {flex:0 0 42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
          background:rgba(46,111,158,.10); color:#2e6f9e;}
.kpi.ok .ic {background:rgba(46,158,91,.12); color:#2e9e5b;}
.kpi.warn .ic {background:rgba(243,156,18,.15); color:#d68910;}
.kpi.bad .ic {background:rgba(220,38,38,.12); color:#dc2626;}
.kpi.off .ic {background:rgba(156,163,175,.15); color:#9ca3af;}
.kpi .ic svg {width:22px; height:22px; stroke:currentColor; fill:none; stroke-width:2;
              stroke-linecap:round; stroke-linejoin:round;}
.kpi .v {font-size:1.35rem; font-weight:700; line-height:1.15;}
.kpi .l {font-size:.8rem; opacity:.8;}
.kpi .s {font-size:.72rem; opacity:.6;}
.find {border:1px solid var(--border-color-primary); border-radius:14px; padding:10px 16px; margin:6px 0 12px;
       background:var(--background-fill-secondary);}
.find li {margin:4px 0;}
.find li.caution {color:#b45309;}
.empty {border:1px dashed var(--border-color-primary); border-radius:14px; padding:22px; text-align:center; opacity:.75;}
.req {display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 12px;}
.req span {padding:4px 10px; border-radius:999px; font-size:.82rem;
           border:1px solid var(--border-color-primary); background:var(--background-fill-secondary);}
.pipe {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0 16px;}
.chip {padding:6px 12px; border-radius:999px; border:1px solid var(--border-color-primary);
       background:var(--background-fill-secondary); font-size:.9rem;}
.arrow {opacity:.5;}
"""
REQ_HTML = ('<div class="req"><span>One ECG lead</span><span>CSV (millivolts) or PhysioNet .hea + .dat</span>'
            f'<span>{MIN_SECONDS} s to {MAX_HOURS} h</span><span>Any sampling rate from 100 Hz</span>'
            '<span>Research tool · not a medical device</span></div>')
PIPE_HTML = ('<div class="pipe">' + '<span class="arrow">→</span>'.join(
    f'<span class="chip">{s}</span>' for s in
    ["Raw ECG", "Polarity and units check", "Quality gate per 10-s strip (Project 3)",
     "R peaks + every beat N / S / V (Project 1)", "Rhythm per 30-s window: AF or not (Project 2)",
     "ECG shape per strip, limb electrodes only (Project 3)", "Holter measures", "PDF report"]) + "</div>")


def sec(t):
    return f'<div class="sec">{t}</div>'


def cards_html(items):
    out = []
    for icon, label, value, sub, status in items:
        out.append(f'<div class="kpi {status}"><div class="ic"><svg viewBox="0 0 24 24">{ICON[icon]}</svg></div>'
                   f'<div><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div></div>')
    return '<div class="kpi-row">' + "".join(out) + "</div>"


def hm(sec_):
    sec_ = int(round(sec_))
    return f"{sec_ // 3600} h {sec_ % 3600 // 60:02d} min" if sec_ >= 3600 else f"{sec_ / 60:.1f} min"


def t_axis(res, t):
    """Plot x values: real clock time when known, otherwise time from the start shown as h:mm."""
    t = np.asarray(t, float)
    base = res["start_time"] or BASE
    return pd.to_datetime(base) + pd.to_timedelta(t, unit="s")


# ---------------------------------------------------------------- figures
def timeline_fig(res, s):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.06)
    hr = s["hr_minute"]
    x = t_axis(res, np.arange(len(hr)) * 60 + 30)
    fig.add_trace(go.Scatter(x=x, y=hr, mode="lines", line=dict(color="#1b2a41", width=1.6), name="heart rate",
                             hovertemplate="%{y:.0f} bpm<extra></extra>"), 1, 1)
    for a, b in s["af_intervals"]:
        fig.add_vrect(x0=t_axis(res, [a])[0], x1=t_axis(res, [b])[0], fillcolor="rgba(200,65,75,0.14)", line_width=0,
                      row=1, col=1)
    classes = list(RHYTHM_COLOR)
    if len(res["win_start"]):
        z = [[classes.index(c) for c in res["win_class"]]]
        scale = []
        for i, c in enumerate(classes):
            scale += [[i / len(classes), RHYTHM_COLOR[c]], [(i + 1) / len(classes), RHYTHM_COLOR[c]]]
        mids = (res["win_start"] + res["win_end"]) / 2
        fig.add_trace(go.Heatmap(x=t_axis(res, mids), y=["rhythm"], z=z, zmin=-0.5, zmax=len(classes) - 0.5,
                                 colorscale=scale, showscale=False, hoverinfo="skip"), 2, 1)
    for a, b in s["unreadable"]:
        fig.add_vrect(x0=t_axis(res, [a])[0], x1=t_axis(res, [min(b, s["dur"])])[0], fillcolor="rgba(120,120,120,0.35)",
                      line_width=0, row=2, col=1)
    for c in classes:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=RHYTHM_NAME[c],
                                 marker=dict(size=10, color=RHYTHM_COLOR[c], symbol="square")), 1, 1)
    fig.update_yaxes(title_text="bpm", row=1, col=1, gridcolor="#e5e7eb")
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(tickformat="%H:%M", row=2, col=1,
                     title_text="clock time" if res["start_time"] else "time from the start (h:mm)")
    fig.update_layout(template="plotly_white", height=290, margin=dict(l=55, r=20, t=30, b=45), font=FONT,
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def ecg_fig(res, s, t0, L=10.0, title=""):
    fs = res["signal_fs"]
    t0 = float(np.clip(t0, 0, max(res["dur"] - L, 0)))
    i0, i1 = int(t0 * fs), int(min(len(res["signal"]), (t0 + L) * fs))
    v = np.asarray(res["signal"][i0:i1], float)
    v = v - np.median(v) if len(v) else v
    t = t0 + np.arange(len(v)) / fs
    hi = float(np.clip(np.ceil(np.percentile(v, 99.8) / 0.5) * 0.5 + 0.5, 1.5, 4.0)) if len(v) else 1.5
    lo = float(np.clip(np.floor(np.percentile(v, 0.2) / 0.5) * 0.5 - 0.5, -3.0, -1.0)) if len(v) else -1.0
    fig = go.Figure(go.Scatter(x=t, y=v, mode="lines", line=dict(color="#111", width=1.2), hoverinfo="skip"))
    sel = (s["beat_t"] >= t0) & (s["beat_t"] < t0 + L)
    bt, bl = s["beat_t"][sel], s["beat_label"][sel]
    bl = np.where(np.isin(bl, ["F", "Q"]), "U", bl)
    for tb, lab in zip(bt, bl):
        fig.add_annotation(x=tb, y=hi - 0.12, text=f"<b>{lab}</b>" if lab != "N" else lab, showarrow=False,
                           font=dict(size=13 if lab != "N" else 10, color=BEAT_COLOR[lab]))
    for k in range(1, len(bt)):
        fig.add_annotation(x=(bt[k] + bt[k - 1]) / 2, y=lo + 0.15, text=f"{1000 * (bt[k] - bt[k - 1]):.0f}",
                           showarrow=False, font=dict(size=9, color="#6b7280"))
    ok = res["strip_ok"]
    for k in range(int(t0 // 10), int(np.ceil((t0 + L) / 10))):
        if k < len(ok) and not ok[k]:
            fig.add_vrect(x0=max(k * 10, t0), x1=min((k + 1) * 10, t0 + L), fillcolor="rgba(120,120,120,0.25)",
                          line_width=0, annotation_text="unreadable", annotation_position="top left")
    xt = np.arange(np.floor(t0 / 0.2) * 0.2, t0 + L + 1e-9, 0.2)
    yt = np.arange(lo, hi + 1e-9, 0.5)
    fig.update_xaxes(range=[t0, t0 + L], tickmode="array", tickvals=xt,
                     ticktext=[clock(res, v_) if abs(v_ - round(v_)) < 1e-6 and round(v_) % 2 == 0 else "" for v_ in xt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.04, showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="grid: 0.2 s × 0.5 mV (small squares 0.04 s × 0.1 mV) · letters: beat class · numbers: RR in ms")
    fig.update_yaxes(range=[lo, hi], tickmode="array", tickvals=yt,
                     ticktext=[f"{v_:.0f}" if abs(v_ - round(v_)) < 1e-6 else "" for v_ in yt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.1, showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="mV")
    fig.update_layout(plot_bgcolor=PAPER, paper_bgcolor="white", height=340, margin=dict(l=55, r=20, t=50, b=50),
                      font=FONT, title=dict(text=title or f"{clock(res, t0)} - {clock(res, t0 + L)}", font=dict(size=14)),
                      showlegend=False)
    return fig


def rr_fig(res, s):
    fig = go.Figure(go.Scattergl(x=t_axis(res, s["rr_t"]), y=s["rr"], mode="markers",
                                 marker=dict(size=3, color="#1b2a41", opacity=0.6), hoverinfo="skip"))
    for a, b in s["af_intervals"]:
        fig.add_vrect(x0=t_axis(res, [a])[0], x1=t_axis(res, [b])[0], fillcolor="rgba(200,65,75,0.14)", line_width=0)
    fig.add_hline(y=2.0, line=dict(color=C_BAD, dash="dash", width=1), annotation_text="pause limit 2 s")
    fig.update_yaxes(title_text="RR interval (s)", range=[0.2, max(2.4, float(np.max(s["rr"])) + 0.2) if len(s["rr"]) else 2.4])
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(template="plotly_white", height=280, margin=dict(l=55, r=20, t=40, b=40), font=FONT,
                      title=dict(text="RR intervals, one dot per beat (irregular cloud = atrial fibrillation)", font=dict(size=13)))
    return fig


def quality_fig(res):
    est = np.clip(res["strip_est"], -15, 35)
    x = t_axis(res, np.arange(len(est)) * 10 + 5)
    col = np.where(res["strip_ok"], C_OK, C_GREY)
    fig = go.Figure(go.Scattergl(x=x, y=est, mode="markers", marker=dict(size=4, color=col), hoverinfo="skip"))
    fig.add_hline(y=res["gate_threshold"], line=dict(color="#111", dash="dash", width=1),
                  annotation_text=f"gate {res['gate_threshold']:.1f} dB")
    fig.update_yaxes(title_text="estimated SNR (dB)")
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=55, r=20, t=40, b=40), font=FONT,
                      title=dict(text="Signal quality per 10-s strip (grey = unreadable, left out of every count)",
                                 font=dict(size=13)))
    return fig


# ---------------------------------------------------------------- texts and tables
def basic_cards(res, s):
    c, n = s["counts"], max(s["n_beats"], 1)
    hrs = max(s["readable_s"] / 3600, 1e-9)
    af = bool(s["af_intervals"])
    hr_ok = np.isfinite(s["hr_mean"])
    rng = (f"min {s['hr_min'][0]:.0f} · max {s['hr_max'][0]:.0f} (1-min means)" if np.isfinite(s["hr_min"][0])
           else "too short for a minute trend")
    if not s["morph_run"]:
        shape = ("shape", "ECG shape", "Not checked", "needs limb electrodes (see How it works)", "off")
    else:
        alerts = [GROUP[g["code"]] for g in s["groups"][1:] if g["level"]]
        shape = ("shape", "ECG shape", f"{len(alerts)} alert(s)" if alerts else "No alert",
                 ", ".join(alerts) if alerts else "a single lead does not rule everything out", "bad" if alerts else "ok")
    items = [
        ("clock", "Analyzed", hm(s["readable_s"]), f"of {hm(s['dur'])} · "
         f"{100 * s['readable_share']:.0f}% readable" if s["readable_share"] in (0.0, 1.0) else
         f"of {hm(s['dur'])} · {min(99.9, 100 * s['readable_share']):.1f}% readable",
         "ok" if s["readable_share"] >= 0.8 else "warn"),
        ("heart", "Average heart rate", f"{s['hr_mean']:.0f} bpm" if hr_ok else "-", rng,
         "ok" if hr_ok and 50 <= s["hr_mean"] <= 100 else "warn"),
        ("pulse", "Atrial fibrillation (share of time)", f"{100 * s['af_burden']:.1f}%" if af else "Not detected",
         f"{len(s['af_intervals'])} episode(s), longest {max(b - a for a, b in s['af_intervals']) / 60:.1f} min" if af
         else "30-s windows checked", "bad" if af else "ok"),
        ("zap", "Ventricular extra beats per hour", f"{c['V'] / hrs:.0f}", f"{c['V']:,} in total ({100 * c['V'] / n:.1f}%) · "
         f"S {c['S']:,} (low reliability)", "warn" if c["V"] / hrs >= 30 or s["patterns"]["V_runs"] else "ok"),
        ("pause", "Pauses of 2 s or more", f"{len(s['pauses'])}",
         f"longest {max(p['dur'] for p in s['pauses']):.1f} s" if s["pauses"] else "none found",
         "bad" if s["pauses"] else "ok"),
        shape,
    ]
    return cards_html(items)


def findings_html(res, s):
    li = "".join(f'<li class="{"caution" if f.startswith("Caution") else ""}">{f}</li>' for f in s["findings"])
    notes = (f"<p><small><b>Signal notes:</b> {'; '.join(res['notes'])}.</small></p>" if res["notes"] else "")
    return f'<div class="find"><b>What the analysis found</b><ul>{li}</ul>{notes}</div>'


def events(res, s):
    """(label, start time of the 7.5-s strip) for the event picker."""
    return [(f"{clock(res, t0)} · {title}", float(t0)) for _, title, t0 in s["events"]]


def details_md(res, s):
    c = s["counts"]
    pat = s["patterns"]
    rows = [("Duration", hm(s["dur"])), ("Readable", f"{hm(s['readable_s'])} ({100 * s['readable_share']:.1f}%)"),
            ("Sampling rate", f"{res['fs']:g} Hz"), ("Electrodes", res["placement"]),
            ("Polarity", "flipped for beats and rhythm" if res["inverted"] else "as recorded"),
            ("Beats (readable)", f"{s['n_beats']:,}: N {c['N']:,} · S {c['S']:,} · V {c['V']:,} · unclassified {c['U']:,}"),
            ("V patterns", f"couplets {pat['V_couplets']} · runs {len(pat['V_runs'])} · bigeminy {pat['bigeminy']} · "
                           f"trigeminy {pat['trigeminy']}"),
            ("S patterns", f"couplets {pat['S_couplets']} · runs {len(pat['S_runs'])}"),
            ("AF", f"burden {100 * s['af_burden']:.1f}% · episodes {len(s['af_intervals'])} · "
                   f"V share inside AF {100 * s['af_ectopy']:.0f}%"),
            ("Other rhythm (not AF)", hm(s["other_time"])), ("Analysis time", f"{res['seconds']:.0f} s")]
    return "| Item | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def shape_tables(res, s):
    if not s["morph_run"]:
        return (pd.DataFrame({"ECG shape screening": ["Not performed: chest electrodes (see How it works)."]}),
                pd.DataFrame({"Named findings": ["Not performed: chest electrodes."]}))
    g = pd.DataFrame([{"group": GROUP[x["code"]], "readable strips flagged %": round(100 * x["share"], 1),
                       "level": x["level"] or ("-" if x["code"] != "NORM" else ""),
                       "threshold %": round(100 * res["thr5"][x["code"]], 1)} for x in s["groups"]])
    n = pd.DataFrame([{"finding": x["name"], "readable strips flagged %": round(100 * x["share"], 1),
                       "reliability": x["tier"], "lead II AUC": round(x["auc"], 3)} for x in s["named"]])
    if n.empty:
        n = pd.DataFrame({"Named findings": ["None in 10% or more of the readable strips."]})
    return g, n


def export_files(res, s, folder):
    bt, bl, bp = s["beat_t"], s["beat_label"], res["beat_prob"][s["beat_idx"]]
    beats = pd.DataFrame({"time_s": np.round(bt, 3), "clock": [clock(res, t) for t in bt], "class": bl})
    for j, c in enumerate("NSVFQ"):
        beats[f"p_{c}"] = np.round(bp[:, j], 4) if len(bp) else []
    win = pd.DataFrame({"start_s": res["win_start"], "end_s": res["win_end"], "class": res["win_class"]})
    for j, c in enumerate(res["win_class_names"]):
        win[f"p_{c}"] = np.round(res["win_prob"][:, j], 4)
    strips = pd.DataFrame({"start_s": np.arange(len(res["strip_ok"])) * 10, "estimated_snr_db": np.round(res["strip_est"], 2),
                           "readable": res["strip_ok"]})
    if s["morph_run"]:
        for j, c in enumerate(["NORM", "MI", "STTC", "CD", "HYP"]):
            strips[f"p_{c}"] = np.round(res["strip_p5"][:, j], 4)
    ev = pd.DataFrame(event_log(res, s), columns=["time", "event", "details"])
    out = []
    for name, df in (("beats", beats), ("rhythm_windows", win), ("strips_quality", strips), ("events", ev),
                     ("summary_per_period", s["hours"])):
        p = os.path.join(folder, f"{name}.csv")
        df.to_csv(p, index=False)
        out.append(p)
    return out


def about_md():
    v = validation.all_numbers()
    b, r, m = v["beats"], v["rhythm"], v["morphology"]
    links = " · ".join(f"[Project {i + 1}: {vendor.REPOS[role][0]}]({vendor.url(role)}) (commit `{vendor.REPOS[role][1][:7]}`)"
                       for i, role in enumerate(("beats", "rhythm", "morphology")))
    rows = [("V beats", f"MIT-BIH DS2, 22 unseen patients, {b['V']['n']:,} V beats", f"Se {b['V']['se']:.1f}% · PPV {b['V']['ppv']:.1f}%"),
            ("S beats", f"MIT-BIH DS2, {b['S']['n']:,} S beats", f"Se {b['S']['se']:.1f}% · PPV {b['S']['ppv']:.1f}% (low)"),
            ("Fusion beats", f"MIT-BIH DS2, {b['F']['n']:,} F beats", "not detected (shown as unclassified)")]
    for db in ("CinC 2017 test", "MIT-BIH DS2", "MIT-BIH AF DB"):
        q = r[db]
        rows.append(("Atrial fibrillation", f"{db}, {q['n']:,} {q['unit']}",
                     f"Se {q['se']:.1f}% · PPV {q['ppv']:.1f}% · Sp {q['sp']:.1f}%"))
    rows += [("ECG shape (limb lead II)", "PTB-XL test / Chapman-Shaoxing external",
              f"mean AUC {m['internal_macro']:.3f} / {m['external_macro']:.3f}"),
             ("Quality gate", "PTB-XL test + real electrode noise",
              f"catches {m['gate_sensitivity']} of strips below 0 dB · {m['gate_false_rejections']} false rejections")]
    table = "| Part | Test data (never used in training) | Result |\n|---|---|---|\n" + "\n".join(
        f"| {a} | {b_} | {c} |" for a, b_, c in rows)
    chest = os.path.join(ROOT, "results", "morphology_on_chest_leads.csv")
    chest_md = ""
    if os.path.exists(chest):
        t = pd.read_csv(chest)
        md = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n" + "\n".join(
            "| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False))
        chest_md = ("\n\n**The evidence.** The ECG-shape models, forced to run on chest-lead recordings (percentage of "
                    "readable strips flagged per group):\n\n" + md)
    fa = ", ".join(f"{rec} ({pct:.0f}%)" for rec, pct in r["false_alarm_records"])
    return f"""
### What it does
It reads one ECG lead, up to {MAX_HOURS} hours, the way a Holter analysis does: it checks the signal quality of every
10-s strip, finds and classifies every heartbeat, reads the rhythm every 15 seconds, and turns all of it into the
measures a cardiologist looks for (heart rate, extra beats and their patterns, pauses, atrial fibrillation and its
burden). Everything that falls in an unreadable strip is left out, so noise is never reported as a finding.

### Which recordings work?
| Recording | Heartbeats | Rhythm (AF) | ECG shape |
|---|---|---|---|
| Holter, patch or AD8232 with **chest** electrodes | ✅ tested (MIT-BIH) | ✅ tested (MIT-BIH, AF Database) | ❌ not performed (see below) |
| AD8232 or 12-lead with **limb** electrodes (lead II, at rest) | ⚠️ not measured | ⚠️ not measured | ✅ tested (PTB-XL, Chapman-Shaoxing) |
| Lead I from the fingers (handheld or smartwatch style) | ⚠️ not measured | ✅ tested (CinC 2017) | ❌ trained on lead II only |

✅ = accuracy measured on patients never used in training (numbers below). ⚠️ = should work, but not measured yet.

#### Why is the ECG shape not checked for chest electrodes?
The shape models (Project 3) learned from resting hospital
ECGs, where lead II is recorded with electrodes on the wrists and ankle. On Holter recordings the electrodes sit on the
chest, which changes the wave shapes. Run on the MIT-BIH Holter examples, the models called almost every strip
abnormal, even in record 100, which has a normal rhythm; so their output is not shown for chest recordings.{chest_md}

### Accuracy of the three engines
{table}

### Limitations
- A single lead does not replace a 12-lead ECG or a cardiologist: findings are prompts for review.
- S beats are detected poorly, and fusion beats not at all.
- Frequent extra beats can be mistaken for AF: in the external test, MIT-BIH records {fa} got false AF burden.
  The app warns when AF episodes contain many ventricular extra beats.
- Tested on public databases only; not yet on the AD8232 board or on new patients.

### Built from
{links}
"""


# ---------------------------------------------------------------- app
def build_demo():
    state = {"A": None}

    def model():
        if state["A"] is None:
            state["A"] = Analyzer()
        return state["A"]

    z = np.load(vendor.path("rhythm", "app", "examples", "examples.npz"))
    meta = {m["key"]: m for m in json.loads(str(z["meta"]))}
    ex_choices = [(lab, k) for k, lab in EXAMPLES.items() if k in meta]

    def load_input(source, example, files, channel, fs, start_txt):
        if source == "Example":
            m = meta[example]
            return z[example] / 1000.0, float(m["fs"]), None, False, f"{EXAMPLES[example].split(' · ')[0]} (example)"
        if not files:
            raise gr.Error("Upload a recording first (a .csv file, or a .hea file with its .dat file).")
        paths = files if isinstance(files, list) else [files]
        start, date_known = hio.parse_start(start_txt)
        if any(str(p).lower().endswith(".hea") for p in paths):
            rec = hio.wfdb_record(paths)
            x, f, st, dk = hio.read_wfdb(rec, channel)
            if start is None:
                start, date_known = st, dk
            return x, f, start, date_known, os.path.basename(rec) + f" ({channel})"
        csvs = [p for p in paths if str(p).lower().endswith((".csv", ".txt"))]
        if not csvs:
            raise gr.Error("Use a .csv file, or a .hea file together with its .dat file.")
        if not fs or fs < 100:
            raise gr.Error("Enter the sampling rate of the CSV file (100 Hz or more).")
        return hio.read_csv(csvs[0], channel), float(fs), start, date_known, os.path.basename(csvs[0]) + f" ({channel})"

    def analyze(source, example, files, channel, fs, start_txt, placement, name, age, sex, ind, progress=gr.Progress()):
        try:
            x, f, start, date_known, src = load_input(source, example, files, channel, fs, start_txt)
            if len(x) / f > MAX_HOURS * 3600 * 1.001:
                gr.Warning(f"Longer than {MAX_HOURS} h: only the first {MAX_HOURS} h are analyzed.")
            progress(0.0, desc="Loading the models")
            res = model().analyze(x, f, start_time=start, placement=PLACEMENT[placement],
                                  progress=lambda v, msg="": progress(0.9 * v, desc=msg))
            res["date_known"] = date_known
            progress(0.92, desc="Building the report")
            s = summarize(res)
        except gr.Error:
            raise
        except Exception as e:                                 # unreadable file, too short, etc.
            raise gr.Error(str(e))
        folder = tempfile.mkdtemp(prefix="holter_")
        pdf = make_pdf(res, s, src, folder, name, age, sex, ind)
        csvs = export_files(res, s, folder)
        st = dict(res=res, s=s, src=src, folder=folder)
        evs = events(res, s)
        ev_dd = gr.update(choices=[e[0] for e in evs], value=evs[0][0] if evs else None)
        first = ecg_fig(res, s, evs[0][1], 7.5, evs[0][0]) if evs else ecg_fig(res, s, 0)
        g, n = shape_tables(res, s)
        win = pd.DataFrame({"start": [clock(res, t) for t in res["win_start"]], "rhythm": [RHYTHM_NAME[c] for c in res["win_class"]]})
        for j, c in enumerate(res["win_class_names"]):
            win[f"P({RHYTHM_NAME[c]}) %"] = (100 * res["win_prob"][:, j].astype(float)).round(1)
        hours = s["hours"].rename(columns={"start": "from", "readable_min": "readable (min)", "hr_min": "HR min",
                                           "hr_mean": "HR mean", "hr_max": "HR max", "af_min": "AF (min)"}).round(1)
        progress(1.0, desc="Done")
        return (st, basic_cards(res, s), findings_html(res, s), timeline_fig(res, s), pdf, ev_dd, first,
                details_md(res, s), rr_fig(res, s), quality_fig(res), hours, win, g, n,
                ecg_fig(res, s, 0), 0, csvs)

    def make_pdf(res, s, src, folder, name, age, sex, ind):
        path = os.path.join(folder, "ECG_report.pdf")
        build_pdf(res, s, path, patient=dict(name=name, age=str(age or ""), sex=sex or "", indication=ind), source=src)
        return path

    def rebuild(st, name, age, sex, ind):
        if not st:
            raise gr.Error("Analyze a recording first.")
        gr.Info("PDF updated with the patient details.")
        return make_pdf(st["res"], st["s"], st["src"], st["folder"], name, age, sex, ind)

    def show_event(st, label):
        if not st or not label:
            return gr.update()
        t0 = dict(events(st["res"], st["s"]))[label]
        return ecg_fig(st["res"], st["s"], t0, 7.5, label)

    def view(st, t0, step):
        if not st:
            return gr.update(), t0
        t = float(np.clip((t0 or 0) + step, 0, max(st["res"]["dur"] - 10, 0)))
        return ecg_fig(st["res"], st["s"], t), t

    def on_files(files):
        if not files:
            return gr.update(choices=[], value=None), gr.update(interactive=True), gr.update()
        paths = files if isinstance(files, list) else [files]
        try:
            if any(str(p).lower().endswith(".hea") for p in paths):
                names, f = hio.wfdb_channels(hio.wfdb_record(paths))
                return (gr.update(choices=names, value=hio.default_column(names), label="Channel"),
                        gr.update(value=f, interactive=False), gr.update(placeholder="from the header, or type one"))
            csvs = [p for p in paths if str(p).lower().endswith((".csv", ".txt"))]
            names, _ = hio.csv_columns(csvs[0])
            return (gr.update(choices=names, value=hio.default_column(names), label="ECG column"),
                    gr.update(interactive=True), gr.update(placeholder="HH:MM (optional)"))
        except Exception as e:
            raise gr.Error(f"Could not read the file: {e}")

    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw, launch_kw = dict(theme=theme, css=CSS), {}
    else:
        blocks_kw, launch_kw = {}, dict(theme=theme, css=CSS)

    with gr.Blocks(title="Single-Lead ECG Analyzer", **blocks_kw) as demo:
        st = gr.State()
        gr.HTML('<div class="title">Single-Lead ECG Analyzer</div><div class="subtitle">A Holter-style analysis of one '
                'ECG lead: heartbeats, rhythm, pauses and signal quality, with a PDF report</div>')
        with gr.Group():
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.HTML('<div class="step">1 · Choose a recording</div>')
                    source = gr.Radio(["Example", "Upload my recording"], value="Example", show_label=False)
                    example = gr.Dropdown(ex_choices, value=ex_choices[0][1] if ex_choices else None,
                                          label="Example (public databases, never used for training the rhythm model)")
                    with gr.Column(visible=False) as up_col:
                        files = gr.File(label="CSV file, or PhysioNet .hea + .dat", file_count="multiple",
                                        file_types=[".csv", ".txt", ".hea", ".dat"], type="filepath")
                        with gr.Row():
                            channel = gr.Dropdown([], label="ECG column", allow_custom_value=True)
                            fs = gr.Number(250, label="Sampling rate (Hz)", precision=0)
                            start_txt = gr.Textbox(label="Start time (optional)", placeholder="HH:MM (optional)")
                with gr.Column(scale=2):
                    gr.HTML('<div class="step">2 · Electrodes</div>')
                    placement = gr.Radio(list(PLACEMENT), value=list(PLACEMENT)[0], show_label=False)
                    with gr.Accordion("Patient details for the PDF (optional, never stored)", open=False):
                        p_name = gr.Textbox(label="Name or ID")
                        with gr.Row():
                            p_age = gr.Textbox(label="Age")
                            p_sex = gr.Dropdown(["", "Female", "Male"], value="", label="Sex")
                        p_ind = gr.Textbox(label="Reason for the recording", placeholder="e.g. palpitations")
                        b_rebuild = gr.Button("Update the PDF with these details", size="sm")
                    gr.HTML('<div class="step">3 · Analyze</div>')
                    run_btn = gr.Button("Analyze recording", variant="primary", size="lg")
        with gr.Tabs():
            with gr.Tab("Basic"):
                b_cards = gr.HTML('<div class="empty">Choose an example or upload a recording, then press '
                                  '<b>Analyze recording</b>. A 30-minute recording takes about 10 to 30 seconds.</div>')
                b_find = gr.HTML()
                gr.HTML(sec("Heart rate and rhythm over the whole recording"))
                b_time = gr.Plot(show_label=False)
                b_pdf = gr.File(label="PDF report (Holter-style, A4)")
                gr.HTML(sec("Main events"))
                b_ev = gr.Dropdown([], label="Event", show_label=False)
                b_ecg = gr.Plot(show_label=False)
                gr.Markdown("<small>Research tool, not a medical device. Automated single-lead results must be reviewed "
                            "by a qualified clinician.</small>")
            with gr.Tab("Advanced"):
                gr.HTML(sec("Recording and counts"))
                a_details = gr.Markdown()
                gr.HTML(sec("Rhythm"))
                a_rr = gr.Plot(show_label=False)
                a_win = gr.Dataframe(interactive=False, max_height=260, label="Every 30-s window")
                gr.HTML(sec("Signal quality"))
                a_q = gr.Plot(show_label=False)
                gr.HTML(sec("Summary per period"))
                a_hours = gr.Dataframe(interactive=False, max_height=300)
                gr.HTML(sec("ECG shape (limb electrodes only)"))
                a_g = gr.Dataframe(interactive=False)
                a_n = gr.Dataframe(interactive=False, label="Named findings (high and moderate single-lead reliability)")
                gr.HTML(sec("ECG viewer"))
                a_ecg = gr.Plot(show_label=False)
                with gr.Row():
                    a_back = gr.Button("◀ 10 s", size="sm")
                    a_t = gr.Number(0, label="Start (s)", precision=1)
                    a_go = gr.Button("Show", size="sm")
                    a_fwd = gr.Button("10 s ▶", size="sm")
                gr.HTML(sec("Downloads"))
                a_csv = gr.File(label="Beats, rhythm windows, strip quality, events and period summary (CSV)",
                                file_count="multiple")
            with gr.Tab("How it works"):
                gr.HTML(REQ_HTML)
                gr.HTML(PIPE_HTML)
                gr.Markdown(about_md())
        gr.Markdown("<center><small>Fady Mostafa Omar · 2026 · Data: MIT-BIH Arrhythmia and AF databases, PTB-XL, "
                    "PhysioNet/CinC Challenge 2017, Chapman-Shaoxing (PhysioNet)</small></center>")

        source.change(lambda s_: (gr.update(visible=s_ == "Example"), gr.update(visible=s_ != "Example")),
                      source, [example, up_col])
        files.change(on_files, files, [channel, fs, start_txt])
        outs = [st, b_cards, b_find, b_time, b_pdf, b_ev, b_ecg, a_details, a_rr, a_q, a_hours, a_win, a_g, a_n,
                a_ecg, a_t, a_csv]
        run_btn.click(analyze, [source, example, files, channel, fs, start_txt, placement, p_name, p_age, p_sex, p_ind],
                      outs)
        b_rebuild.click(rebuild, [st, p_name, p_age, p_sex, p_ind], b_pdf)
        b_ev.change(show_event, [st, b_ev], b_ecg)
        a_back.click(lambda s_, t: view(s_, t, -10), [st, a_t], [a_ecg, a_t])
        a_fwd.click(lambda s_, t: view(s_, t, 10), [st, a_t], [a_ecg, a_t])
        a_go.click(lambda s_, t: view(s_, t, 0), [st, a_t], [a_ecg, a_t])
    return demo, launch_kw


def main():
    import argparse
    p = argparse.ArgumentParser(description="Single-Lead ECG Analyzer (Gradio app)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = p.parse_args()
    demo, launch_kw = build_demo()
    demo.queue().launch(share=a.share, server_port=a.port, **launch_kw)


if __name__ == "__main__":
    main()

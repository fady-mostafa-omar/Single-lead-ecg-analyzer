"""Holter-style PDF report (A4).
  Page 1   summary: patient and recording details, key measures, automatic findings, physician review box
  Page 2   trends: heart rate, RR intervals, rhythm and signal quality, extra beats per period
  Page 3   period table and event log
  Then     ECG strips of the main events, drawn to scale (25 mm/s, 10 mm/mV) on a 1-mm grid
  Last     methods, validation numbers of the three engines, limitations
Patient details are only written into the PDF; nothing is stored."""
import datetime as dt
import io
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import Paragraph, Table, TableStyle  # noqa: E402

from . import derive, validation, vendor  # noqa: E402

W, H = A4
MX = 10 * mm                                   # side margin: 190 mm usable, a 7.5-s strip at 25 mm/s is 187.5 mm
NAVY, ACCENT = colors.HexColor("#1B2A41"), colors.HexColor("#2E6F9E")
INK, MUTED, RULE = colors.HexColor("#222831"), colors.HexColor("#5B6573"), colors.HexColor("#C9D1DB")
PANEL, ALERT, WARN = colors.HexColor("#F1F4F8"), colors.HexColor("#B23A48"), colors.HexColor("#C98A1B")
GRID_MINOR, GRID_MAJOR = colors.HexColor("#F7D9D9"), colors.HexColor("#E9A8A8")
RHYTHM_COLOR = {"N": "#4C9F70", "A": "#C8414B", "O": "#E0A030", "~": "#9AA3AD"}
RHYTHM_NAME = {"N": "Normal rhythm", "A": "Atrial fibrillation", "O": "Other rhythm", "~": "Noisy"}
BEAT_COLOR = {"N": "#7A8494", "S": "#1F5FBF", "V": "#B23A48", "U": "#7B4FA0"}
STRIP_S, STRIP_H, PER_PAGE = 7.5, 30 * mm, 6
DISCLAIMER = ("Research prototype, not a medical device. Automated single-lead results must be reviewed by a qualified "
              "clinician before any clinical use.")

P_BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=8.5, leading=11.2, textColor=INK, alignment=TA_LEFT)
P_SMALL = ParagraphStyle("small", parent=P_BODY, fontSize=7.6, leading=9.8)
P_H = ParagraphStyle("h", parent=P_BODY, fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=NAVY)


def _fmt(v, f="{:.0f}", none="-"):
    return none if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def _ascii(text):
    """Helvetica covers Latin-1 only: replace the few symbols it lacks."""
    return (str(text).replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("→", "->")
            .encode("latin-1", "replace").decode("latin-1"))


class Report:
    def __init__(self, res, s, path, patient=None, source=""):
        self.res, self.s, self.path, self.source = res, s, path, source
        self.patient = {k: (v or "").strip() for k, v in (patient or {}).items()}
        self.c = rl_canvas.Canvas(path, pagesize=A4, pageCompression=1)
        self.c.setTitle("Ambulatory ECG report")
        self.c.setAuthor("single-lead-ecg-analyzer")
        self.c.setCreator("single-lead-ecg-analyzer")
        self.page = 0
        self.events = s["events"]
        self.n_strip_pages = int(np.ceil(len(self.events) / PER_PAGE))
        self.total = 3 + self.n_strip_pages + 1
        self.created = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------ helpers
    def text(self, x, y, txt, size=8.5, font="Helvetica", color=INK, align="left"):
        c = self.c
        c.setFont(font, size)
        c.setFillColor(color)
        txt = _ascii(txt)
        {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[align](x, y, txt)

    def para(self, x, y_top, w, html, style=P_BODY):
        p = Paragraph(_ascii(html), style)
        _, h = p.wrap(w, H)
        p.drawOn(self.c, x, y_top - h)
        return h

    def new_page(self, title):
        if self.page:
            self.c.showPage()
        self.page += 1
        c = self.c
        if self.page == 1:
            c.setFillColor(NAVY)
            c.rect(0, H - 24 * mm, W, 24 * mm, stroke=0, fill=1)
            self.text(MX, H - 12 * mm, "Ambulatory ECG Report", 17, "Helvetica-Bold", colors.white)
            self.text(MX, H - 18.5 * mm, "Single-lead automated analysis: heartbeats, rhythm, signal quality"
                      + (" and ECG shape" if self.s["morph_run"] else ""), 8.5, color=colors.HexColor("#C9D6E8"))
            self.text(W - MX, H - 12 * mm, f"Generated {self.created}", 8, color=colors.HexColor("#C9D6E8"), align="right")
            self.text(W - MX, H - 18.5 * mm, "single-lead-ecg-analyzer", 8, color=colors.HexColor("#C9D6E8"), align="right")
            top = H - 30 * mm
        else:
            c.setFillColor(NAVY)
            c.rect(0, H - 13 * mm, W, 13 * mm, stroke=0, fill=1)
            self.text(MX, H - 8.6 * mm, title, 11.5, "Helvetica-Bold", colors.white)
            who = self.patient.get("name") or ""
            self.text(W - MX, H - 8.6 * mm, ("Ambulatory ECG Report" + (f"  ·  {who}" if who else "")), 8,
                      color=colors.HexColor("#C9D6E8"), align="right")
            top = H - 19 * mm
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.line(MX, 12 * mm, W - MX, 12 * mm)
        self.text(MX, 8 * mm, DISCLAIMER, 6.8, color=MUTED)
        self.text(W - MX, 8 * mm, f"Page {self.page} of {self.total}", 7, color=MUTED, align="right")
        return top

    def section(self, y, title):
        self.text(MX, y, title.upper(), 8, "Helvetica-Bold", ACCENT)
        self.c.setStrokeColor(ACCENT)
        self.c.setLineWidth(0.8)
        self.c.line(MX, y - 1.6 * mm, W - MX, y - 1.6 * mm)
        return y - 6 * mm

    # ------------------------------------------------------------------ page 1
    def page_summary(self):
        res, s = self.res, self.s
        y = self.new_page("Summary")
        # patient and recording
        y = self.section(y, "Patient and recording")
        p = self.patient
        left = [("Name / ID", p.get("name")), ("Age", p.get("age")), ("Sex", p.get("sex")),
                ("Indication", p.get("indication"))]
        start = res.get("start_time")
        dur = s["dur"]
        start_txt = ("not given (times are from the start)" if start is None else
                     start.strftime("%Y-%m-%d %H:%M:%S" if res.get("date_known", True) else "%H:%M:%S (date not given)"))
        right = [("Recording start", start_txt),
                 ("Duration / analyzed", f"{_hm(dur)}  /  {_hm(s['readable_s'])} readable"),
                 ("Signal", f"single lead, {res['fs']:g} Hz" + (", polarity flipped for analysis" if res["inverted"] else "")),
                 ("Electrodes", "limb (wrists and ankle)" if res["placement"] == "limb" else "chest (Holter placement)")]
        for i, ((k1, v1), (k2, v2)) in enumerate(zip(left, right)):
            yy = y - i * 5.2 * mm
            self.text(MX, yy, k1, 7.8, color=MUTED)
            self.text(MX + 24 * mm, yy, (v1 or "-")[:60], 8.6, "Helvetica-Bold" if v1 else "Helvetica")
            self.text(MX + 96 * mm, yy, k2, 7.8, color=MUTED)
            self.text(MX + 127 * mm, yy, v2[:62], 8.6)
        if self.source:
            self.text(MX + 96 * mm, y - 4 * 5.2 * mm, "Source file", 7.8, color=MUTED)
            self.text(MX + 127 * mm, y - 4 * 5.2 * mm, self.source[:40], 8.6)
        y -= 5 * 5.2 * mm + 2 * mm

        # key measures
        y = self.section(y, "Key measures")
        c = s["counts"]
        hrs = max(s["readable_s"] / 3600, 1e-9)
        n = max(s["n_beats"], 1)
        pat = s["patterns"]
        vr, sr = pat["V_runs"], pat["S_runs"]
        brady_min = sum(b - a + 1 for a, b in s["brady"])
        tachy_min = sum(b - a + 1 for a, b in s["tachy"])
        longest_af = max((b - a for a, b in s["af_intervals"]), default=0)
        longest_p = max((q["dur"] for q in s["pauses"]), default=None)
        tiles = [
            ("Heart rate", f"{_fmt(s['hr_mean'])} bpm", "average",
             [f"Min {_fmt(s['hr_min'][0])} bpm at {derive.clock(res, s['hr_min'][1] * 60)}" if np.isfinite(s["hr_min"][0]) else "Min -",
              f"Max {_fmt(s['hr_max'][0])} bpm at {derive.clock(res, s['hr_max'][1] * 60)}" if np.isfinite(s["hr_max"][0]) else "Max -",
              "1-minute means"], None),
            ("Beats analyzed", f"{s['n_beats']:,}", "", [f"Readable time {_hm(s['readable_s'])}",
                                                      f"Unclassified {c['U']:,}"], None),
            ("Ventricular extra beats", f"{c['V']:,}", f"{100 * c['V'] / n:.1f}%",
             [f"{c['V'] / hrs:.0f} per hour", f"Couplets {pat['V_couplets']}  ·  Runs {len(vr)}"
              + (f" (max {max(r['beats'] for r in vr)})" if vr else ""), f"Bigeminy episodes {pat['bigeminy']}"],
             WARN if (vr or pat["V_couplets"]) else None),
            ("Supraventricular extra beats", f"{c['S']:,}", f"{100 * c['S'] / n:.1f}%",
             [f"{c['S'] / hrs:.0f} per hour", f"Couplets {pat['S_couplets']}  ·  Runs {len(sr)}", "Low reliability (see Methods)"],
             None),
            ("Atrial fibrillation", f"{100 * s['af_burden']:.1f}%", "burden",
             [f"Episodes {len(s['af_intervals'])}", f"Longest {longest_af / 60:.1f} min" if s["af_intervals"] else "Longest -",
              "of the readable time"], ALERT if s["af_intervals"] else None),
            ("Pauses of 2 s or longer", f"{len(s['pauses'])}", "",
             [f"Longest {longest_p:.1f} s" if longest_p else "Longest -", "RR interval of 2.0 s or more"],
             ALERT if s["pauses"] else None),
            ("Slow / fast heart rate", f"{brady_min} / {tachy_min}", "min",
             ["Minutes below 50 bpm / above 100 bpm", "(1-minute means)"], None),
            ("Signal quality", f"{_pct(s['readable_share'])}%", "readable",
             [f"Unreadable {_hm(s['dur'] - s['readable_s'])}", "left out of every count"],
             WARN if s["readable_share"] < 0.8 else None),
        ]
        tw, th, gap = (W - 2 * MX - 3 * 3 * mm) / 4, 27 * mm, 3 * mm
        for i, (label, big, unit, lines, accent) in enumerate(tiles):
            col, row = i % 4, i // 4
            x0, y0 = MX + col * (tw + gap), y - (row + 1) * th - row * gap
            self.c.setFillColor(PANEL)
            self.c.setStrokeColor(PANEL)
            self.c.roundRect(x0, y0, tw, th, 1.6 * mm, stroke=0, fill=1)
            self.c.setFillColor(accent or ACCENT)
            self.c.rect(x0, y0 + 2 * mm, 1.1 * mm, th - 4 * mm, stroke=0, fill=1)
            self.text(x0 + 3.5 * mm, y0 + th - 5 * mm, label, 7.4, "Helvetica-Bold", MUTED)
            self.text(x0 + 3.5 * mm, y0 + th - 12.6 * mm, big, 15, "Helvetica-Bold", accent or INK)
            bw = self.c.stringWidth(_ascii(big), "Helvetica-Bold", 15)
            self.text(x0 + 4.5 * mm + bw, y0 + th - 12.6 * mm, unit, 7.6, color=MUTED)
            for k, ln in enumerate(lines):
                self.text(x0 + 3.5 * mm, y0 + th - 17.3 * mm - k * 3.4 * mm, ln, 6.9, color=INK)
        y -= 2 * th + gap + 7 * mm

        # automatic findings
        y = self.section(y, "Automatic findings")
        for f in s["findings"]:
            h = self.para(MX + 4 * mm, y + 2.6 * mm, W - 2 * MX - 4 * mm, escape(f), P_BODY)
            self.c.setFillColor(ACCENT)
            self.c.circle(MX + 1.4 * mm, y + 2.6 * mm - 1.6 * mm, 0.7 * mm, stroke=0, fill=1)
            y -= h + 1.3 * mm
        if res["notes"]:
            y -= 1 * mm
            h = self.para(MX, y + 2.6 * mm, W - 2 * MX, "<b>Signal notes:</b> " + escape("; ".join(res["notes"])) + ".",
                          P_SMALL)
            y -= h + 1 * mm

        # physician review (and the overview chart above it when the findings leave room)
        box_h = 40 * mm
        y_box = 16 * mm
        ov_h = 35 * mm
        if y - 2 * mm - ov_h >= y_box + box_h + 4 * mm:
            yo = y_box + box_h + 4 * mm + ov_h
            yo = self.section(yo + 1 * mm, "Overview of the whole recording") + 1 * mm
            png = overview_figure(res, s)
            img = ImageReader(io.BytesIO(png))
            iw, ih = img.getSize()
            w = W - 2 * MX
            hh = min(w * ih / iw, yo - (y_box + box_h + 4 * mm))
            self.c.drawImage(img, MX, yo - hh, w, hh)
        self.c.setStrokeColor(NAVY)
        self.c.setLineWidth(0.8)
        self.c.rect(MX, y_box, W - 2 * MX, box_h, stroke=1, fill=0)
        self.text(MX + 3 * mm, y_box + box_h - 5.5 * mm, "PHYSICIAN REVIEW AND INTERPRETATION", 8, "Helvetica-Bold", NAVY)
        self.c.setStrokeColor(RULE)
        self.c.setLineWidth(0.4)
        for k in range(3):
            yy = y_box + box_h - 13 * mm - k * 6.5 * mm
            self.c.line(MX + 3 * mm, yy, W - MX - 3 * mm, yy)
        for x0, lab, x1 in ((MX + 3 * mm, "Reviewed by", MX + 64 * mm), (MX + 70 * mm, "Signature", MX + 128 * mm),
                            (MX + 135 * mm, "Date", W - MX - 3 * mm)):
            self.text(x0, y_box + 4 * mm, lab, 7.4, color=MUTED)
            self.c.line(x0 + (17 if lab != "Date" else 9) * mm, y_box + 3.6 * mm, x1, y_box + 3.6 * mm)

    # ------------------------------------------------------------------ page 2
    def page_trends(self):
        res, s = self.res, self.s
        y = self.new_page("Trends")
        png = trend_figure(res, s)
        img = ImageReader(io.BytesIO(png))
        iw, ih = img.getSize()
        w = W - 2 * MX
        h = w * ih / iw
        self.c.drawImage(img, MX, y - h, w, h)
        y -= h + 3 * mm
        legend = ("Heart rate: 1-minute means of readable minutes; shaded red = atrial fibrillation episodes. "
                  "RR intervals: one dot per beat, the line marks the 2-s pause limit. Rhythm: one colour per 30-s window "
                  "(15-s step). Signal: grey = unreadable 10-s strips, left out of every count.")
        self.para(MX, y, W - 2 * MX, legend, P_SMALL)

    # ------------------------------------------------------------------ page 3
    def page_tables(self):
        res, s = self.res, self.s
        y = self.new_page("Tables and event log")
        unit = "Hour" if s["block"] == 3600 else "10-min block"
        y = self.section(y, f"Summary per {unit.lower()}")
        hdr = [unit + " from", "Readable\nmin", "HR min", "HR mean", "HR max", "Beats", "V", "S", "AF min", "Pauses"]
        rows = [hdr]
        for _, r in s["hours"].iterrows():
            rows.append([r["start"], f"{r['readable_min']:.1f}", _fmt(r["hr_min"]), _fmt(r["hr_mean"]), _fmt(r["hr_max"]),
                         f"{int(r['beats']):,}", f"{int(r['V']):,}", f"{int(r['S']):,}", f"{r['af_min']:.1f}",
                         f"{int(r['pauses'])}"])
        tot = s["hours"]
        rows.append(["Total", f"{tot['readable_min'].sum():.1f}", _fmt(s["hr_min"][0]), _fmt(s["hr_mean"]),
                     _fmt(s["hr_max"][0]), f"{int(tot['beats'].sum()):,}", f"{int(tot['V'].sum()):,}",
                     f"{int(tot['S'].sum()):,}", f"{tot['af_min'].sum():.1f}", f"{int(tot['pauses'].sum())}"])
        cw = [27 * mm] + [(W - 2 * MX - 27 * mm) / 9] * 9
        t = Table(rows, colWidths=cw, rowHeights=[8 * mm] + [4.6 * mm] * (len(rows) - 1), repeatRows=1)
        t.setStyle(_table_style(len(rows), total_row=True))
        _, th = t.wrap(W, H)
        t.drawOn(self.c, MX, y - th)
        y -= th + 8 * mm

        y = self.section(y, "Event log")
        log = event_log(res, s)
        max_rows = max(3, int((y - 22 * mm) / (4.6 * mm)) - 1)
        more = len(log) - max_rows
        rows = [["Time", "Event", "Details"]] + [[a, b, c] for a, b, c in top_events(log, max_rows)]
        if not log:
            rows.append(["-", "No events", ""])
        t = Table(rows, colWidths=[27 * mm, 55 * mm, W - 2 * MX - 82 * mm],
                  rowHeights=[5.6 * mm] + [4.6 * mm] * (len(rows) - 1))
        t.setStyle(_table_style(len(rows), align_left=True))
        _, th = t.wrap(W, H)
        t.drawOn(self.c, MX, y - th)
        y -= th + 3 * mm
        if more > 0:
            self.text(MX, y, f"{more} shorter or less important events are not listed (pauses, ventricular runs and "
                      "AF episodes of 1 min or more come first; all events are in the app's CSV download).", 7.4, color=MUTED)

    # ------------------------------------------------------------------ strips
    def page_strips(self):
        res, s = self.res, self.s
        for k in range(self.n_strip_pages):
            y = self.new_page("ECG strips of the main events")
            self.text(MX, y + 1 * mm, "25 mm/s  ·  10 mm/mV  ·  grid 1 mm / 5 mm  ·  letters: N normal, S supraventricular, "
                      "V ventricular, U unclassified  ·  numbers: RR interval in ms", 7, color=MUTED)
            y -= 4 * mm
            for j, (kind, title, t0) in enumerate(self.events[k * PER_PAGE:(k + 1) * PER_PAGE]):
                num = k * PER_PAGE + j + 1
                in_strip = (s["beat_t"] >= t0) & (s["beat_t"] < t0 + STRIP_S)
                bt = s["beat_t"][in_strip]
                hr = f"  ·  HR {60 / np.diff(bt).mean():.0f} bpm" if len(bt) > 2 else ""
                self.text(MX, y - 3 * mm, f"{num}. {title}", 8.6, "Helvetica-Bold", INK)
                self.text(W - MX, y - 3 * mm, f"{derive.clock(res, t0)} - {derive.clock(res, t0 + STRIP_S)}{hr}", 7.8,
                          color=MUTED, align="right")
                draw_strip(self.c, MX, y - 8.4 * mm - STRIP_H, res, s, t0)
                y -= 8.4 * mm + STRIP_H + 4.6 * mm

    # ------------------------------------------------------------------ methods
    def page_methods(self):
        res, s = self.res, self.s
        v = validation.all_numbers()
        b, r, m = v["beats"], v["rhythm"], v["morphology"]
        y = self.new_page("Methods, validation and limitations")
        y = self.section(y, "How the analysis works")
        steps = [
            f"<b>1. Signal quality.</b> The recording is cut into 10-s strips. A quality gate estimates each strip's "
            f"signal-to-noise ratio; strips below {res['gate_threshold']:.1f} dB are marked unreadable and left out of every "
            "count, so noise is never reported as a finding.",
            "<b>2. Heartbeats.</b> R peaks are found once (Project 1 detector, 360 Hz). Every beat is classified as N "
            "(normal), S (supraventricular extra beat) or V (ventricular extra beat) by a 5-network ensemble that uses "
            "the beat shape and the RR intervals around it. Fusion and unclear beats are shown as U.",
            "<b>3. Rhythm.</b> Every 30-s window (15-s step) is classified as normal rhythm, atrial fibrillation (AF), "
            "other rhythm or noisy by a 5-network hybrid ensemble (Project 2). AF windows are merged into episodes; the AF "
            "burden is AF time divided by readable time. Supraventricular extra beats are not counted during AF.",
            "<b>4. ECG shape</b> (limb electrodes only). Each readable 10-s strip is screened for five diagnostic groups and "
            "named findings by single-lead networks trained on resting ECGs (Project 3). "
            + ("Performed for this recording." if s["morph_run"] else
               "Not performed for this recording: on chest (Holter) electrodes these models called almost every strip "
               "abnormal, even in a normal-rhythm recording, so their output is not valid there."),
            "<b>5. Measures.</b> Heart rate from RR intervals of consecutive readable beats; 1-minute means need 40 s of "
            "readable signal. Pause: RR of 2.0 s or more. Run: 3 or more consecutive extra beats. Bigeminy: an extra beat "
            "after every normal beat for 3 or more cycles.",
        ]
        for st in steps:
            h = self.para(MX, y + 2.6 * mm, W - 2 * MX, st, P_SMALL)
            y -= h + 1.6 * mm
        y -= 2 * mm
        y = self.section(y, "Validation on public databases (patients never used in training)")
        rows = [["Part", "Test data", "Result"],
                ["V beats (detector + classifier)", f"MIT-BIH DS2, 22 patients, {b['V']['n']:,} V beats",
                 f"Sensitivity {b['V']['se']:.1f}%, PPV {b['V']['ppv']:.1f}%"],
                ["S beats", f"MIT-BIH DS2, {b['S']['n']:,} S beats",
                 f"Sensitivity {b['S']['se']:.1f}%, PPV {b['S']['ppv']:.1f}% (low)"],
                ["Fusion beats (F)", f"MIT-BIH DS2, {b['F']['n']:,} F beats", f"Sensitivity {b['F']['se']:.0f}%: not detected"],
                ["Normal beats", f"MIT-BIH DS2, {b['N']['n']:,} N beats",
                 f"Sensitivity {b['N']['se']:.1f}%, PPV {b['N']['ppv']:.1f}%"]]
        for db in ("CinC 2017 test", "MIT-BIH DS2", "MIT-BIH AF DB"):
            q = r[db]
            rows.append(["Atrial fibrillation" if db == "CinC 2017 test" else "", f"{db}, {q['n']:,} {q['unit']}",
                         f"Sensitivity {q['se']:.1f}%, PPV {q['ppv']:.1f}%, specificity {q['sp']:.1f}%"])
        rows += [["ECG shape, 5 groups", "PTB-XL test (internal), lead II", f"Mean AUC {m['internal_macro']:.3f}"],
                 ["(limb electrodes)", "Chapman-Shaoxing (external), lead II", f"Mean AUC {m['external_macro']:.3f}"],
                 ["Quality gate", "PTB-XL test + real electrode noise", f"Catches {m['gate_sensitivity']} of strips below "
                  f"0 dB; {m['gate_false_rejections']} false rejections"]]
        t = Table(rows, colWidths=[46 * mm, 70 * mm, W - 2 * MX - 116 * mm], rowHeights=[5.6 * mm] + [4.8 * mm] * (len(rows) - 1))
        t.setStyle(_table_style(len(rows), align_left=True))
        _, th = t.wrap(W, H)
        t.drawOn(self.c, MX, y - th)
        y -= th + 6 * mm
        y = self.section(y, "Limitations")
        fa = ", ".join(f"{rec} ({pct:.0f}% false AF burden)" for rec, pct in r["false_alarm_records"])
        lims = [
            "A single lead cannot replace a 12-lead ECG or a cardiologist's review; findings are prompts for review, "
            "not diagnoses.",
            f"S beats are detected poorly (sensitivity {b['S']['se']:.0f}%) and fusion beats not at all; "
            f"{100 - b['N']['se']:.1f}% of normal beats get another label.",
            f"Frequent extra beats can be mistaken for AF: MIT-BIH records {fa} in the external test. The report warns when "
            "AF episodes contain many ventricular extra beats.",
            "The ECG-shape screening was validated only on resting 10-s ECGs with limb electrodes, and its operating "
            f"points have modest PPV (for example MI {m['operating']['MI']['ppv']:.0f}%, hypertrophy "
            f"{m['operating']['HYP']['ppv']:.0f}%).",
            "Validated on public databases only. Not yet tested on the AD8232 hardware or on new patients.",
        ]
        for lm in lims:
            h = self.para(MX + 4 * mm, y + 2.6 * mm, W - 2 * MX - 4 * mm, escape(lm), P_SMALL)
            self.c.setFillColor(MUTED)
            self.c.circle(MX + 1.4 * mm, y + 2.6 * mm - 1.5 * mm, 0.6 * mm, stroke=0, fill=1)
            y -= h + 1.2 * mm
        y -= 3 * mm
        y = self.section(y, "Software")
        sw = "; ".join(f"{name} <font color='#5B6573'>({vendor.REPOS[role][1][:7]})</font>"
                       for role, name in (("beats", "Project 1: ecg-arrhythmia-detector"),
                                          ("rhythm", "Project 2: ecg-af-detector"),
                                          ("morphology", "Project 3: ecg-lead2-diagnosis")))
        self.para(MX, y + 2.6 * mm, W - 2 * MX, f"Engines (GitHub, pinned commits): {sw}. Analysis time "
                  f"{res['seconds']:.0f} s on CPU.", P_SMALL)

    def build(self):
        self.page_summary()
        self.page_trends()
        self.page_tables()
        self.page_strips()
        self.page_methods()
        self.c.save()
        return self.path


# ---------------------------------------------------------------------- pieces
def _pct(x):
    """Readable share: one decimal unless it is exactly 0 or 100%."""
    return f"{100 * x:.0f}" if x in (0.0, 1.0) else f"{min(100 * x, 99.9):.1f}"


def _hm(sec):
    sec = int(round(sec))
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min" if sec >= 3600 else f"{sec / 60:.1f} min"


def _table_style(n, total_row=False, align_left=False):
    st = [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.4), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.4),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("BACKGROUND", (0, 0), (-1, 0), NAVY),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "LEFT" if align_left else "CENTER"),
          ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]
    for i in range(2, n, 2):
        st.append(("BACKGROUND", (0, i), (-1, i), PANEL))
    if total_row:
        st += [("FONT", (0, n - 1), (-1, n - 1), "Helvetica-Bold", 7.4), ("LINEABOVE", (0, n - 1), (-1, n - 1), 0.8, NAVY)]
    return TableStyle(st)


def event_log(res, s):
    ev = []
    for a, b in s["af_intervals"]:
        ev.append((a, "Atrial fibrillation", f"until {derive.clock(res, b)}, {(b - a) / 60:.1f} min"))
    for p in s["pauses"]:
        ev.append((p["t"], "Pause", f"{p['dur']:.2f} s"))
    for c, name in (("V", "Ventricular run"), ("S", "Supraventricular run")):
        for x in s["patterns"][f"{c}_runs"]:
            ev.append((x["t"], name, f"{x['beats']} beats at {x['rate']:.0f} bpm"))
    for kind, key in (("Heart rate below 50 bpm", "brady"), ("Heart rate above 100 bpm", "tachy")):
        for a, b in s["rate_episodes"][key]:
            hr = s["hr_minute"][a:b + 1]
            ev.append((a * 60.0, kind, f"{b - a + 1} min episode, {'lowest' if key == 'brady' else 'highest'} "
                                       f"{(np.nanmin(hr) if key == 'brady' else np.nanmax(hr)):.0f} bpm"))
    for a, b in s["unreadable"]:
        if b - a >= 60:
            ev.append((float(a), "Unreadable signal", f"{(b - a) / 60:.1f} min left out"))
    ev.sort(key=lambda e: e[0])
    return [(derive.clock(res, t), k, d) for t, k, d in ev]


PRIORITY = {"Pause": 0, "Ventricular run": 1, "Atrial fibrillation": 2, "Heart rate below 50 bpm": 3,
            "Heart rate above 100 bpm": 3, "Supraventricular run": 4, "Unreadable signal": 6}


def top_events(log, n):
    """At most n events: pauses, ventricular runs and AF episodes of 1 min or more first; then back in time order."""
    def rank(i):
        _, k, d = log[i]
        p = PRIORITY.get(k, 5)
        if k == "Atrial fibrillation" and float(d.split(", ")[-1].split(" ")[0]) < 1.0:
            p = 5
        return (p, i)
    keep = sorted(sorted(range(len(log)), key=rank)[:n])
    return [log[i] for i in keep]


def draw_strip(c, x, y, res, s, t0, L=STRIP_S, h=STRIP_H):
    """One ECG strip at 25 mm/s and 10 mm/mV on a 1-mm / 5-mm grid, with beat letters and RR intervals."""
    w = L * 25 * mm
    c.saveState()
    c.setLineWidth(0.18)
    c.setStrokeColor(GRID_MINOR)
    for i in range(int(round(w / mm)) + 1):
        if i % 5:
            c.line(x + i * mm, y, x + i * mm, y + h)
    for j in range(int(round(h / mm)) + 1):
        if j % 5:
            c.line(x, y + j * mm, x + w, y + j * mm)
    c.setLineWidth(0.45)
    c.setStrokeColor(GRID_MAJOR)
    for i in range(0, int(round(w / mm)) + 1, 5):
        c.line(x + i * mm, y, x + i * mm, y + h)
    for j in range(0, int(round(h / mm)) + 1, 5):
        c.line(x, y + j * mm, x + w, y + j * mm)
    # 1-s ticks
    c.setStrokeColor(colors.HexColor("#C77"))
    c.setLineWidth(0.6)
    for k in range(int(L) + 1):
        c.line(x + k * 25 * mm, y, x + k * 25 * mm, y - 1.2 * mm)
    # trace
    fs = res["signal_fs"]
    i0, i1 = int(round(t0 * fs)), int(round((t0 + L) * fs))
    v = np.asarray(res["signal"][i0:i1], float)
    if len(v) > 1:
        v = v - np.median(v)
        base = y + 0.40 * h
        p = c.beginPath()
        p.rect(x, y, w, h)
        c.clipPath(p, stroke=0, fill=0)
        xs = x + np.arange(len(v)) / fs * 25 * mm
        ys = base + np.clip(v, -2.5, 4.0) * 10 * mm
        path = c.beginPath()
        path.moveTo(xs[0], ys[0])
        for a, b in zip(xs[1:], ys[1:]):
            path.lineTo(a, b)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.55)
        c.setLineJoin(1)
        c.drawPath(path, stroke=1, fill=0)
    c.restoreState()
    # beat letters and RR intervals
    sel = (s["beat_t"] >= t0) & (s["beat_t"] < t0 + L)
    bt, bl = s["beat_t"][sel], s["beat_label"][sel]
    prev = s["beat_t"][np.searchsorted(s["beat_t"], t0) - 1] if np.searchsorted(s["beat_t"], t0) > 0 else None
    for k, (t, lab) in enumerate(zip(bt, bl)):
        lab = "U" if lab in ("F", "Q") else lab
        xx = x + (t - t0) * 25 * mm
        c.setFont("Helvetica-Bold" if lab != "N" else "Helvetica", 7.2 if lab != "N" else 6.2)
        c.setFillColor(colors.HexColor(BEAT_COLOR[lab]))
        c.drawCentredString(xx, y + h + 1.1 * mm, lab)
        before = bt[k - 1] if k else prev
        if before is not None and t - before < 6 and (t + before) / 2 >= t0 + 0.12:
            c.setFont("Helvetica", 5.6)
            c.setFillColor(MUTED)
            c.drawCentredString(x + ((t + before) / 2 - t0) * 25 * mm, y + 1.2 * mm, f"{1000 * (t - before):.0f}")
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.rect(x, y, w, h, stroke=1, fill=0)


def _time_axis(res, dur):
    """Tick spacing in whole minutes or hours, labels h:mm (with seconds only for recordings under 10 min)."""
    steps = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600]
    step = next((x for x in steps if dur / x <= 10), 21600) if dur >= 600 else max(10, round(dur / 8 / 10) * 10)
    short = dur < 600 and not res.get("start_time")
    fmt = FuncFormatter(lambda t, _: derive.clock(res, t) if short else derive.clock(res, t)[:-3])
    return MultipleLocator(step), fmt


def trend_figure(res, s):
    dur = s["dur"]
    loc, fmt = _time_axis(res, dur)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#8A94A3", "xtick.color": "#5B6573", "ytick.color": "#5B6573",
                         "axes.labelcolor": "#222831", "axes.titlesize": 8.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlecolor": "#1B2A41"})
    fig, ax = plt.subplots(5, 1, figsize=(7.48, 9.5), dpi=220, sharex=True,
                           gridspec_kw=dict(height_ratios=[3.2, 2.6, 0.55, 0.55, 2.2], hspace=0.42))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(0, dur)
        a.xaxis.set_major_formatter(fmt)
        a.xaxis.set_major_locator(loc)

    def af_shade(a):
        for p, q in s["af_intervals"]:
            a.axvspan(p, q, color="#C8414B", alpha=0.12, lw=0)

    # heart rate
    hr = s["hr_minute"]
    t = np.arange(len(hr)) * 60 + 30
    af_shade(ax[0])
    ax[0].plot(t, hr, color="#1B2A41", lw=0.9)
    ax[0].axhline(derive.BRADY, color="#2E6F9E", lw=0.6, ls="--")
    ax[0].axhline(derive.TACHY, color="#C98A1B", lw=0.6, ls="--")
    lo = np.nanmin(hr) if np.isfinite(hr).any() else 40
    hi = np.nanmax(hr) if np.isfinite(hr).any() else 120
    ax[0].set_ylim(min(40, lo - 5), max(110, hi + 8))
    ax[0].set_ylabel("bpm")
    ax[0].set_title("Heart rate (1-minute means)")
    ax[0].grid(axis="y", color="#E3E7ED", lw=0.5)
    # RR tachogram
    af_shade(ax[1])
    ax[1].scatter(s["rr_t"], s["rr"], s=0.6 if len(s["rr"]) > 20000 else 1.5, color="#1B2A41", lw=0, alpha=0.6,
                  rasterized=True)
    ax[1].axhline(derive.PAUSE_S, color="#B23A48", lw=0.6, ls="--")
    ax[1].set_ylim(0.2, max(2.4, min(4.0, float(np.max(s["rr"])) + 0.2) if len(s["rr"]) else 2.4))
    ax[1].set_ylabel("RR (s)")
    ax[1].set_title("RR intervals (one dot per beat)")
    ax[1].grid(axis="y", color="#E3E7ED", lw=0.5)
    # rhythm band
    for a0, a1, cl in zip(res["win_start"], res["win_end"], res["win_class"]):
        ax[2].axvspan(a0, a1, color=RHYTHM_COLOR[cl], lw=0)
    ax[2].set_yticks([])
    ax[2].set_title("Rhythm (30-s windows)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RHYTHM_COLOR[k]) for k in RHYTHM_COLOR]
    ax[2].legend(handles, [RHYTHM_NAME[k] for k in RHYTHM_COLOR], ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0),
                 frameon=False, fontsize=6.8, handlelength=1.2, columnspacing=1.2)
    # signal quality band
    ax[3].axvspan(0, dur, color="#D8EBDD", lw=0)
    for a0, a1 in s["unreadable"]:
        ax[3].axvspan(a0, min(a1, dur), color="#8A94A3", lw=0)
    ax[3].set_yticks([])
    ax[3].set_title(f"Signal quality ({100 * s['readable_share']:.0f}% readable)")
    # extra beats per period
    blk = s["block"]
    edges = np.arange(0, dur + blk, blk)
    vt, vl = s["beat_t"], s["beat_label"]
    nv = np.histogram(vt[vl == "V"], edges)[0]
    ns = np.histogram(vt[vl == "S"], edges)[0]
    mid = edges[:-1] + blk / 2
    bw = blk * 0.38
    ax[4].bar(mid - bw / 2, nv, bw, color="#B23A48", label="V (ventricular)")
    ax[4].bar(mid + bw / 2, ns, bw, color="#1F5FBF", label="S (supraventricular)")
    ax[4].set_title(f"Extra beats per {'hour' if blk == 3600 else '10 minutes'}")
    ax[4].set_ylabel("beats")
    ax[4].legend(frameon=False, fontsize=6.8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax[4].grid(axis="y", color="#E3E7ED", lw=0.5)
    ax[4].set_xlabel("clock time" if res.get("start_time") else
                     f"time from the start ({'h:mm:ss' if dur < 600 else 'h:mm'})")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return buf.getvalue()


def overview_figure(res, s):
    """Page-1 overview: heart rate with AF shading, and the rhythm band, over the whole recording."""
    dur = s["dur"]
    loc, fmt = _time_axis(res, dur)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#8A94A3", "xtick.color": "#5B6573", "ytick.color": "#5B6573"})
    fig, ax = plt.subplots(2, 1, figsize=(7.48, 1.45), dpi=220, sharex=True,
                           gridspec_kw=dict(height_ratios=[3.2, 0.7], hspace=0.12))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(0, dur)
        a.xaxis.set_major_formatter(fmt)
        a.xaxis.set_major_locator(loc)
    for p, q in s["af_intervals"]:
        ax[0].axvspan(p, q, color="#C8414B", alpha=0.12, lw=0)
    hr = s["hr_minute"]
    ax[0].plot(np.arange(len(hr)) * 60 + 30, hr, color="#1B2A41", lw=0.9)
    ax[0].set_ylabel("HR (bpm)", fontsize=6.8)
    ax[0].grid(axis="y", color="#E3E7ED", lw=0.5)
    if np.isfinite(hr).any():
        ax[0].set_ylim(max(0, np.nanmin(hr) - 10), np.nanmax(hr) + 10)
    for a0, a1, cl in zip(res["win_start"], res["win_end"], res["win_class"]):
        ax[1].axvspan(a0, a1, color=RHYTHM_COLOR[cl], lw=0)
    for a0, a1 in s["unreadable"]:
        ax[1].axvspan(a0, min(a1, dur), color="#FFFFFF", alpha=0.75, lw=0, hatch="////", ec="#8A94A3")
    ax[1].set_yticks([0.5])
    ax[1].set_yticklabels(["rhythm"], fontsize=6.8)
    ax[1].tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RHYTHM_COLOR[k]) for k in RHYTHM_COLOR]
    ax[0].legend(handles, [RHYTHM_NAME[k] for k in RHYTHM_COLOR], ncol=4, loc="lower right", bbox_to_anchor=(1, 0.98),
                 frameon=False, fontsize=6.3, handlelength=1.1, columnspacing=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return buf.getvalue()


def build_pdf(res, s, path, patient=None, source=""):
    return Report(res, s, path, patient, source).build()

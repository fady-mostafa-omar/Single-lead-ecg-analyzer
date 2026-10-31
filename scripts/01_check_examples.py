"""Runs the full analyzer on the five Holter recordings bundled with Project 2 (MIT-BIH 100, 200, 202, 210 and a
1-h excerpt of AF Database 04043; chest electrodes; expert AF annotations) and writes:
  results/examples_summary.csv          beats, heart rate and AF burden (model vs cardiologists) per recording
  results/morphology_on_chest_leads.csv the ECG-shape models forced to run on these chest-lead recordings
  docs/sample_report.pdf (+ PNG previews of its pages)
Records 100, 200, 202 and 210 are MIT-BIH DS2 (never used to train the beat model); none of the five was used to
train the rhythm model."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from holter import vendor  # noqa: E402
from holter.derive import STRIP, summarize  # noqa: E402
from holter.engine import Analyzer  # noqa: E402
from holter.report import build_pdf  # noqa: E402

RES, DOCS = os.path.join(ROOT, "results"), os.path.join(ROOT, "docs")
os.makedirs(RES, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)
ORDER = ["mit100", "mit200", "mit202", "mit210", "afdb04043"]
NAME = {"mit100": "MIT-BIH 100", "mit200": "MIT-BIH 200", "mit202": "MIT-BIH 202", "mit210": "MIT-BIH 210",
        "afdb04043": "AFDB 04043 (1 h)"}
GROUPS = ["NORM", "MI", "STTC", "CD", "HYP"]


def truth_burden(af, ok, dur):
    """Cardiologists' AF time inside readable strips / readable time, on the same 10-s grid as the model."""
    t = np.arange(len(ok)) * STRIP
    covered = np.zeros(len(ok))
    for a, b in af:
        covered += np.clip(np.minimum(t + STRIP, b) - np.maximum(t, a), 0, None)
    return float((covered * ok).sum() / max(ok.sum() * STRIP, 1e-9))


def main():
    z = np.load(vendor.path("rhythm", "app", "examples", "examples.npz"))
    meta = {m["key"]: m for m in json.loads(str(z["meta"]))}
    A = Analyzer()
    rows, morph = [], []
    for k in ORDER:
        m = meta[k]
        x = z[k] / 1000.0
        t0 = time.time()
        r = A.analyze(x, m["fs"], placement="chest")
        sec = time.time() - t0
        s = summarize(r)
        c = s["counts"]
        rows.append({"recording": NAME[k], "minutes": round(r["dur"] / 60, 1), "sampling rate (Hz)": m["fs"],
                     "readable %": round(100 * s["readable_share"], 1), "beats": s["n_beats"], "V": c["V"], "S": c["S"],
                     "unclassified": c["U"], "mean HR (bpm)": round(s["hr_mean"], 1),
                     "AF burden, cardiologists %": round(100 * truth_burden(m["af"], r["strip_ok"], r["dur"]), 1),
                     "AF burden, model %": round(100 * s["af_burden"], 1), "AF episodes, model": len(s["af_intervals"]),
                     "V share inside AF %": round(100 * s["af_ectopy"], 1),
                     "polarity flipped": r["inverted"], "analysis time (s)": round(sec, 1)})
        f = A.analyze(x, m["fs"], placement="limb")          # forced: shape models on a chest-lead recording
        ok = f["strip_ok"]
        rec = {"recording": NAME[k], "readable strips": int(ok.sum())}
        for j, g in enumerate(GROUPS):
            rec[f"{g} %"] = round(100 * float((f["strip_p5"][ok, j] >= f["thr5"][g]).mean()), 1)
        morph.append(rec)
        print(f"{NAME[k]:18s} AF model {rows[-1]['AF burden, model %']:5.1f}% vs cardiologists "
              f"{rows[-1]['AF burden, cardiologists %']:5.1f}% | V {c['V']} | shape forced: "
              + " ".join(f"{g} {rec[g + ' %']:.0f}%" for g in GROUPS))
        if k == "mit202":
            pdf = os.path.join(DOCS, "sample_report.pdf")
            build_pdf(r, s, pdf, patient=dict(name="Example patient", age="", sex="", indication="Palpitations"),
                      source="MIT-BIH record 202")
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(pdf)
                for i in range(len(doc)):
                    doc[i].render(scale=1.4).to_pil().convert("RGB").save(os.path.join(DOCS, f"sample_report_p{i + 1}.png"),
                                                                          optimize=True)
                doc.close()
            except ImportError:
                print("pypdfium2 not installed: PNG previews skipped")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "examples_summary.csv"), index=False)
    pd.DataFrame(morph).to_csv(os.path.join(RES, "morphology_on_chest_leads.csv"), index=False)
    print("Wrote results/examples_summary.csv, results/morphology_on_chest_leads.csv and docs/sample_report.pdf")


if __name__ == "__main__":
    main()

"""Validation numbers of the three engines, read from the result files of their own repositories (pinned commits),
so the report and the README never retype a number."""
import pandas as pd

from . import vendor


def _csv(role, name):
    return pd.read_csv(vendor.path(role, "results", name))


def beats():
    """Project 1, end to end (own R-peak detector + beat network) on MIT-BIH DS2: 22 patients never used in training."""
    t = _csv("beats", "end_to_end_ds2.csv").set_index("class")
    return {c: dict(n=int(t.loc[c, "n_test"]), se=float(t.loc[c, "Se %"]), ppv=float(t.loc[c, "PPV %"]))
            for c in ("N", "S", "V", "F")}


def rhythm():
    """Project 2: AF detection per 30-s window on two databases never used in training, plus the internal test."""
    w = _csv("rhythm", "external_windows.csv")
    w = w[w["model"] == "CNN ensemble"].set_index("database")
    t = _csv("rhythm", "af_detection_test.csv").set_index("model").loc["CNN ensemble"]
    out = {db: dict(n=int(w.loc[db, "windows"]), unit="30-s windows", se=float(w.loc[db, "Se %"]),
                    ppv=float(w.loc[db, "PPV %"]), sp=float(w.loc[db, "Specificity %"])) for db in w.index}
    out["CinC 2017 test"] = dict(n=int(t["AF recordings"]), unit="AF recordings", se=float(t["Se %"]),
                                 ppv=float(t["PPV %"]), sp=float(t["Specificity %"]))
    b = _csv("rhythm", "external_burden.csv")
    fa = b[(b["database"] == "mitdb") & (b["true AF %"] == 0) & (b["CNN AF %"] > 5)]
    out["false_alarm_records"] = [(str(r["record"]), float(r["CNN AF %"])) for _, r in fa.iterrows()]
    return out


def morphology():
    """Project 3: lead II, 10-s resting ECGs, limb electrodes. Internal PTB-XL test and external Chapman-Shaoxing."""
    a = _csv("morphology", "cnn_auc.csv")
    internal = a[(a["config"] == "lead II") & (a["model"] == "CNN, 5-seed ensemble")].iloc[0]
    e = _csv("morphology", "external_chapman_auc.csv")
    e = e[e["model"] == "lead II, noise-trained"]
    ext = {r["task"]: float(r["AUC external"]) for _, r in e.iterrows()}
    groups = ["NORM", "MI", "STTC", "CD", "HYP"]
    thr = _csv("morphology", "thresholds_test_performance.csv")
    thr = thr[thr["kind"] == "group"].set_index("name")
    q = _csv("morphology", "quality_gate_rejections.csv").iloc[0]
    return dict(internal_macro=float(internal["macro AUC"]),
                internal={g: float(internal[g]) for g in groups},
                external=ext,
                external_macro=float(sum(ext[g] for g in groups if g in ext) / sum(g in ext for g in groups)),
                operating={g: dict(se=float(thr.loc[g, "sensitivity %"]), sp=float(thr.loc[g, "specificity %"]),
                                   ppv=float(thr.loc[g, "PPV %"])) for g in groups},
                gate_sensitivity=str(q["sensitivity (SNR < 0 dB rejected)"]),
                gate_false_rejections=str(q["false rejections (SNR >= 0 dB)"]))


def all_numbers():
    return dict(beats=beats(), rhythm=rhythm(), morphology=morphology())

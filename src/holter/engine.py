"""Runs the three engines on one single-lead recording.
  Beats (Project 1): every beat N / S / V / F / Q, at 360 Hz.
  Rhythm (Project 2): every 30-s window (15-s step) normal / AF / other / noisy.
  Morphology (Project 3): every 10-s strip at 100 Hz: quality gate first, then 5 diagnostic groups and 23 named findings.
R peaks are detected once (Project 1 detector, in 10-min chunks) and reused for the rhythm features and the heart rate.
Beats, windows and strips that fall where the quality gate refused the signal are marked unreadable and left out of
every count.
Electrode placement matters for the morphology step only: its models were trained on resting ECGs with standard limb
electrodes (wrists and ankle). On chest-placed Holter leads they called almost every strip abnormal, even in a
normal-rhythm recording (see results/), so with placement="chest" the step only runs the quality gate."""
import json
import os
import time
from fractions import Fraction

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import vendor

vendor.activate()
from ecgsuite.beats import CLASSES as BEAT_CLASSES, FS as FS_BEAT, bandpass as beat_bandpass  # noqa: E402
from ecgsuite.cnn import load_checkpoint as load_beatnet  # noqa: E402
from ecgsuite.detect import detect_r_peaks  # noqa: E402
from ecgsuite.features import rr_features  # noqa: E402
from ecgsuite.pipeline import beats_from_peaks  # noqa: E402
from afsuite.cnn import load_members as load_afnet, make_windows, predict_windows, window_starts  # noqa: E402
from afsuite.data import CLASSES as RHYTHM_CLASSES  # noqa: E402
from leadsuite.cnn import load_members as load_leadnet, predict as lead_predict, preprocess as lead_preprocess  # noqa: E402
from leadsuite.data import SUPER  # noqa: E402
from leadsuite.quality import estimate_snr  # noqa: E402

FS_MORPH, STRIP = 100, 10
MAX_HOURS = 24
MIN_SECONDS = 10
PLACEMENTS = ("chest", "limb")


def resample(x, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, float)
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, float), fr.numerator, fr.denominator)


def _bandpass(x, fs):
    sos = butter(3, [0.5, 40.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def check_signal(sig, fs):
    """Polarity and amplitude check. Returns (inverted, scale, scale_morph, notes).
    Polarity is decided on the beats themselves: the recording counts as upside down when most R peaks of its first
    10 minutes point down. (The percentile rule of the Project 1 and 2 apps flipped MIT-BIH 200, whose normal beats are
    upright but whose large ventricular beats point down.) Beats and rhythm use the corrected signal; morphology uses
    the signal as recorded (lead II polarity is diagnostic)."""
    head = sig[: int(min(len(sig), 600 * fs))]
    f = _bandpass(head, fs)
    amp = np.percentile(np.abs(f), 99.5)
    try:
        x360 = resample(head, fs, FS_BEAT)
        pk = detect_r_peaks(x360)
        up = beat_bandpass(x360)[pk] if len(pk) else np.zeros(0)
    except Exception:
        up = np.zeros(0)
    if len(up) >= 5:
        inverted = bool(np.median(up) < 0)
    else:                                   # too few beats found: fall back to the percentile rule
        inverted = bool(-np.percentile(f, 0.5) > 1.5 * np.percentile(f, 99.5))
    scale, scale_morph = 1.0, 1.0
    notes = []
    if inverted:
        notes.append("R waves point down: beats and rhythm were analyzed on the flipped signal")
    if amp > 5 or amp < 0.3:          # beats and rhythm: the rule of Projects 1 and 2
        scale = 1.2 / max(amp, 1e-9)
    if amp > 5 or amp < 0.05:         # morphology: low voltage is a finding, so only clearly non-mV units are rescaled
        scale_morph = scale
        notes.append(f"amplitude rescaled ×{scale:.3g} (values did not look like millivolts); voltage-based findings "
                     "(hypertrophy) are unreliable")
    return inverted, scale, scale_morph, notes


class Analyzer:
    def __init__(self):
        self.beatnet, self.beat_std = load_beatnet(vendor.path("beats", "models", "beatnet.pt"))
        self.afnet, self.af_scaler, _ = load_afnet(vendor.path("rhythm", "models", "afnet_ensemble.pt"))
        m = vendor.path("morphology", "models")
        self.lead5, ck = load_leadnet(os.path.join(m, "cnn_leadII_noise.pt"))
        self.lead23, _ = load_leadnet(os.path.join(m, "cnn_leadII_sub23.pt"))
        self.lead_scale = np.float32(ck["scale"][0])
        self.gate = joblib.load(os.path.join(m, "quality_gate.joblib"))
        th = json.load(open(os.path.join(m, "thresholds.json")))
        self.thr5, self.thr23, self.names23 = th["super"], th["sub23"], th["sub23_names"]
        rel = pd.read_csv(vendor.path("morphology", "results", "sub23_leadII_auc.csv"))
        self.reliability = {r["subclass"]: dict(auc=float(r["AUC"]), ci=str(r["95% CI"]),
                                                tier="high" if r["AUC"] >= 0.90 else ("moderate" if r["AUC"] >= 0.85 else "low"))
                            for _, r in rel.iterrows()}
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    # ------------------------------------------------------------------ steps
    def _peaks(self, sig360, progress):
        chunk, pad = 600 * FS_BEAT, 5 * FS_BEAT
        out = []
        for k, a in enumerate(range(0, len(sig360), chunk)):
            lo, hi = max(0, a - pad), min(len(sig360), a + chunk + pad)
            try:
                p = detect_r_peaks(sig360[lo:hi]) + lo
            except Exception:                         # flat or very short chunk
                p = np.array([], int)
            out.append(p[(p >= a) & (p < a + chunk)])
            progress(0.05 + 0.30 * min(1.0, (a + chunk) / len(sig360)), "Finding the heartbeats")
        p = np.unique(np.concatenate(out)) if out else np.array([], int)
        if len(p) > 1:                                # chunk borders: drop detections closer than 200 ms
            keep = np.r_[True, np.diff(p) > int(0.2 * FS_BEAT)]
            p = p[keep]
        return p

    def _beats(self, sig360, peaks):
        f = beat_bandpass(sig360)
        p, X, rr = beats_from_peaks(f, peaks)
        if len(p) == 0:
            return p, np.zeros((0, len(BEAT_CLASSES)), np.float32)
        probs = []
        with torch.no_grad():
            for i in range(0, len(p), 20000):
                w, r = self.beat_std(X[i:i + 20000], rr_features(rr[i:i + 20000]))
                probs.append(torch.softmax(self.beatnet(w, r), 1).numpy())
        return p, np.concatenate(probs)

    def _rhythm(self, sig, fs, peaks_native):
        waves, feats = make_windows(sig, fs, peaks_native)
        F = self.af_scaler(pd.DataFrame(feats).to_numpy(float))
        P = np.mean([predict_windows(m, waves, F, "cpu") for m in self.afnet], 0)
        starts = np.array(window_starts(len(sig), fs))
        return starts, P

    def _morphology(self, sig100, progress, classify=True):
        n = len(sig100) // (STRIP * FS_MORPH)
        if n == 0:
            return dict(est=np.zeros(0), ok=np.zeros(0, bool), p5=np.zeros((0, 5)), p23=np.zeros((0, len(self.names23))))
        strips = sig100[: n * STRIP * FS_MORPH].reshape(n, STRIP * FS_MORPH)
        xb = lead_preprocess(strips[:, :, None].astype(np.float32), 1.0) / self.lead_scale
        est = estimate_snr(self.gate, xb[:, 0])
        ok = est >= self.gate["threshold"]
        p5 = np.zeros((n, 5), np.float32)
        p23 = np.zeros((n, len(self.names23)), np.float32)
        idx = np.where(ok)[0] if classify else np.zeros(0, int)
        for k in range(0, len(idx), 2000):
            j = idx[k:k + 2000]
            p5[j] = np.mean([lead_predict(m, xb[j], "cpu") for m in self.lead5], 0)
            p23[j] = np.mean([lead_predict(m, xb[j], "cpu") for m in self.lead23], 0)
            progress(0.70 + 0.28 * min(1.0, (k + 2000) / max(len(idx), 1)), "Checking the ECG shape")
        return dict(est=est, ok=ok, p5=p5, p23=p23)

    # ------------------------------------------------------------------ main
    def analyze(self, sig, fs, start_time=None, placement="chest", progress=None):
        assert placement in PLACEMENTS, placement
        t_all = time.time()
        progress = progress or (lambda f, msg="": None)
        sig = np.nan_to_num(np.asarray(sig, float))
        notes = []
        if len(sig) > MAX_HOURS * 3600 * fs:
            sig = sig[: int(MAX_HOURS * 3600 * fs)]
            notes.append(f"longer than {MAX_HOURS} h: only the first {MAX_HOURS} h were analyzed")
        dur = len(sig) / fs
        if dur < MIN_SECONDS:
            raise ValueError(f"The recording is {dur:.1f} s long; at least {MIN_SECONDS} s are needed.")
        if dur < 30:
            notes.append("shorter than 30 s: the rhythm was read from one padded window")
        inverted, scale, scale_morph, n2 = check_signal(sig, fs)
        notes += n2
        if inverted and placement == "limb":
            notes.append("the ECG-shape screening used the signal as recorded: in a correctly placed lead II the R waves "
                         "point up, so check the electrodes before trusting it")
        sig_rr = (-sig if inverted else sig) * scale          # beats and rhythm
        sig_morph = sig * scale_morph                          # morphology: polarity as recorded
        progress(0.02, "Preparing the signal")
        sig360 = resample(sig_rr, fs, FS_BEAT)
        peaks360 = self._peaks(sig360, progress)
        progress(0.37, "Classifying every heartbeat")
        bp, bprob = self._beats(sig360, peaks360)
        progress(0.45, "Reading the rhythm (30-s windows)")
        peaks_native = np.round(peaks360 * fs / FS_BEAT).astype(int)
        w_start, w_prob = self._rhythm(sig_rr, fs, peaks_native)
        progress(0.68, "Checking signal quality and ECG shape (10-s strips)")
        morph = self._morphology(resample(sig_morph, fs, FS_MORPH), progress, classify=placement == "limb")
        res = dict(fs=fs, dur=dur, start_time=start_time, notes=notes, placement=placement, morph_run=placement == "limb", inverted=inverted, scale=scale,
                   beat_t=bp / FS_BEAT, beat_label=np.array(BEAT_CLASSES)[bprob.argmax(1)] if len(bp) else np.array([], str),
                   beat_prob=bprob, peaks_t=peaks360 / FS_BEAT,
                   win_start=w_start, win_end=np.r_[w_start[1:], dur] if len(w_start) else w_start, win_prob=w_prob,
                   win_class=np.array(RHYTHM_CLASSES)[w_prob.argmax(1)], win_class_names=list(RHYTHM_CLASSES), strip_est=morph["est"], strip_ok=morph["ok"],
                   strip_p5=morph["p5"], strip_p23=morph["p23"], thr5=self.thr5, thr23=self.thr23,
                   names23=self.names23, reliability=self.reliability, gate_threshold=float(self.gate["threshold"]),
                   signal=_bandpass(sig_rr, fs).astype(np.float32), signal_fs=fs, seconds=time.time() - t_all)
        progress(1.0, "Done")
        return res

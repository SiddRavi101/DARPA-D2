"""
Accelerometry classifier (new — SecondLife had no IMU support).

Detects, from a body- or medic-worn accelerometer:
  * cpr              — chest compressions: strongly periodic at 100-120/min (1.6-2.0 Hz;
                       we accept 1.4-2.5 Hz to tolerate real-world rate drift)
  * movement         — casualty carry / litter / evacuation: sustained aperiodic energy
  * stationary       — at rest / stabilised

Deterministic signal processing, no training data needed, so it works on day
one and its decisions are explainable. `window_features` is exposed so a
learned model (e.g. gradient boosting on DARPA's labelled streams) can be
dropped in later behind the same `detect_events` interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FS = 50.0          # resample rate (Hz)
WIN_S = 5.0        # analysis window
HOP_S = 1.0
CPR_BAND = (1.4, 2.5)
G = 9.80665


@dataclass
class ImuEvent:
    label: str
    start: float
    end: float
    confidence: float
    rate_per_min: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.end - self.start, 1)
        return d


def _to_g(xyz: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(xyz, axis=1)
    return xyz / G if np.median(mag) > 4.0 else xyz  # m/s² → g


def _resample(t: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t, xyz = t[order], xyz[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, xyz = t[keep], xyz[keep]
    tu = np.arange(t[0], t[-1], 1.0 / FS)
    return tu, np.stack([np.interp(tu, t, xyz[:, k]) for k in range(3)], 1)


def _dynamic(xyz_g: np.ndarray) -> np.ndarray:
    """Remove gravity with a 1 s moving-average high-pass; return dynamic magnitude along main axis."""
    k = int(FS)
    kernel = np.ones(k) / k
    pad = np.pad(xyz_g, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")  # no zero-pad edge spikes
    grav = np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(3)], 1)
    dyn = xyz_g - grav
    # project on principal axis: compressions are 1-D, keeps sign → clean periodicity
    if len(dyn) > 3:
        _, _, vt = np.linalg.svd(dyn - dyn.mean(0), full_matrices=False)
        return dyn @ vt[0]
    return np.linalg.norm(dyn, axis=1)


def window_features(sig: np.ndarray) -> dict:
    sig = sig - sig.mean()
    rms = float(np.sqrt(np.mean(sig ** 2)))
    n = len(sig)
    spec = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / FS)
    band = (freqs >= 0.5) & (freqs <= 6.0)
    peak_f = float(freqs[band][np.argmax(spec[band])]) if band.any() else 0.0
    cpr_band = (freqs >= CPR_BAND[0]) & (freqs <= CPR_BAND[1])
    band_ratio = float(spec[cpr_band].sum() / (spec[band].sum() + 1e-12))
    # autocorrelation periodicity at the dominant period
    ac = np.correlate(sig, sig, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag = int(round(FS / peak_f)) if peak_f > 0 else 0
    periodicity = float(ac[lag]) if 0 < lag < n else 0.0
    return {"rms_g": rms, "peak_hz": peak_f, "cpr_band_ratio": band_ratio, "periodicity": periodicity}


def classify_window(f: dict) -> tuple[str, float]:
    if (f["rms_g"] > 0.08 and CPR_BAND[0] <= f["peak_hz"] <= CPR_BAND[1]
            and f["periodicity"] > 0.45 and f["cpr_band_ratio"] > 0.45):
        conf = min(1.0, 0.5 * f["periodicity"] + 0.5 * f["cpr_band_ratio"])
        return "cpr", conf
    if f["rms_g"] > 0.04:
        return "movement", min(1.0, f["rms_g"] / 0.15)
    return "stationary", min(1.0, 1 - f["rms_g"] / 0.04)


def detect_events(t: np.ndarray, xyz: np.ndarray, offset: float = 0.0,
                  min_duration: dict | None = None) -> list[ImuEvent]:
    """Window → classify → merge consecutive same-label windows → drop blips."""
    if len(t) < FS * WIN_S / 2:
        return []
    min_duration = min_duration or {"cpr": 6.0, "movement": 5.0, "stationary": 10.0}
    tu, xyz_u = _resample(np.asarray(t, float), np.asarray(xyz, float))
    sig = _dynamic(_to_g(xyz_u))
    win, hop = int(WIN_S * FS), int(HOP_S * FS)
    windows = []
    for a in range(0, max(1, len(sig) - win + 1), hop):
        f = window_features(sig[a:a + win])
        lab, conf = classify_window(f)
        windows.append((tu[a], tu[min(a + win, len(tu) - 1)], lab, conf, f["peak_hz"]))

    events: list[ImuEvent] = []
    for s, e, lab, conf, hz in windows:
        if events and events[-1].label == lab and s <= events[-1].end:
            ev = events[-1]
            n = getattr(ev, "_n", 1)
            ev.confidence = (ev.confidence * n + conf) / (n + 1)
            if lab == "cpr":
                ev.rate_per_min = (ev.rate_per_min * n + hz * 60) / (n + 1)
            ev._n = n + 1  # type: ignore[attr-defined]
            ev.end = e
        else:
            ev = ImuEvent(lab, s, e, conf, hz * 60 if lab == "cpr" else None)
            ev._n = 1  # type: ignore[attr-defined]
            events.append(ev)
    out = []
    for ev in events:
        if ev.end - ev.start >= min_duration.get(ev.label, 0):
            ev.start += offset
            ev.end += offset
            ev.confidence = round(float(ev.confidence), 3)
            if ev.rate_per_min is not None:
                ev.rate_per_min = round(float(ev.rate_per_min), 1)
            out.append(ev)
    return out

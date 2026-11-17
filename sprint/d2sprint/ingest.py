"""
Batch ingestion of a pre-recorded multi-sensor scenario (replaces SecondLife's
live Android AudioRecord / CameraX capture).

A scenario is a directory. We discover files by extension, so the exact DARPA
naming convention (not yet published for the final event) only needs a
mapping tweak here:

    scenario/
      *.wav|.flac|.mp3|.m4a           audio track(s)
      *.mp4|.mov|.mkv|.avi            video (audio is extracted if no audio file)
      *accel*.csv|*imu*.csv           accelerometry: time + x,y,z columns
      *.srt|.vtt|transcript*.json     optional pre-made transcript (skips ASR)
      metadata.json                   optional {"start_time": ISO8601, "offsets": {file: seconds}}

All timestamps downstream are seconds from scenario start (float).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
TRANSCRIPT_EXTS = {".srt", ".vtt"}
SAMPLE_RATE = 16_000  # SecondLife audio contract: PCM16 mono 16 kHz


@dataclass
class Scenario:
    root: Path
    audio: list[Path] = field(default_factory=list)
    video: list[Path] = field(default_factory=list)
    accel: list[Path] = field(default_factory=list)
    transcripts: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        return self.metadata.get("scenario_id", self.root.name)

    def offset(self, path: Path) -> float:
        return float(self.metadata.get("offsets", {}).get(path.name, 0.0))


def discover(root: str | Path) -> Scenario:
    root = Path(root)
    sc = Scenario(root=root)
    meta = root / "metadata.json"
    if meta.exists():
        sc.metadata = json.loads(meta.read_text(encoding="utf-8"))
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suf, low = p.suffix.lower(), p.name.lower()
        if suf in AUDIO_EXTS:
            sc.audio.append(p)
        elif suf in VIDEO_EXTS:
            sc.video.append(p)
        elif suf == ".csv" and any(k in low for k in ("accel", "imu", "acc_", "motion")):
            sc.accel.append(p)
        elif suf in TRANSCRIPT_EXTS or (suf == ".json" and "transcript" in low):
            sc.transcripts.append(p)
    return sc


def discover_many(input_dir: str | Path) -> list[Scenario]:
    """Input may be one scenario or a directory of scenario sub-directories."""
    input_dir = Path(input_dir)
    subs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()] if input_dir.is_dir() else []
    scenarios = [discover(d) for d in subs]
    scenarios = [s for s in scenarios if s.audio or s.video or s.accel or s.transcripts]
    return scenarios or [discover(input_dir)]


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_audio(path: Path) -> np.ndarray:
    """Return float32 mono 16 kHz. Uses soundfile for wav/flac, ffmpeg for everything else."""
    if path.suffix.lower() in {".wav", ".flac"}:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return _resample(data.mean(axis=1), sr, SAMPLE_RATE)
        except ImportError:
            if path.suffix.lower() == ".wav":
                return _read_wav_stdlib(path)
    return _ffmpeg_decode(path)


def _read_wav_stdlib(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128) / 128 if width == 1 else x / float(np.iinfo(dtype).max)
    x = x.reshape(-1, ch).mean(axis=1)
    return _resample(x, sr, SAMPLE_RATE)


def _ffmpeg_decode(path: Path) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"ffmpeg required to decode {path}")
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"], capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def scenario_audio(sc: Scenario) -> Optional[tuple[np.ndarray, float, Path]]:
    """Primary audio track (+ its offset). Falls back to the first video's soundtrack."""
    for p in sc.audio + sc.video:
        try:
            return load_audio(p), sc.offset(p), p
        except Exception:
            continue
    return None


# ── Video frames (for future vision models; 768x768 SecondLife contract) ─────

def sample_frames(path: Path, every_s: float = 5.0, size: int = 768):
    """Yield (t_seconds, PIL.Image 768x768) using OpenCV if available."""
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every_s)))
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((size, size))
                canvas = Image.new("RGB", (size, size))
                canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
                yield idx / fps, canvas
        idx += 1
    cap.release()


# ── Accelerometry ─────────────────────────────────────────────────────────────

_T_COLS = ("t", "time", "timestamp", "time_s", "seconds", "elapsed", "ts")
_AXES = (("x", "ax", "acc_x", "accel_x", "accx"), ("y", "ay", "acc_y", "accel_y", "accy"),
         ("z", "az", "acc_z", "accel_z", "accz"))


def load_accel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """→ (t seconds from file start, xyz [N,3]). Accepts seconds, ms, or ISO timestamps."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.zeros(0), np.zeros((0, 3))
    cols = {c.strip().lower(): c for c in rows[0]}
    tcol = next((cols[c] for c in _T_COLS if c in cols), None)
    axes = []
    for names in _AXES:
        c = next((cols[n] for n in names if n in cols), None)
        if c is None:
            raise ValueError(f"{path}: missing accel axis {names[0]} in {list(rows[0])}")
        axes.append(c)
    xyz = np.array([[float(r[a]) for a in axes] for r in rows], dtype=np.float64)
    if tcol is None:
        raise ValueError(f"{path}: no time column")
    try:
        t = np.array([float(r[tcol]) for r in rows])
    except ValueError:
        from datetime import datetime

        t = np.array([datetime.fromisoformat(r[tcol].replace("Z", "+00:00")).timestamp() for r in rows])
    t = t - t[0]
    if len(t) > 1 and np.median(np.diff(t)) > 1.0:  # looks like milliseconds
        t = t / 1000.0
    return t, xyz

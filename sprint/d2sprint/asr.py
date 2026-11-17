"""
Offline speech-to-text + lightweight speaker diarization.

Replaces SecondLife's live push-to-talk capture: the whole 20-minute scenario
track is transcribed once, with segment timestamps, so downstream extraction
and RAG run continuously along the timeline.

Backends:
  * faster-whisper (CTranslate2) — model dir baked into the container at
    $D2_WHISPER_MODEL (default /app/models/whisper); runs with --network none.
  * pre-made transcripts (.srt / .vtt / transcript*.json) — used if present,
    which also makes the pipeline testable without a GPU.

Diarization: medics/casualty/team voices are separated with a cheap spectral
embedding + agglomerative clustering. It is good enough to tag "who said the
vitals"; swap in pyannote if/when the license and container size allow.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .ingest import SAMPLE_RATE

log = logging.getLogger("d2sprint.asr")

# Domain vocabulary biases Whisper toward TCCC terms it otherwise mangles.
TCCC_PROMPT = (
    "TCCC MARCH. Tourniquet, CAT, junctional, hemostatic gauze, Combat Gauze, pressure dressing, "
    "NPA, nasopharyngeal airway, cric, chest seal, needle decompression, NDC, tension pneumothorax, "
    "TXA, tranexamic acid, IV, IO, whole blood, ketamine, fentanyl, OTFC, moxifloxacin, ertapenem, "
    "hypothermia, HPMK, blizzard blanket, AVPU, SpO2, GSW, IED, blast, amputation, 9-line, MEDEVAC."
)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Transcript files ─────────────────────────────────────────────────────────

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp {s!r}")
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def load_transcript(path: Path, offset: float = 0.0) -> list[Segment]:
    suf = path.suffix.lower()
    segs: list[Segment] = []
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("segments", data) if isinstance(data, dict) else data
        for it in items:
            segs.append(Segment(float(it["start"]) + offset, float(it.get("end", it["start"])) + offset,
                                str(it["text"]).strip(), str(it.get("speaker", "unknown"))))
        return segs
    # SRT / VTT
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").replace("\r", ""))
    for b in blocks:
        lines = [l for l in b.strip().split("\n") if l.strip() and l.strip() != "WEBVTT"]
        arrow = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if arrow is None:
            continue
        a, z = lines[arrow].split("-->")
        text = " ".join(lines[arrow + 1:]).strip()
        speaker = "unknown"
        m = re.match(r"^(?:<v\s+([^>]+)>|\[?([A-Za-z0-9_ ]{1,24})\]?:)\s*(.*)$", text)
        if m:
            speaker = (m.group(1) or m.group(2)).strip()
            text = m.group(3)
        text = re.sub(r"<[^>]+>", "", text)
        segs.append(Segment(_ts(a) + offset, _ts(z) + offset, text, speaker))
    return segs


# ── faster-whisper ───────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray, offset: float = 0.0, model_dir: Optional[str] = None,
               device: Optional[str] = None) -> list[Segment]:
    from faster_whisper import WhisperModel

    model_dir = model_dir or os.environ.get("D2_WHISPER_MODEL", "/app/models/whisper")
    if device is None:
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    model = WhisperModel(model_dir, device=device, compute_type="float16" if device == "cuda" else "int8",
                         local_files_only=True)
    segments, info = model.transcribe(
        audio, language="en", beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        initial_prompt=TCCC_PROMPT, condition_on_previous_text=False,
    )
    out = [Segment(s.start + offset, s.end + offset, s.text.strip()) for s in segments if s.text.strip()]
    log.info("ASR: %d segments, %.0fs audio", len(out), info.duration)
    return out


# ── Diarization ──────────────────────────────────────────────────────────────

def _segment_embedding(x: np.ndarray, n_bands: int = 40) -> np.ndarray:
    """Mean + std of log band energies over 25 ms frames — a crude speaker fingerprint."""
    win, hop = 400, 160
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * np.hanning(win)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    edges = np.geomspace(2, spec.shape[1] - 1, n_bands + 1).astype(int)
    bands = np.stack([spec[:, a:max(a + 1, b)].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    logb = np.log(bands + 1e-9)
    logb -= logb.mean(1, keepdims=True)  # remove loudness
    return np.concatenate([logb.mean(0), logb.std(0)])


def diarize(audio: np.ndarray, segments: list[Segment], offset: float = 0.0,
            max_speakers: int = 4, threshold: float = 0.35) -> list[Segment]:
    """Assign speaker_N labels in place to segments whose speaker is unknown."""
    todo = [s for s in segments if s.speaker == "unknown" and s.end - s.start >= 0.3]
    if len(todo) < 2:
        for s in todo:
            s.speaker = "speaker_0"
        return segments
    from scipy.cluster.hierarchy import fcluster, linkage

    embs = []
    for s in todo:
        a, b = int((s.start - offset) * SAMPLE_RATE), int((s.end - offset) * SAMPLE_RATE)
        embs.append(_segment_embedding(audio[max(0, a):max(a + 1, b)]))
    E = np.array(embs)
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    Z = linkage(E, method="average", metric="cosine")
    lab = fcluster(Z, t=threshold, criterion="distance")
    if lab.max() > max_speakers:
        lab = fcluster(Z, t=max_speakers, criterion="maxclust")
    # Name clusters by first appearance for stable output
    order: dict[int, str] = {}
    for s, l in zip(todo, lab):
        order.setdefault(int(l), f"speaker_{len(order)}")
        s.speaker = order[int(l)]
    return segments


def transcribe_scenario(sc, audio_bundle=None) -> list[Segment]:
    """Prefer provided transcripts; else run faster-whisper + diarization on the main track."""
    if sc.transcripts:
        segs = []
        for p in sc.transcripts:
            segs.extend(load_transcript(p, sc.offset(p)))
        return sorted(segs, key=lambda s: s.start)
    if audio_bundle is None:
        return []
    audio, offset, path = audio_bundle
    try:
        segs = transcribe(audio, offset)
    except Exception as e:
        log.error("ASR unavailable (%s) — continuing with sensors only", e)
        return []
    try:
        diarize(audio, segs, offset)
    except Exception as e:
        log.warning("diarization failed: %s", e)
    return segs

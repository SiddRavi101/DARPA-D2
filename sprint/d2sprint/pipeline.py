"""
Scenario → timeline → Lane 1 (DD1380) and/or Lane 2 (decision support).

    audio/video ─► ASR + diarization ─► transcript segments ─► extract ─┐
    accelerometry ─► IMU classifier ─► cpr / movement / stationary ─────┤
                                                                         ▼
                                    time-ordered event stream ─► DD1380Card.apply()
                                                                         │ changed?
                                                                         ▼
                                              DecisionSupport.on_update() (+RAG cites)

Outputs per scenario (in <output>/<scenario_id>/ when several scenarios are given):
    dd1380.json            final card + per-change timeline snapshots (Lane 1)
    decision_support.json  decision points with prioritized MARCH recommendations (Lane 2)
    transcript.json        diarized ASR segments
    imu_events.json        accelerometry events
    audit_log.jsonl        hash-chained trail of every change
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

from .asr import Segment, transcribe_scenario
from .audit import AuditLog
from .dd1380 import DD1380Card
from .decision import DecisionSupport
from .extract import Event, extract_all
from .imu import detect_events
from .ingest import Scenario, discover_many, load_accel, scenario_audio
from .rag import ProtocolRetriever

log = logging.getLogger("d2sprint")


def imu_to_events(sc: Scenario) -> tuple[list[Event], list[dict]]:
    events, raw = [], []
    for p in sc.accel:
        try:
            t, xyz = load_accel(p)
            for e in detect_events(t, xyz, sc.offset(p)):
                d = e.to_dict()
                d["file"] = p.name
                raw.append(d)
                if e.label in ("cpr", "movement"):
                    desc = f"accelerometry: {e.label}" + (f" ~{e.rate_per_min:.0f}/min" if e.rate_per_min else "")
                    events.append(Event("imu", e.start, {"label": e.label, "end": e.end,
                                                         "rate_per_min": e.rate_per_min,
                                                         "confidence": e.confidence}, desc, "imu"))
        except Exception as ex:
            log.error("accelerometry %s failed: %s", p.name, ex)
    return events, raw


def build_timeline(segments: list[Segment], imu_events: list[Event]) -> list[Event]:
    order = {"mechanism": 0, "injury": 1, "vitals": 2, "evac": 3, "intervention": 4,
             "medication": 5, "fluid": 6, "imu": 7}
    evs = extract_all(segments) + imu_events
    return sorted(evs, key=lambda e: (e.t, order.get(e.kind, 9)))


def process_scenario(sc: Scenario, out_dir: Path, lanes: Iterable[str],
                     retriever: ProtocolRetriever | None, llm=None) -> dict:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes = set(lanes)

    audio = None if sc.transcripts else scenario_audio(sc)
    segments = transcribe_scenario(sc, audio)
    imu_events, imu_raw = imu_to_events(sc)
    timeline = build_timeline(segments, imu_events)
    log.info("[%s] %d segments, %d imu events, %d timeline events",
             sc.scenario_id, len(segments), len(imu_raw), len(timeline))

    card = DD1380Card(sc.scenario_id)
    ds = DecisionSupport(retriever, llm) if "decision_support" in lanes else None
    audit = AuditLog(out_dir / "audit_log.jsonl")
    snapshots = []
    for ev in timeline:
        if card.apply(ev):
            audit.log("card_update", ev.t, {"event": ev.to_dict()})
            snapshots.append({"t": round(ev.t, 2), "trigger": ev.kind, "evidence": ev.evidence[:200]})
            if ds is not None:
                point = ds.on_update(card, ev)
                if point:
                    audit.log("decision_point", ev.t, {"next_action": point["next_action"], "new": point["new"]})

    _dump(out_dir / "transcript.json", [s.to_dict() for s in segments])
    _dump(out_dir / "imu_events.json", imu_raw)
    result = {"scenario_id": sc.scenario_id, "runtime_s": None}
    if "documentation" in lanes:
        doc = card.to_output()
        doc["change_log"] = snapshots
        _dump(out_dir / "dd1380.json", doc)
    if ds is not None:
        _dump(out_dir / "decision_support.json", {
            "scenario_id": sc.scenario_id,
            "framework": "CoTCCC MARCH-PAWS",
            "decision_points": ds.points,
            "final_recommendations": ds.points[-1]["recommendations"] if ds.points else [],
        })
    result["runtime_s"] = round(time.monotonic() - t0, 2)
    result["audit_chain_valid"] = AuditLog.verify(out_dir / "audit_log.jsonl")
    return result


def _dump(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)


def run(input_dir: str | Path, output_dir: str | Path, lanes: Iterable[str]) -> list[dict]:
    from .llm import GemmaAdvisor

    output_dir = Path(output_dir)
    scenarios = discover_many(input_dir)
    try:
        retriever = ProtocolRetriever()
    except Exception as e:
        log.error("RAG disabled: %s", e)
        retriever = None
    llm = GemmaAdvisor.try_load()
    results = []
    for sc in scenarios:
        sub = output_dir / sc.scenario_id if len(scenarios) > 1 else output_dir
        try:
            results.append(process_scenario(sc, sub, lanes, retriever, llm))
        except Exception as e:
            log.exception("scenario %s failed: %s", sc.scenario_id, e)
            results.append({"scenario_id": sc.scenario_id, "error": str(e)})
    _dump(output_dir / "run_summary.json", results)
    return results

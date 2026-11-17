import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from d2sprint.asr import Segment, load_transcript  # noqa: E402
from d2sprint.audit import AuditLog  # noqa: E402
from d2sprint.dd1380 import DD1380Card  # noqa: E402
from d2sprint.decision import DecisionSupport, evaluate  # noqa: E402
from d2sprint.extract import extract, words_to_numbers  # noqa: E402
from d2sprint.imu import detect_events  # noqa: E402
from d2sprint.rag import ProtocolRetriever  # noqa: E402


def seg(t, text):
    return Segment(t, t + 3, text, "medic")


def kinds(evs):
    return {(e.kind, e.data.get("name") or e.data.get("type") or e.data.get("mechanism")) for e in evs}


# ── extraction ────────────────────────────────────────────────────────────────

def test_words_to_numbers():
    assert words_to_numbers("pulse one thirty") == "pulse 130"
    assert words_to_numbers("sats ninety eight") == "sats 98"
    assert words_to_numbers("bp one hundred over sixty") == "bp 100 over 60"


def test_extract_tq_with_site_and_time():
    evs = extract(seg(10, "Tourniquet going on right leg high and tight, time 1432."))
    tq = [e for e in evs if e.kind == "intervention"][0]
    assert tq.data["name"] == "extremity_tq" and tq.data["sites"] == ["RLE"]
    assert tq.data["stated_time"] == "1432"


def test_negation_and_planning_are_not_interventions():
    assert not [e for e in extract(seg(0, "He does not need a tourniquet")) if e.kind == "intervention"]
    assert not [e for e in extract(seg(0, "Get me a chest seal")) if e.kind == "intervention"]


def test_extract_vitals_spoken():
    v = [e for e in extract(seg(0, "Vitals: pulse one thirty radial weak, resps twenty eight, sats eighty eight."))
         if e.kind == "vitals"][0].data
    assert v["pulse_rate"] == 130 and v["resp_rate"] == 28 and v["spo2"] == 88
    assert v["pulse_location"] == "radial" and v["pulse_quality"] == "weak/absent"


def test_extract_med_dose_route():
    m = [e for e in extract(seg(0, "Giving TXA two grams IV slow push")) if e.kind == "medication"][0].data
    assert m["name"] == "TXA" and m["dose"] == "2 g" and m["route"] == "IV"


# ── DD1380 card ───────────────────────────────────────────────────────────────

def test_card_dedups_repeated_mentions():
    card = DD1380Card("s")
    for t in (10, 40, 90):
        for e in extract(seg(t, "Tourniquet is on the right leg")):
            card.apply(e)
    assert card.card["tourniquets"]["RLE"]["t"] == 10
    assert len(card.card["treatments"]["c_hemorrhage"]["extremity_tq"]) == 1
    for t in (100, 130):
        for e in extract(seg(t, "Giving TXA 2 grams IV")):
            card.apply(e)
    assert len(card.card["treatments"]["medications"]["other"]) == 1


def test_card_severity_upgrade():
    card = DD1380Card("s")
    for text in ("bleeding from the left leg", "left leg is amputated"):
        for e in extract(seg(0, text)):
            card.apply(e)
    assert [(i["site"], i["type"]) for i in card.card["injuries"]] == [("LLE", "amputation")]


# ── IMU ───────────────────────────────────────────────────────────────────────

def _accel(segments, fs=50):
    t = np.arange(0, sum(d for d, _ in segments), 1 / fs)
    xyz = np.zeros((len(t), 3))
    xyz[:, 2] = 9.81
    rng = np.random.default_rng(1)
    xyz += rng.normal(0, 0.05, xyz.shape)
    a = 0
    for dur, kind in segments:
        m = (t >= a) & (t < a + dur)
        if kind == "cpr":
            xyz[m, 2] += 6 * np.sin(2 * np.pi * (110 / 60) * t[m])
        elif kind == "move":
            xyz[m] += rng.normal(0, 2.0, (m.sum(), 3))
        a += dur
    return t, xyz


def test_imu_detects_cpr_rate_and_movement():
    t, xyz = _accel([(30, "rest"), (40, "move"), (10, "rest"), (60, "cpr"), (30, "rest")])
    evs = detect_events(t, xyz)
    labels = [e.label for e in evs]
    assert "cpr" in labels and "movement" in labels
    cpr = next(e for e in evs if e.label == "cpr")
    assert 100 <= cpr.rate_per_min <= 120
    assert 75 <= cpr.start <= 90 and 135 <= cpr.end <= 150


def test_imu_rest_only_has_no_cpr():
    t, xyz = _accel([(120, "rest")])
    assert all(e.label == "stationary" for e in detect_events(t, xyz))


# ── decision support ──────────────────────────────────────────────────────────

def test_march_priorities():
    card = DD1380Card("s")
    for text in ("right leg amputation, arterial bleeding", "pulse 130 radial weak, sats 85, resps 32",
                 "shrapnel wound to the left chest"):
        for e in extract(seg(0, text)):
            card.apply(e)
    recs = evaluate(card, 10)
    ids = [r.id for r in recs]
    assert ids[0] == "M-tq-RLE"
    assert {"R-seal", "C-txa", "C-access", "C-blood"} <= set(ids)
    assert ids.index("M-tq-RLE") < ids.index("R-seal") < ids.index("C-txa")


def test_rag_finds_tccc_protocols():
    r = ProtocolRetriever(use_dense=False)
    hit = r.retrieve("tranexamic acid TXA shock amputation", top_k=1, march="C")[0]
    assert "TXA" in hit["text"] and hit["citation"]


# ── end to end ────────────────────────────────────────────────────────────────

def test_end_to_end_demo(tmp_path):
    import make_demo
    from d2sprint.__main__ import main

    scen = tmp_path / "demo"
    make_demo.main(str(scen))
    out = tmp_path / "out"
    assert main(["both", "--input", str(scen), "--output", str(out)]) == 0

    card = json.loads((out / "dd1380.json").read_text(encoding="utf-8"))["card"]
    assert card["tourniquets"]["RLE"] is not None
    assert len(card["treatments"]["c_hemorrhage"]["extremity_tq"]) == 1  # said 3x, stored once
    assert card["treatments"]["b_breathing"]["chest_seal"]
    assert card["treatments"]["c_circulation"]["iv_io_access"]
    assert [m["name"] for m in card["treatments"]["medications"]["other"]] == ["TXA"]
    assert card["treatments"]["other"]["hypothermia_prevention"]
    assert card["treatments"]["other"]["cpr"], "CPR from accelerometry/speech"
    assert card["casualty"]["evac_category"] == "Urgent Surgical"
    assert {m["value"] for m in card["mechanism_of_injury"]} == {"IED"}

    ds = json.loads((out / "decision_support.json").read_text(encoding="utf-8"))
    first = ds["decision_points"][0]
    assert ds["decision_points"]
    tq_point = next(p for p in ds["decision_points"] if "M-tq-RLE" in p["new"])
    assert tq_point["next_action"].startswith("Apply limb tourniquet to RLE")
    assert all(r["citation"] for p in ds["decision_points"] for r in p["recommendations"])
    final_ids = {r["id"] for r in ds["final_recommendations"]}
    assert "M-tq-RLE" not in final_ids and "C-txa" not in final_ids  # done → resolved
    assert AuditLog.verify(out / "audit_log.jsonl")
    assert first["t"] <= tq_point["t"]

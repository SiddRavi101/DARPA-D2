"""
Lane 2 — MARCH-PAWS decision support.

At each decision point (a new finding, intervention, vitals row or IMU event
that changed the casualty card) the rules engine compares the current
casualty state against CoTCCC priorities and emits a prioritized list of
interventions still outstanding, each with a rationale and a protocol
citation retrieved by the SecondLife RAG (now MARCH-tagged).

Design choices:
  * Rules decide WHAT to recommend (deterministic, auditable, testable).
    RAG supplies the WHY/citation. The optional LLM (llm.py) only rewords
    — it can never add or drop an intervention.
  * A recommendation is re-emitted only when still unmet AND its trigger
    changed, so medics are not spammed every few seconds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .dd1380 import LIMB_SITES, DD1380Card, fmt_clock
from .extract import Event
from .rag import ProtocolRetriever

MARCH_ORDER = {"M": 0, "A": 1, "R": 2, "C": 3, "H": 4, "PAWS": 5}
JUNCTIONAL = {"groin", "axilla", "neck", "R_groin", "L_groin", "R_axilla", "L_axilla", "pelvis", "buttock"}
CHEST = {"chest", "R_chest", "L_chest", "back"}


@dataclass
class Recommendation:
    id: str
    march: str
    action: str
    rationale: str
    urgency: str = "immediate"   # immediate | soon | monitor
    citation: str = ""
    protocol_excerpt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _shock(v: dict, card: DD1380Card) -> bool:
    if v.get("pulse_quality") == "weak/absent":
        return True
    if v.get("pulse_rate", 0) >= 120:
        return True
    if "bp" in v:
        try:
            if int(v["bp"].split("/")[0]) < 90:
                return True
        except ValueError:
            pass
    head = any(i["type"] == "head_injury" for i in card.card["injuries"])
    return v.get("avpu") in ("V", "P", "U") and not head


def evaluate(card: DD1380Card, now: float, imu_recent: Optional[dict] = None) -> list[Recommendation]:
    c = card.card
    tx = c["treatments"]
    v = card.latest_vitals()
    injuries = c["injuries"]
    recs: list[Recommendation] = []

    # ── M: massive hemorrhage ──────────────────────────────────────────────
    for inj in injuries:
        site = inj["site"]
        if inj["type"] in ("amputation", "massive_hemorrhage") and site in LIMB_SITES and not card.has_tq(site):
            recs.append(Recommendation(
                f"M-tq-{site}", "M",
                f"Apply limb tourniquet to {site} high and tight; tighten until bleeding stops and distal pulse is gone. Record time.",
                f"{inj['type'].replace('_', ' ')} at {site} without a documented tourniquet."))
        if inj["type"] in ("amputation", "massive_hemorrhage", "bleeding") and site in JUNCTIONAL \
                and not tx["c_hemorrhage"]["hemostatic_dressing"] and not tx["c_hemorrhage"]["junctional_tq"]:
            recs.append(Recommendation(
                f"M-junctional-{site}", "M",
                f"Pack {site} wound with hemostatic gauze, hold ≥3 min direct pressure, apply pressure bandage; junctional tourniquet if available.",
                "Junctional bleeding is not tourniquet-amenable."))
    for site in LIMB_SITES:
        tq = c["tourniquets"][site]
        if tq is None:
            continue
        if imu_recent and imu_recent.get("label") == "movement" and imu_recent["t"] > tq["t"]:
            recs.append(Recommendation(
                f"M-reassess-{site}", "M",
                f"Reassess {site} tourniquet after casualty movement: confirm bleeding controlled and distal pulse absent.",
                "Tourniquets loosen during carries/litter moves.", "soon"))
        elapsed = now - tq["t"]
        if elapsed >= 2 * 3600 - 600:
            recs.append(Recommendation(
                f"M-convert-{site}", "M",
                f"{site} tourniquet on {fmt_clock(elapsed)}: consider conversion to hemostatic/pressure dressing if not in shock, monitorable, and not an amputation.",
                "CoTCCC: attempt conversion within 2 h when criteria are met.", "soon"))

    # ── A: airway ─────────────────────────────────────────────────────────
    has_airway = any(tx["a_airway"][k] for k in ("npa", "cric", "et_tube", "sga", "recovery_position"))
    if v.get("avpu") in ("P", "U") and not has_airway:
        recs.append(Recommendation("A-npa", "A",
                                   "Insert nasopharyngeal airway and place in recovery position.",
                                   f"Altered mental status (AVPU={v['avpu']}) with no airway adjunct documented."))
    if any(i["type"] == "burn" and i["site"] in ("face", "head", "neck") for i in injuries) and not tx["a_airway"]["cric"]:
        recs.append(Recommendation("A-burn-airway", "A",
                                   "Facial/neck burn: monitor airway closely; prepare for surgical cricothyroidotomy if distress develops.",
                                   "Inhalation injury risk.", "soon"))

    # ── R: respiration ────────────────────────────────────────────────────
    chest_inj = [i for i in injuries if i["site"] in CHEST or i["type"] == "open_chest_wound"]
    if chest_inj and not tx["b_breathing"]["chest_seal"]:
        recs.append(Recommendation("R-seal", "R", "Apply vented chest seal to penetrating chest wound(s).",
                                   "Penetrating torso injury without chest seal."))
    resp_distress = v.get("spo2", 100) < 90 or v.get("resp_rate", 16) >= 30 or v.get("resp_rate", 16) <= 8
    if (chest_inj or any(m["value"] == "IED" for m in c["mechanism_of_injury"])) and resp_distress \
            and not tx["b_breathing"]["needle_d"]:
        recs.append(Recommendation("R-ndc", "R",
                                   "Suspected tension pneumothorax: needle decompression 14G 3.25\" at 5th ICS AAL (or 2nd ICS MCL) on the injured side.",
                                   f"Torso/blast trauma with respiratory compromise (SpO2={v.get('spo2')}, RR={v.get('resp_rate')})."))

    # ── C: circulation ────────────────────────────────────────────────────
    major = any(i["type"] in ("amputation", "massive_hemorrhage") for i in injuries) or \
        any(i["site"] in CHEST | {"abdomen"} and i["type"] in ("penetrating_wound", "open_chest_wound") for i in injuries)
    shock = _shock(v, card)
    if (shock or major) and not tx["c_circulation"]["iv_io_access"]:
        recs.append(Recommendation("C-access", "C", "Establish IV or IO access.",
                                   "Shock signs or likely transfusion requirement."))
    if (shock or major or any(i["type"] == "head_injury" for i in injuries)) and not card.med_given("TXA"):
        recs.append(Recommendation("C-txa", "C", "Give TXA 2 g slow IV/IO push as soon as possible (must be within 3 h of injury).",
                                   "Meets TXA criteria (shock / major amputation / penetrating torso / significant TBI)."))
    if shock and not tx["c_circulation"]["blood_products"]:
        recs.append(Recommendation("C-blood", "C",
                                   "Resuscitate with whole blood (preferred) in 1-unit increments to radial pulse / mentation / SBP ~100.",
                                   f"Hemorrhagic shock signs ({_fmt_v(v)})."))
    cpr = tx["other"]["cpr"]
    if cpr and (injuries or c["mechanism_of_injury"]) and not tx["b_breathing"]["needle_d"]:
        recs.append(Recommendation("C-cpr-ndc", "C",
                                   "Traumatic arrest: perform bilateral needle decompression before stopping resuscitation; reassess futility.",
                                   "CPR detected (accelerometry/speech) in a trauma casualty."))

    # ── H: hypothermia / head ─────────────────────────────────────────────
    if (injuries or shock) and not tx["other"]["hypothermia_prevention"]:
        recs.append(Recommendation("H-hypothermia", "H",
                                   "Hypothermia prevention: insulate from ground, active heating blanket + HPMK/blizzard wrap, cover head.",
                                   "Every trauma casualty, earlier if in shock.", "soon"))
    if any(i["type"] == "head_injury" for i in injuries):
        recs.append(Recommendation("H-tbi", "H",
                                   "TBI: keep SpO2 > 90% and SBP adequate, elevate head 30° if not in shock, serial AVPU/pupils.",
                                   "Head injury documented.", "soon"))

    # ── PAWS ──────────────────────────────────────────────────────────────
    if v.get("pain", 0) >= 4 and not tx["medications"]["analgesic"]:
        if shock:
            recs.append(Recommendation("P-ketamine", "PAWS", "Analgesia: ketamine 20-30 mg slow IV/IO (or 50-100 mg IM/IN).",
                                       f"Pain {v['pain']}/10 with shock signs → avoid opioids.", "soon"))
        else:
            recs.append(Recommendation("P-otfc", "PAWS", "Analgesia: OTFC 800 mcg (or ketamine if shock develops).",
                                       f"Pain {v['pain']}/10, not in shock.", "soon"))
    open_wound = any(i["type"] in ("penetrating_wound", "amputation", "laceration", "open_chest_wound", "burn")
                     for i in injuries)
    if open_wound and not tx["medications"]["antibiotic"]:
        po_ok = v.get("avpu", "A") == "A" and not shock
        recs.append(Recommendation("P-abx", "PAWS",
                                   "Antibiotics: moxifloxacin 400 mg PO." if po_ok else
                                   "Antibiotics: ertapenem 1 g IV/IM (unable to take PO).",
                                   "Open combat wound.", "monitor"))
    if c["injuries"] and not c["signs_and_symptoms"]:
        recs.append(Recommendation("P-vitals", "PAWS", "Obtain and record a full set of vitals (pulse, BP, RR, SpO2, AVPU, pain).",
                                   "No vitals documented yet.", "soon"))
    if any(i["type"] == "fracture" for i in injuries) and not tx["other"]["splint"]:
        recs.append(Recommendation("P-splint", "PAWS", "Splint suspected fracture; check distal pulse before and after.",
                                   "Fracture documented.", "monitor"))

    urg = {"immediate": 0, "soon": 1, "monitor": 2}
    recs.sort(key=lambda r: (urg[r.urgency], MARCH_ORDER[r.march]))
    return recs


def _fmt_v(v: dict) -> str:
    return ", ".join(f"{k}={val}" for k, val in v.items()) or "no vitals"


class DecisionSupport:
    def __init__(self, retriever: Optional[ProtocolRetriever] = None, llm=None):
        self.rag = retriever
        self.llm = llm
        self.points: list[dict] = []
        self._last_ids: dict[str, str] = {}
        self._imu_recent: Optional[dict] = None
        self._cite_cache: dict[str, tuple[str, str]] = {}

    def _cite(self, rec: Recommendation) -> None:
        if not self.rag:
            return
        key = rec.id.split("-")[0] + rec.id.split("-")[1]
        if key not in self._cite_cache:
            hits = self.rag.retrieve(f"{rec.action} {rec.rationale}", top_k=1, march=rec.march)
            self._cite_cache[key] = (hits[0]["citation"], hits[0]["text"][:400]) if hits else ("", "")
        rec.citation, rec.protocol_excerpt = self._cite_cache[key]

    def on_update(self, card: DD1380Card, ev: Event) -> Optional[dict]:
        if ev.kind == "imu":
            self._imu_recent = {"label": ev.data["label"], "t": ev.t}
        recs = evaluate(card, ev.t, self._imu_recent)
        current = {r.id: r.action for r in recs}
        new = [r for r in recs if self._last_ids.get(r.id) != r.action]
        resolved = sorted(set(self._last_ids) - set(current))
        self._last_ids = current
        if not new and not resolved:
            return None
        for r in recs:
            self._cite(r)
        if self.llm is not None:
            try:
                self.llm.polish(recs, card)
            except Exception:
                pass
        point = {
            "t": round(ev.t, 2),
            "time": fmt_clock(ev.t),
            "trigger": {"kind": ev.kind, "source": ev.source, "evidence": ev.evidence[:200], "data": ev.data},
            "recommendations": [r.to_dict() for r in recs],
            "new": [r.id for r in new],
            "resolved": resolved,
            "next_action": recs[0].action if recs else "Continue reassessment (MARCH-PAWS) and documentation.",
        }
        self.points.append(point)
        return point

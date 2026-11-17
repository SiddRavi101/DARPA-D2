"""
Stateful DD Form 1380 (TCCC Card) builder — Lane 1 emitter.

Events stream in along the scenario timeline; `apply()` folds each into the
card and returns whether anything changed. Repeated mentions ("tourniquet's
on the left leg" said three times) collapse into one field with the earliest
timestamp; each field keeps provenance (t, evidence, source).

Field layout follows the printed DD1380 (Jun 2014): casualty block,
mechanism, injury sites, tourniquets per limb, signs & symptoms time series,
treatments grouped C / A / B / C(fluids) / Meds / Other, notes. DARPA has not
published the final-event output schema yet; when it does, only
`to_output()` should need to change.
"""

from __future__ import annotations

import copy
from typing import Optional

from .extract import Event

LIMB_SITES = ("RUE", "LUE", "RLE", "LLE")
DEDUP_WINDOW_S = 300.0     # same med / same fluid within 5 min = same administration
VITALS_MERGE_S = 30.0      # vitals spoken within 30 s form one row


def _prov(ev: Event) -> dict:
    return {"t": round(ev.t, 2), "source": ev.source, "evidence": ev.evidence[:300]}


def fmt_clock(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


class DD1380Card:
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.card: dict = {
            "casualty": {"battle_roster_no": None, "evac_category": None, "name": None,
                         "service": None, "unit": None, "allergies": None, "date": None, "time": None},
            "mechanism_of_injury": [],
            "injuries": [],
            "tourniquets": {s: None for s in LIMB_SITES},
            "signs_and_symptoms": [],
            "treatments": {
                "c_hemorrhage": {"extremity_tq": [], "junctional_tq": [], "truncal_tq": [],
                                 "hemostatic_dressing": [], "pressure_dressing": []},
                "a_airway": {"intact": None, "npa": None, "cric": None, "et_tube": None, "sga": None,
                             "recovery_position": None},
                "b_breathing": {"o2": None, "needle_d": [], "chest_tube": [], "chest_seal": []},
                "c_circulation": {"iv_io_access": None, "fluids": [], "blood_products": []},
                "medications": {"analgesic": [], "antibiotic": [], "other": []},
                "other": {"combat_pill_pack": None, "eye_shield": [], "splint": [],
                          "hypothermia_prevention": None, "cpr": []},
            },
            "notes": [],
        }

    # ── public ──────────────────────────────────────────────────────────────

    def apply(self, ev: Event) -> bool:
        before = copy.deepcopy(self.card)
        getattr(self, f"_on_{ev.kind}", lambda e: None)(ev)
        return self.card != before

    def snapshot(self) -> dict:
        return copy.deepcopy(self.card)

    def to_output(self) -> dict:
        return {"scenario_id": self.scenario_id, "form": "DD1380", "form_version": "JUN 2014",
                "card": self.snapshot()}

    # ── handlers ────────────────────────────────────────────────────────────

    def _on_mechanism(self, ev: Event):
        mech = ev.data["mechanism"]
        if not any(m["value"] == mech for m in self.card["mechanism_of_injury"]):
            self.card["mechanism_of_injury"].append({"value": mech, **_prov(ev)})

    def _on_injury(self, ev: Event):
        for site in ev.data.get("sites") or ["unspecified"]:
            existing = [i for i in self.card["injuries"] if i["site"] == site]
            # an unspecified "bleeding" mention adds nothing once a site-specific finding exists
            if site == "unspecified" and self.card["injuries"]:
                continue
            if any(i["type"] == ev.data["type"] for i in existing):
                continue
            # upgrade severity in place (bleeding → massive_hemorrhage → amputation)
            rank = {"bleeding": 1, "penetrating_wound": 1, "laceration": 1, "massive_hemorrhage": 2,
                    "amputation": 3}
            up = [i for i in existing if rank.get(i["type"], 0) and rank.get(ev.data["type"], 0) > rank.get(i["type"], 0)]
            if up:
                up[0]["type"] = ev.data["type"]
                up[0]["upgraded_at"] = round(ev.t, 2)
                continue
            if any(rank.get(i["type"], 0) > rank.get(ev.data["type"], 0) for i in existing) and rank.get(ev.data["type"], 0):
                continue
            self.card["injuries"].append({"site": site, "type": ev.data["type"], **_prov(ev)})

    def _on_vitals(self, ev: Event):
        rows = self.card["signs_and_symptoms"]
        if rows and ev.t - rows[-1]["t"] <= VITALS_MERGE_S and not (set(ev.data) & set(rows[-1]["values"])):
            rows[-1]["values"].update(ev.data)
            return
        if rows and rows[-1]["values"] == ev.data:
            return  # repeated read-back of the same numbers
        rows.append({"time": fmt_clock(ev.t), "values": dict(ev.data), **_prov(ev)})

    def _on_evac(self, ev: Event):
        self.card["casualty"]["evac_category"] = ev.data["category"]

    def _on_intervention(self, ev: Event):
        tx = self.card["treatments"]
        name, sites = ev.data["name"], ev.data.get("sites") or []
        entry = {**_prov(ev)}
        if ev.data.get("stated_time"):
            entry["stated_time"] = ev.data["stated_time"]

        if name == "extremity_tq":
            limb_sites = [s for s in sites if s in LIMB_SITES] or ["unspecified"]
            for s in limb_sites:
                if s in LIMB_SITES:
                    cur = self.card["tourniquets"][s]
                    if cur is None:
                        self.card["tourniquets"][s] = {"type": "Extremity", "time": fmt_clock(ev.t), "count": 1, **entry}
                    elif ev.data.get("additional") and ev.t - cur["t"] > 10:
                        cur["count"] += 1
                    else:
                        continue
                if not any(x.get("site") == s for x in tx["c_hemorrhage"]["extremity_tq"]) or ev.data.get("additional"):
                    tx["c_hemorrhage"]["extremity_tq"].append({"site": s, **entry})
            # drop a previously unspecified TQ once its limb is known
            if any(s in LIMB_SITES for s in limb_sites):
                tx["c_hemorrhage"]["extremity_tq"] = [x for x in tx["c_hemorrhage"]["extremity_tq"]
                                                      if x["site"] != "unspecified"]
            return
        if name in ("junctional_tq", "hemostatic_dressing", "pressure_dressing"):
            lst = tx["c_hemorrhage"][name]
            site = sites[0] if sites else "unspecified"
            if not any(x["site"] == site for x in lst):
                lst.append({"site": site, **entry})
            return
        if name in ("npa", "cric", "et_tube", "sga", "recovery_position"):
            if tx["a_airway"][name] is None:
                tx["a_airway"][name] = entry
            return
        if name in ("needle_d", "chest_tube", "chest_seal"):
            lst = tx["b_breathing"][name]
            site = sites[0] if sites else "unspecified"
            if not any(x["site"] == site for x in lst):
                lst.append({"site": site, **entry})
            return
        if name == "o2":
            tx["b_breathing"]["o2"] = tx["b_breathing"]["o2"] or entry
            return
        if name == "iv_io_access":
            tx["c_circulation"]["iv_io_access"] = tx["c_circulation"]["iv_io_access"] or entry
            return
        if name in ("combat_pill_pack", "hypothermia_prevention"):
            tx["other"][name] = tx["other"][name] or entry
            return
        if name in ("eye_shield", "splint"):
            lst = tx["other"][name]
            site = sites[0] if sites else "unspecified"
            if not any(x["site"] == site for x in lst):
                lst.append({"site": site, **entry})
            return
        if name == "cpr":
            self._add_cpr(ev.t, ev.t, entry)

    def _add_cpr(self, start: float, end: float, entry: dict):
        lst = self.card["treatments"]["other"]["cpr"]
        for x in lst:
            if start <= x["end"] + 30 and end >= x["start"] - 30:  # overlapping / adjacent episode
                x["start"], x["end"] = min(x["start"], start), max(x["end"], end)
                return
        lst.append({"start": round(start, 2), "end": round(end, 2), **entry})

    def _on_medication(self, ev: Event):
        lst = self.card["treatments"]["medications"][ev.data["class"]]
        for m in lst:
            if m["name"] == ev.data["name"] and ev.t - m["t"] < DEDUP_WINDOW_S and not ev.data.get("repeat"):
                m["dose"] = m["dose"] or ev.data.get("dose")
                m["route"] = m["route"] or ev.data.get("route")
                return
        lst.append({"name": ev.data["name"], "dose": ev.data.get("dose"), "route": ev.data.get("route"),
                    "time": fmt_clock(ev.t), **_prov(ev)})

    def _on_fluid(self, ev: Event):
        key = "blood_products" if ev.data["name"] in ("Whole Blood", "PRBC", "Plasma") else "fluids"
        lst = self.card["treatments"]["c_circulation"][key]
        for f in lst:
            if f["name"] == ev.data["name"] and ev.t - f["t"] < DEDUP_WINDOW_S:
                f["volume"] = f["volume"] or ev.data.get("volume")
                f["route"] = f["route"] or ev.data.get("route")
                return
        lst.append({"name": ev.data["name"], "volume": ev.data.get("volume"), "route": ev.data.get("route"),
                    "time": fmt_clock(ev.t), **_prov(ev)})

    def _on_imu(self, ev: Event):
        if ev.data["label"] == "cpr":
            self._add_cpr(ev.t, ev.data["end"], {**_prov(ev), "rate_per_min": ev.data.get("rate_per_min")})
        elif ev.data["label"] == "movement":
            self.card["notes"].append({"text": f"Casualty movement detected {fmt_clock(ev.t)}-"
                                               f"{fmt_clock(ev.data['end'])} (accelerometry)", **_prov(ev)})

    # ── queries used by decision support ────────────────────────────────────

    def has_tq(self, site: Optional[str] = None) -> bool:
        if site:
            return self.card["tourniquets"].get(site) is not None
        return any(self.card["tourniquets"].values()) or bool(self.card["treatments"]["c_hemorrhage"]["extremity_tq"])

    def latest_vitals(self) -> dict:
        merged: dict = {}
        for row in self.card["signs_and_symptoms"]:
            merged.update(row["values"])
        return merged

    def med_given(self, name: str) -> bool:
        return any(m["name"].startswith(name) for lst in self.card["treatments"]["medications"].values() for m in lst)

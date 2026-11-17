"""
Clinical event extraction from transcript segments.

Deterministic, timestamped, negation-aware pattern extraction of the facts a
DD Form 1380 needs (injuries, mechanism, vitals, interventions, meds). It is
the always-on baseline; an LLM extractor (llm.py) can add events on top, and
the stateful card de-duplicates whatever both produce.

Every event keeps `t` (seconds from scenario start) and `evidence` (the
source utterance) so graders and medics can audit each card field.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .asr import Segment


@dataclass
class Event:
    kind: str               # injury | mechanism | vitals | intervention | medication | fluid | evac | imu
    t: float
    data: dict = field(default_factory=dict)
    evidence: str = ""
    source: str = "speech"  # speech | imu | llm
    speaker: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Vocabulary ────────────────────────────────────────────────────────────────

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

LIMBS = {
    "arm": "UE", "forearm": "UE", "upper arm": "UE", "hand": "UE", "elbow": "UE", "wrist": "UE",
    "shoulder": "UE", "bicep": "UE", "biceps": "UE",
    "leg": "LE", "thigh": "LE", "knee": "LE", "calf": "LE", "shin": "LE", "foot": "LE",
    "ankle": "LE", "lower leg": "LE", "femur": "LE",
}
TORSO = {"chest": "chest", "abdomen": "abdomen", "belly": "abdomen", "stomach": "abdomen",
         "head": "head", "face": "face", "neck": "neck", "groin": "groin", "pelvis": "pelvis",
         "back": "back", "axilla": "axilla", "armpit": "axilla", "buttock": "buttock", "eye": "eye"}

MECHANISMS = {
    "gsw": "GSW", "gunshot": "GSW", "shot": "GSW", "bullet": "GSW",
    "ied": "IED", "blast": "IED", "explosion": "IED", "grenade": "Grenade", "rpg": "RPG",
    "mortar": "Artillery", "artillery": "Artillery", "landmine": "Landmine", "mine": "Landmine",
    "shrapnel": "IED", "fragment": "IED", "frag": "IED",
    "fall": "Fall", "fell": "Fall", "vehicle": "MVC", "rollover": "MVC", "mvc": "MVC",
    "burn": "Burn", "burned": "Burn", "burns": "Burn", "fire": "Burn", "blunt": "Blunt",
}

INJURY_TYPES = [
    (r"amputat\w*|blown off|severed|missing (?:his|her|the)? ?(?:leg|arm|foot|hand)", "amputation"),
    (r"arterial|spurting|pulsatile|bleeding out|massive (?:bleed|hemorrhage)|bleeding heavily|"
     r"heavy bleeding|life[- ]threatening (?:bleed|hemorrhage)", "massive_hemorrhage"),
    (r"sucking chest|open chest|chest wound|penetrating chest", "open_chest_wound"),
    (r"bleed\w*|hemorrhag\w*", "bleeding"),
    (r"fractur\w*|broken|deformit\w*", "fracture"),
    (r"burn\w*", "burn"),
    (r"laceration|lac\b|cut\b|gash", "laceration"),
    (r"wound|entry|exit wound|hole|gsw|shrapnel|frag\w*", "penetrating_wound"),
    (r"head injury|tbi|concuss\w*|unequal pupils", "head_injury"),
]

INTERVENTIONS = [
    # (regex, category, name)
    (r"junctional tourniquet|sam junctional|jett\b|croc\b", "C", "junctional_tq"),
    (r"tourniquet|\btq\b|\bcat\b|soft[- ]t", "C", "extremity_tq"),
    (r"combat gauze|hemostatic|celox|quikclot|chitogauze|pack(?:ed|ing)? the wound|wound packing", "C", "hemostatic_dressing"),
    (r"pressure dressing|pressure bandage|israeli bandage|emergency bandage|olaes", "C", "pressure_dressing"),
    (r"\bnpa\b|nasopharyngeal|nasal airway|nasal trumpet", "A", "npa"),
    (r"\bcric\b|cricothyroidotomy|surgical airway", "A", "cric"),
    (r"\bi-?gel\b|supraglottic|\bsga\b|king airway", "A", "sga"),
    (r"intubat\w*|et tube|endotracheal", "A", "et_tube"),
    (r"recovery position", "A", "recovery_position"),
    (r"chest seal|hyfin|halo seal|occlusive dressing", "B", "chest_seal"),
    (r"needle (?:decompress\w*|d\b|chest)|\bndc\b|decompress\w*", "B", "needle_d"),
    (r"chest tube|finger thoracostomy", "B", "chest_tube"),
    (r"\boxygen\b|\bo2\b|non-?rebreather", "B", "o2"),
    (r"hypothermia|blizzard|hpmk|ready-?heat|warming blanket|space blanket|heat reflective", "H", "hypothermia_prevention"),
    (r"splint\w*|sam splint|traction", "O", "splint"),
    (r"eye shield|rigid shield", "O", "eye_shield"),
    (r"pill pack|combat wound med\w*|cwmp", "O", "combat_pill_pack"),
    (r"\bcpr\b|chest compressions|compressions", "O", "cpr"),
    (r"\b(?:iv|io) (?:access|line|in|established|placed)\b|intraosseous|\bfast ?1\b|\bez-?io\b|"
     r"\bline (?:is )?in\b", "C", "iv_io_access"),
]

MEDS = {
    "txa": ("TXA", "other"), "tranexamic": ("TXA", "other"),
    "ketamine": ("Ketamine", "analgesic"), "fentanyl": ("Fentanyl", "analgesic"),
    "otfc": ("Fentanyl (OTFC)", "analgesic"), "lollipop": ("Fentanyl (OTFC)", "analgesic"),
    "morphine": ("Morphine", "analgesic"), "meloxicam": ("Meloxicam", "analgesic"),
    "tylenol": ("Acetaminophen", "analgesic"), "acetaminophen": ("Acetaminophen", "analgesic"),
    "moxifloxacin": ("Moxifloxacin", "antibiotic"), "moxi": ("Moxifloxacin", "antibiotic"),
    "ertapenem": ("Ertapenem", "antibiotic"), "ceftriaxone": ("Ceftriaxone", "antibiotic"),
    "cefazolin": ("Cefazolin", "antibiotic"), "naloxone": ("Naloxone", "other"),
    "narcan": ("Naloxone", "other"), "ondansetron": ("Ondansetron", "other"),
    "zofran": ("Ondansetron", "other"), "calcium": ("Calcium", "other"),
    "lidocaine": ("Lidocaine", "other"),
}
FLUIDS = {
    "whole blood": "Whole Blood", "lto": "Whole Blood", "low titer": "Whole Blood",
    "prbc": "PRBC", "packed red": "PRBC", "plasma": "Plasma", "ffp": "Plasma",
    "freeze dried plasma": "Plasma", "saline": "NS", "normal saline": "NS",
    "lactated ringer": "LR", "\\blr\\b": "LR", "hextend": "Hextend", "plasma-lyte": "Plasma-Lyte",
}

_NEG = re.compile(r"\b(no|not|without|denies|negative for|isn't|wasn't|hasn't|don't|doesn't|"
                  r"didn't|no sign of|absent|need(?:s)? (?:a|an)|get (?:me )?(?:a|an)|"
                  r"should|going to|gonna|prepare|ready)\b[^.,;]{0,25}$")
_ROUTES = r"(iv|io|im|in|po|intranasal|intramuscular|intravenous|intraosseous|oral|by mouth|sublingual|buccal)"


def words_to_numbers(text: str) -> str:
    """'one twenty' → '120', 'ninety eight' → '98' (spoken vitals)."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        w = toks[i].lower().strip(",.")
        if w in _NUM_WORDS:
            val = _NUM_WORDS[w]
            j = i + 1
            if val == 100 and out and out[-1].isdigit() and int(out[-1]) < 10:
                val = int(out.pop()) * 100
            while j < len(toks) and toks[j].lower().strip(",.") in _NUM_WORDS:
                nxt = _NUM_WORDS[toks[j].lower().strip(",.")]
                if nxt == 100:
                    val *= 100
                elif val >= 100 and val % 100 == 0 and nxt < 100:
                    val += nxt
                elif val < 10 and nxt >= 10 and nxt % 10 == 0:  # "one twenty" → 120
                    val = val * 100 + nxt
                elif val % 10 == 0 and nxt < 10:
                    val += nxt
                else:
                    break
                j += 1
            out.append(str(val))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _negated(text: str, start: int) -> bool:
    return bool(_NEG.search(text[max(0, start - 40):start]))


def find_sites(text: str) -> list[str]:
    """Return DD1380-style sites: RUE/LUE/RLE/LLE or torso regions."""
    sites = []
    for m in re.finditer(r"\b(left|right|l|r)\s+(upper arm|lower leg|forearm|arm|hand|elbow|wrist|shoulder|"
                         r"biceps?|leg|thigh|knee|calf|shin|foot|ankle|femur|chest|groin|axilla|armpit|eye)\b",
                         text):
        side = "L" if m.group(1)[0] == "l" else "R"
        part = m.group(2)
        if part in LIMBS:
            sites.append(f"{side}{LIMBS[part]}")
        else:
            sites.append(f"{side}_{TORSO[part]}" if part in TORSO else part)
    if not sites:
        for part, region in TORSO.items():
            if re.search(rf"\b{part}\b", text):
                sites.append(region)
    return list(dict.fromkeys(sites))


_STATED_TIME = re.compile(r"\b(?:at|time|on at)\s*(\d{1,2}):?(\d{2})\b(?:\s*(?:hours|hrs|zulu))?")


def extract(seg: Segment) -> list[Event]:
    raw = seg.text
    text = words_to_numbers(raw.lower())
    t, ev = seg.start, []

    def add(kind, data):
        ev.append(Event(kind, t, data, raw, "speech", seg.speaker))

    sites = find_sites(text)
    stated = _STATED_TIME.search(text)
    stated_time = f"{int(stated.group(1)):02d}{stated.group(2)}" if stated else None

    # Mechanism
    for k, mech in MECHANISMS.items():
        m = re.search(rf"\b{k}\b", text)
        if m and not _negated(text, m.start()):
            add("mechanism", {"mechanism": mech})

    # Injuries
    for pat, itype in INJURY_TYPES:
        m = re.search(pat, text)
        if m and not _negated(text, m.start()):
            add("injury", {"type": itype, "sites": sites or ["unspecified"]})
            if itype in ("amputation", "massive_hemorrhage", "open_chest_wound"):
                break  # the most specific finding wins; avoid double-counting "bleeding"

    # Interventions (performed, not planned / negated)
    done_tq = False
    for pat, cat, name in INTERVENTIONS:
        m = re.search(pat, text)
        if not m or _negated(text, m.start()):
            continue
        if name == "extremity_tq" and done_tq:
            continue
        if name == "junctional_tq":
            done_tq = True
        data = {"category": cat, "name": name, "sites": sites}
        if stated_time:
            data["stated_time"] = stated_time
        if name in ("extremity_tq", "junctional_tq") and re.search(r"\b(second|another|2nd)\b", text):
            data["additional"] = True
        if name == "hypothermia_prevention" and not re.search(r"blanket|blizzard|hpmk|kit|wrap|heat|warm", text):
            continue
        add("intervention", data)

    # Medications
    for k, (name, mclass) in MEDS.items():
        m = re.search(rf"\b{k}\b", text)
        if not m or _negated(text, m.start()):
            continue
        window = text[m.start(): m.start() + 60] + " " + text[max(0, m.start() - 30): m.start()]
        dose = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|micrograms?|milligrams?|g|grams?|ml|cc)\b", window)
        route = re.search(rf"\b{_ROUTES}\b", window)
        add("medication", {
            "name": name, "class": mclass,
            "dose": f"{dose.group(1)} {_unit(dose.group(2))}" if dose else None,
            "route": _route(route.group(1)) if route else None,
            "repeat": bool(re.search(r"\b(second|another|repeat|again|additional)\b", text)),
        })

    # Fluids / blood
    for k, name in FLUIDS.items():
        m = re.search(k if k.startswith("\\") else rf"\b{k}\b", text)
        if not m or _negated(text, m.start()):
            continue
        vol = re.search(r"(\d+)\s*(ml|cc|units?|liters?|l)\b", text)
        route = re.search(rf"\b{_ROUTES}\b", text)
        add("fluid", {"name": name, "volume": f"{vol.group(1)} {vol.group(2)}" if vol else None,
                      "route": _route(route.group(1)) if route else None})

    # Vitals
    v = extract_vitals(text)
    if v:
        add("vitals", v)

    # Evac category
    m = re.search(r"\b(urgent surgical|urgent|priority|routine|expectant)\b", text)
    if m and re.search(r"evac|medevac|casevac|category|cat\b|nine[- ]line|9[- ]line|triage", text):
        add("evac", {"category": m.group(1).title()})
    return ev


def extract_vitals(text: str) -> Optional[dict]:
    v: dict = {}
    m = re.search(r"\b(?:hr|heart rate|pulse(?: rate)?|pulse is|pulse of)\D{0,12}(\d{2,3})\b", text)
    if m:
        v["pulse_rate"] = int(m.group(1))
    m = re.search(r"\b(radial|carotid|femoral|pedal|brachial)\b", text)
    if m and re.search(r"pulse", text):
        v["pulse_location"] = m.group(1)
        if re.search(r"(no|absent|lost|weak|thready)\s+\w*\s*" + m.group(1) + r"|" + m.group(1) + r"\s+\w*\s*(absent|weak|thready)", text):
            v["pulse_quality"] = "weak/absent"
    m = re.search(r"\b(?:bp|blood pressure)\D{0,12}(\d{2,3})\s*(?:/|over)\s*(\d{2,3})", text)
    if m:
        v["bp"] = f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"\b(?:rr|resp(?:iratory)? rate|resps|respirations|breathing)\D{0,12}(\d{1,2})\b", text)
    if m:
        v["resp_rate"] = int(m.group(1))
    m = re.search(r"\b(?:spo2|sp ?o2|sat(?:s|uration)?|pulse ox)\D{0,12}(\d{2,3})\b", text)
    if m and 40 <= int(m.group(1)) <= 100:
        v["spo2"] = int(m.group(1))
    m = re.search(r"\bavpu\D{0,8}\b([avpu])\b|\b(alert|responds to voice|verbal|responds to pain|"
                  r"painful stimuli|unresponsive|unconscious)\b", text)
    if m:
        word = m.group(1) or m.group(2)
        v["avpu"] = {"a": "A", "alert": "A", "v": "V", "verbal": "V", "responds to voice": "V",
                     "p": "P", "responds to pain": "P", "painful stimuli": "P",
                     "u": "U", "unresponsive": "U", "unconscious": "U"}[word]
    m = re.search(r"\bpain\D{0,15}(\d{1,2})\s*(?:/|out of|of)\s*10\b|\bpain (?:is|at|level)\s*(\d{1,2})\b", text)
    if m:
        v["pain"] = int(m.group(1) or m.group(2))
    return v or None


def _unit(u: str) -> str:
    u = u.lower()
    return {"micrograms": "mcg", "microgram": "mcg", "milligrams": "mg", "milligram": "mg",
            "grams": "g", "gram": "g", "cc": "mL", "ml": "mL"}.get(u, u)


def _route(r: str) -> str:
    r = r.lower()
    return {"intravenous": "IV", "intraosseous": "IO", "intramuscular": "IM", "intranasal": "IN",
            "oral": "PO", "by mouth": "PO", "buccal": "OTFC", "sublingual": "SL"}.get(r, r.upper())


def extract_all(segments: list[Segment]) -> list[Event]:
    out = []
    for s in segments:
        out.extend(extract(s))
    return out

"""
Output contract for the D2 Part 2 Qualification Task (ICD v1.1, Oct 2026).

Everything that touches predictions.json goes through this module so the
container can never emit a file that fails the JHU/APL validator.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "1.0"

# Order matters: it is the order of the model heads and of the "sites" array.
SITES: tuple[tuple[str, str], ...] = (
    ("upper_extremity", "left"),
    ("upper_extremity", "right"),
    ("lower_extremity", "left"),
    ("lower_extremity", "right"),
)
SITE_KEYS: tuple[str, ...] = tuple(f"{r}:{l}" for r, l in SITES)

CLASSES: tuple[str, ...] = ("no_injury", "wound", "amputation", "not_testable")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

BODY_REGIONS = {"upper_extremity", "lower_extremity"}
LATERALITIES = {"left", "right"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# A horizontal mirror swaps the casualty's ANATOMICAL left and right.
# Index permutation over SITES used for flip-augmentation and flip-TTA.
FLIP_PERMUTATION: tuple[int, ...] = (1, 0, 3, 2)

# Used only when an image cannot be decoded or the model crashes.
# "no_injury" is the safest guess under an accuracy metric when the
# training prior is unknown; train.py overwrites this with the real prior.
DEFAULT_FALLBACK: tuple[str, ...] = ("no_injury",) * 4


def list_images(input_dir: str | Path) -> list[Path]:
    """Flat directory, case-insensitive extensions, deterministic order."""
    d = Path(input_dir)
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )


def make_prediction(image_id: str, labels: Iterable[str]) -> dict:
    labels = list(labels)
    if len(labels) != 4:
        raise ValueError(f"need 4 site labels, got {len(labels)}")
    sites = []
    for (region, lat), lab in zip(SITES, labels):
        if lab not in CLASS_TO_IDX:
            lab = "no_injury"
        sites.append({"body_region": region, "laterality": lat, "injury_type": lab})
    return {"image_id": image_id, "sites": sites}


def build_document(predictions: list[dict], team_name: str, version: str, email: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "submission": {"team_name": team_name, "version": version, "email": email},
        "predictions": predictions,
    }


def write_document(doc: dict, out_path: str | Path) -> None:
    """Atomic UTF-8 write; refuses NaN/Infinity (ICD 3.2)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=".predictions.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, out_path)


def validate_document(doc: object, expected_image_ids: Iterable[str] | None = None) -> list[str]:
    """
    Re-implementation of the checks listed in ICD 4.2. Returns a list of
    violation paths (empty == valid). The official validate_predictions.py
    from JHU/APL is authoritative; run it too when you have it.
    """
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["$: root must be an object"]

    if not isinstance(doc.get("schema_version"), str):
        errs.append("$.schema_version: required string")
    sub = doc.get("submission")
    if not isinstance(sub, dict):
        errs.append("$.submission: required object")
    else:
        for k in ("team_name", "version", "email"):
            if not isinstance(sub.get(k), str) or not sub.get(k):
                errs.append(f"$.submission.{k}: required non-empty string")

    preds = doc.get("predictions")
    if not isinstance(preds, list):
        return errs + ["$.predictions: required array"]

    seen: set[str] = set()
    for i, p in enumerate(preds):
        base = f"$.predictions[{i}]"
        if not isinstance(p, dict):
            errs.append(f"{base}: must be object")
            continue
        iid = p.get("image_id")
        if not isinstance(iid, str) or not iid:
            errs.append(f"{base}.image_id: required string")
        elif iid in seen:
            errs.append(f"{base}.image_id: duplicate '{iid}'")
        else:
            seen.add(iid)
        sites = p.get("sites")
        if not isinstance(sites, list) or len(sites) != 4:
            errs.append(f"{base}.sites: must contain exactly four site predictions")
            continue
        got_keys = []
        for j, s in enumerate(sites):
            sb = f"{base}.sites[{j}]"
            if not isinstance(s, dict):
                errs.append(f"{sb}: must be object")
                continue
            if s.get("body_region") not in BODY_REGIONS:
                errs.append(f"{sb}.body_region: invalid value {s.get('body_region')!r}")
            if s.get("laterality") not in LATERALITIES:
                errs.append(f"{sb}.laterality: invalid value {s.get('laterality')!r}")
            if s.get("injury_type") not in CLASS_TO_IDX:
                errs.append(f"{sb}.injury_type: invalid value {s.get('injury_type')!r}")
            got_keys.append(f"{s.get('body_region')}:{s.get('laterality')}")
        if sorted(got_keys) != sorted(SITE_KEYS):
            errs.append(f"{base}.sites: must cover each of {list(SITE_KEYS)} exactly once")

    if expected_image_ids is not None:
        missing = sorted(set(expected_image_ids) - seen)
        for m in missing:
            errs.append(f"$.predictions: missing image_id '{m}'")

    # NaN / Infinity can only sneak in via floats; we never emit floats, but check anyway.
    def _walk(o, path):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            errs.append(f"{path}: NaN/Infinity not allowed")
        elif isinstance(o, dict):
            for k, v in o.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for k, v in enumerate(o):
                _walk(v, f"{path}[{k}]")

    _walk(doc, "$")
    return errs

"""
Local stand-in for JHU/APL's validate_predictions.py (ICD §4.2).

    python validate_predictions.py predictions.json [--images DIR]

Exit 0 = valid, 1 = schema violations, 2 = unreadable / invalid JSON.
Run the official script too once JHU/APL distributes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d2qual.schema import list_images, validate_document  # noqa: E402


def _reject_constant(c):
    raise ValueError(f"nonstandard JSON constant {c}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--images", help="input dir; checks every image has a prediction")
    a = ap.parse_args(argv)
    try:
        with open(a.predictions, encoding="utf-8") as f:
            doc = json.load(f, parse_constant=_reject_constant)
    except Exception as e:
        print(f"INVALID JSON: {e}")
        return 2
    expected = [p.name for p in list_images(a.images)] if a.images else None
    errs = validate_document(doc, expected)
    if errs:
        for e in errs:
            print(e)
        print(f"{len(errs)} violation(s)")
        return 1
    print(f"OK: {len(doc['predictions'])} predictions valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())

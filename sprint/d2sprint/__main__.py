"""
    python -m d2sprint documentation    --input /app/input --output /app/output
    python -m d2sprint decision_support --input /app/input --output /app/output
    python -m d2sprint both             ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .pipeline import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d2sprint")
    ap.add_argument("lane", choices=["documentation", "decision_support", "both"],
                    nargs="?", default=os.environ.get("D2_LANE", "both"))
    ap.add_argument("--input", default=os.environ.get("D2_INPUT", "/app/input"))
    ap.add_argument("--output", default=os.environ.get("D2_OUTPUT", "/app/output"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    lanes = ["documentation", "decision_support"] if a.lane == "both" else [a.lane]
    if a.lane == "decision_support":
        lanes.append("documentation")  # DS needs the card state; emitting it is free
    results = run(a.input, a.output, lanes)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

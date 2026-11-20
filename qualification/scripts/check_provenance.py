"""
Exit 0 only if EVERY ensemble member in a weights dir was trained on the DARPA training set
(checkpoint / vlm_meta.json field data_source == "darpa", set by `d2qual.train --data-source darpa`,
which scripts/day_one.py passes). Called by scripts/submit.sh before building the image.

    python scripts/check_provenance.py model/weights
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check(weights_dir: str | Path) -> list[str]:
    import torch

    d = Path(weights_dir)
    bad = []
    for p in sorted(d.glob("*.pt")):
        src = torch.load(p, map_location="cpu", weights_only=False).get("data_source")
        if src != "darpa":
            bad.append(f"{p.name}: data_source={src!r}")
    for m in sorted(d.glob("*/vlm_meta.json")):
        src = json.loads(m.read_text()).get("data_source")
        if src != "darpa":
            bad.append(f"{m.parent.name}: data_source={src!r}")
    return bad


if __name__ == "__main__":
    bad = check(sys.argv[1] if len(sys.argv) > 1 else "model/weights")
    for b in bad:
        print("NOT DARPA-TRAINED:", b)
    sys.exit(1 if bad else 0)

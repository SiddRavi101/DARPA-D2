"""
Counts-only error report for one candidate's out-of-fold predictions (no images, no label rows):
per-site confusion matrices, which wrong class each true class goes to, left/right swaps,
accuracy per casualty (anonymised as rank), and how confident the model is when it is wrong.

    python scripts/error_report.py --images I --labels L --groups-file G --probs runs/<cand>/oof_probs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d2qual.data import load_labels, match_images  # noqa: E402
from d2qual.schema import CLASSES, FLIP_PERMUTATION, SITE_KEYS  # noqa: E402
from d2qual.train import load_groups  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--groups-file", required=True)
    ap.add_argument("--probs", required=True)
    a = ap.parse_args(argv)

    lab = match_images(a.images, load_labels(a.labels))
    probs = json.loads(Path(a.probs).read_text())
    ids = [i for i in probs if i in lab]
    P = np.array([probs[i] for i in ids])                      # (N, 4 sites, 4 classes)
    y = np.array([[CLASSES.index(c) for c in lab[i]] for i in ids])
    pr = P.argmax(-1)
    groups = load_groups(ids, None, a.groups_file)
    short = [c[:5] for c in CLASSES]

    print(f"images {len(ids)}  site predictions {y.size}  accuracy {(pr == y).mean():.4f}")
    print("\n## Confusion per site (rows = truth, cols = predicted)")
    for s, key in enumerate(SITE_KEYS):
        m = np.zeros((4, 4), int)
        np.add.at(m, (y[:, s], pr[:, s]), 1)
        print(f"\n{key}  acc {(pr[:, s] == y[:, s]).mean():.3f}")
        print("        " + " ".join(f"{c:>6}" for c in short))
        for k in range(4):
            print(f"{short[k]:>6}  " + " ".join(f"{v:6d}" for v in m[k]))

    print("\n## Where each true class goes (all sites)")
    tot = np.zeros((4, 4), int)
    np.add.at(tot, (y.ravel(), pr.ravel()), 1)
    for k, c in enumerate(CLASSES):
        row = tot[k]
        print(f"{c:13s} n={row.sum():4d}  " + "  ".join(f"->{short[j]} {row[j] / max(1, row.sum()):.2f}" for j in range(4)))

    # left/right swap: a site wrong, but its mirror site's truth equals this prediction and vice versa
    perm = list(FLIP_PERMUTATION)
    swaps = 0
    for s in (0, 2):
        o = perm[s]
        sw = (pr[:, s] != y[:, s]) & (pr[:, s] == y[:, o]) & (pr[:, o] == y[:, s]) & (y[:, s] != y[:, o])
        swaps += int(sw.sum())
    wrong_pairs = int(((pr[:, 0] != y[:, 0]) | (pr[:, 1] != y[:, 1])).sum() + ((pr[:, 2] != y[:, 2]) | (pr[:, 3] != y[:, 3])).sum())
    print(f"\n## Left/right swaps: {swaps} limb pairs fully swapped (of {wrong_pairs} pairs with any error)")

    print("\n## Accuracy per casualty (anonymised, sorted worst first; n = photos)")
    per = defaultdict(list)
    for n, i in enumerate(ids):
        per[groups[i]].append(n)
    rows = []
    for g, ns in per.items():
        yy, pp = y[ns], pr[ns]
        amp = int((yy == CLASSES.index("amputation")).sum())
        rows.append(((pp == yy).mean(), len(ns), amp, Counter(CLASSES[k] for k in yy.ravel()).most_common(1)[0][0]))
    rows.sort()
    for r, (acc, n, amp, top) in enumerate(rows[:10]):
        print(f"  #{r + 1:2d} acc {acc:.2f}  n={n:3d}  amputation labels={amp:3d}  most common label={top}")
    accs = np.array([r[0] for r in rows])
    print(f"  casualties {len(rows)}: median acc {np.median(accs):.2f}, below 0.5: {(accs < 0.5).sum()}")

    print("\n## Confidence")
    conf = P.max(-1)
    print(f"  mean confidence when right {conf[pr == y].mean():.2f}, when wrong {conf[pr != y].mean():.2f}")
    print(f"  wrong with confidence > 0.8: {int(((pr != y) & (conf > 0.8)).sum())} of {int((pr != y).sum())} errors")


if __name__ == "__main__":
    main()

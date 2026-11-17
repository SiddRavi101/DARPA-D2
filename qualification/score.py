"""
Score predictions.json against ground truth with the Qualification Guide §4.2.2 metric:

    overall accuracy = correct class-at-site predictions / all class-at-site predictions

    python score.py predictions.json ground_truth_labels.json [--json report.json]

Also prints per-site accuracy, per-class precision/recall/F1, macro-F1 and a confusion
matrix ("additional machine learning performance metrics", §4.2.2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d2qual.data import load_labels  # noqa: E402
from d2qual.schema import CLASSES, SITE_KEYS  # noqa: E402


def score(pred: dict[str, list[str]], gt: dict[str, list[str]]) -> dict:
    conf = [[0] * len(CLASSES) for _ in CLASSES]
    site_ok = [0] * 4
    missing = [k for k in gt if k not in pred]
    for iid, truth in gt.items():
        p = pred.get(iid, ["__missing__"] * 4)
        for s in range(4):
            if p[s] == truth[s]:
                site_ok[s] += 1
            if p[s] in CLASSES:
                conf[CLASSES.index(truth[s])][CLASSES.index(p[s])] += 1
    n = len(gt)
    total = 4 * n
    correct = sum(site_ok)
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(len(CLASSES))) - tp
        fn = sum(conf[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                        "support": sum(conf[i])}
    return {
        "images": n, "missing_predictions": len(missing),
        "overall_accuracy": correct / total if total else 0.0,
        "per_site_accuracy": {k: site_ok[i] / n for i, k in enumerate(SITE_KEYS)} if n else {},
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
        "per_class": per_class,
        "confusion": {"rows=truth, cols=pred": list(CLASSES), "matrix": conf},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("ground_truth")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = score(load_labels(a.predictions), load_labels(a.ground_truth))
    print(f"images: {r['images']}  missing: {r['missing_predictions']}")
    print(f"OVERALL ACCURACY: {r['overall_accuracy']:.4f}   macro-F1: {r['macro_f1']:.4f}")
    for k, v in r["per_site_accuracy"].items():
        print(f"  {k:24s} {v:.4f}")
    print(f"  {'class':14s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    for c, v in r["per_class"].items():
        print(f"  {c:14s} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {v['support']:6d}")
    print("  confusion (rows=truth):", *CLASSES)
    for c, row in zip(CLASSES, r["confusion"]["matrix"]):
        print(f"  {c:14s}", row)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

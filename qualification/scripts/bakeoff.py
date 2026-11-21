"""
Bake-off: train many candidate models on IDENTICAL grouped folds, rank them by DARPA's
metric on out-of-fold (OOF) predictions, and build the best weighted ensemble.

  1. run        train candidates (resumable: finished runs are skipped)
       python scripts/bakeoff.py run --images I --labels L --runs runs/ \
           --candidates convnext_s,dinov2_b,eva02_b,siglip_b,convnext_s_limb,vlm_qwen2vl_2b \
           --group-regex '^(scene\\d+)_'
     Parallel: start one pod per subset of --candidates, all writing their own runs/<name>/,
     then copy the run folders together before step 2.
  2. leaderboard  rank + greedy ensemble selection on OOF probabilities
       python scripts/bakeoff.py leaderboard --images I --labels L --runs runs/
  3. assemble     copy the chosen members + ensemble.json into a weights dir for the container
       python scripts/bakeoff.py assemble --runs runs/ --out model/weights

Every candidate uses the same --seed/--folds/groups, so fold i holds the same images in every
run and OOF predictions are comparable and can be ensembled honestly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d2qual.data import load_labels, match_images  # noqa: E402
from d2qual.schema import CLASSES  # noqa: E402

VIT_LR = ["--backbone-lr-mult", "0.05"]
CANDIDATES: dict[str, list[str]] = {
    "convnext_s": ["--backbone", "convnext_small.fb_in22k_ft_in1k_384", "--size", "512"],
    "convnext_b": ["--backbone", "convnext_base.fb_in22k_ft_in1k_384", "--size", "512"],
    "eva02_b": ["--backbone", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", "--size", "448", *VIT_LR],
    "dinov2_b": ["--backbone", "vit_base_patch14_dinov2.lvd142m", "--size", "518", *VIT_LR],
    "siglip_b": ["--backbone", "vit_base_patch16_siglip_384.webli", "--size", "384", *VIT_LR],
    # Frozen DINOv2 (Apache-2.0; DINOv3 is barred: its licence excludes military use).
    # Backbone frozen (--backbone-lr-mult 0), only the head trains -> cheap probes.
    "dinov2_b_reg_probe": ["--backbone", "vit_base_patch14_reg4_dinov2.lvd142m", "--size", "518",
                           "--backbone-lr-mult", "0", "--lr", "1e-3"],
    "dinov2_l_reg_probe": ["--backbone", "vit_large_patch14_reg4_dinov2.lvd142m", "--size", "518",
                           "--backbone-lr-mult", "0", "--lr", "1e-3"],
    # Rohan's plan: Approach A = whole image, frozen backbone, 4 heads (dinov2_*_reg_probe above).
    # Approach B = pose -> shoulder/hip-anchored crop per limb (left crops mirrored) -> one shared
    # classifier with keypoint-confidence + out-of-frame features. 448 px crops keep wound detail.
    "dinov2_b_reg_limb_probe": ["--backbone", "vit_base_patch14_reg4_dinov2.lvd142m", "--size", "518",
                                "--limb-crops", "--crop-size", "448", "--backbone-lr-mult", "0", "--lr", "1e-3"],
    "dinov2_l_reg_limb_probe": ["--backbone", "vit_large_patch14_reg4_dinov2.lvd142m", "--size", "518",
                                "--limb-crops", "--crop-size", "448", "--backbone-lr-mult", "0", "--lr", "1e-3",
                                "--batch", "8"],
    "convnext_s_limb": ["--backbone", "convnext_small.fb_in22k_ft_in1k_384", "--size", "512", "--limb-crops"],
    "dinov2_b_limb": ["--backbone", "vit_base_patch14_dinov2.lvd142m", "--size", "518", "--limb-crops", *VIT_LR],
    "siglip2_b": ["--backbone", "vit_base_patch16_siglip_384.v2_webli", "--size", "384", *VIT_LR],
    "vlm_qwen2vl_2b": ["--backbone", "vlm:Qwen/Qwen2-VL-2B-Instruct", "--lr", "1e-4", "--batch", "4", "--epochs", "3"],
    # Qwen3-VL (Oct 2025, Apache-2.0): newer generation, needs transformers >= 4.57
    "vlm_qwen3vl_2b": ["--backbone", "vlm:Qwen/Qwen3-VL-2B-Instruct", "--lr", "1e-4", "--batch", "4", "--epochs", "3"],
    "vlm_qwen3vl_4b": ["--backbone", "vlm:Qwen/Qwen3-VL-4B-Instruct", "--lr", "1e-4", "--batch", "4", "--epochs", "3"],
}


def cmd_run(a):
    runs = Path(a.runs)
    runs.mkdir(parents=True, exist_ok=True)
    names = list(CANDIDATES) if a.candidates == "all" else a.candidates.split(",")
    for name in names:
        if name not in CANDIDATES:
            raise SystemExit(f"unknown candidate {name}; choose from {list(CANDIDATES)}")
    common = ["--images", a.images, "--labels", a.labels, "--folds", str(a.folds), "--seed", str(a.seed),
              "--workers", str(a.workers), "--keypoints", str(runs / "keypoints.json")]
    if a.group_regex:
        common += ["--group-regex", a.group_regex]
    if a.groups_file:
        common += ["--groups-file", a.groups_file]
    for name in names:
        out = runs / name
        if (out / "oof_probs.json").exists() and not a.force:
            print(f"== {name}: done, skipping")
            continue
        spec = list(CANDIDATES[name])
        if "--epochs" not in spec:
            spec += ["--epochs", str(a.epochs)]
        if "--batch" not in spec:
            spec += ["--batch", str(a.batch)]
        cmd = [sys.executable, "-m", "d2qual.train", *common, "--out", str(out), *spec, *a.extra]
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "train.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (out / "run_info.json").write_text(json.dumps({"returncode": rc, "seconds": round(time.time() - t0),
                                                       "cmd": cmd}, indent=1))
        tail = (out / "train.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        print(f"== {name}: rc={rc} in {time.time() - t0:.0f}s | " + " | ".join(tail), flush=True)


def _load_runs(runs: Path, truth: dict):
    out = {}
    for d in sorted(p for p in runs.iterdir() if (p / "oof_probs.json").exists()):
        raw = json.loads((d / "oof_probs.json").read_text())
        probs = {k: np.asarray(v, np.float64) for k, v in raw.items() if k in truth}
        if probs:
            out[d.name] = probs
    return out


def _acc(probs: dict, truth_idx: dict, ids) -> float:
    return float(np.mean([np.argmax(probs[i], -1) == truth_idx[i] for i in ids]))


def cmd_leaderboard(a):
    runs = Path(a.runs)
    truth = match_images(a.images, load_labels(a.labels))
    tidx = {k: np.array([CLASSES.index(c) for c in v]) for k, v in truth.items()}
    cands = _load_runs(runs, truth)
    if not cands:
        raise SystemExit(f"no finished runs with oof_probs.json under {runs}")
    ids = sorted(set.intersection(*[set(p) for p in cands.values()]))
    rows = []
    for name, probs in cands.items():
        acc = _acc(probs, tidx, ids)
        pred = np.stack([np.argmax(probs[i], -1) for i in ids])
        y = np.stack([tidx[i] for i in ids])
        recall = {c: float((pred[y == k] == k).mean()) if (y == k).any() else None for k, c in enumerate(CLASSES)}
        info = json.loads((runs / name / "run_info.json").read_text()) if (runs / name / "run_info.json").exists() else {}
        rows.append(dict(candidate=name, oof_accuracy=round(acc, 4), seconds=info.get("seconds"),
                         **{f"recall_{c}": None if r is None else round(r, 3) for c, r in recall.items()}))
    rows.sort(key=lambda r: -r["oof_accuracy"])

    # Greedy forward selection with replacement (Caruana et al. 2004) on OOF probabilities.
    chosen: list[str] = []
    best_acc, total = -1.0, None
    for _ in range(a.max_steps):
        trial = []
        for name, probs in cands.items():
            mix = {i: (total[i] if total else 0) + probs[i] for i in ids}
            trial.append((_acc(mix, tidx, ids), name, mix))
        acc, name, mix = max(trial, key=lambda t: (t[0], -len(t[1])))
        if acc <= best_acc + 1e-9 and chosen:
            break
        best_acc, total = acc, mix
        chosen.append(name)
    weights = {n: chosen.count(n) / len(chosen) for n in dict.fromkeys(chosen)}

    lines = [f"# Bake-off leaderboard ({len(ids)} images, out-of-fold, DARPA metric)", "",
             "| Candidate | OOF accuracy | " + " | ".join(f"recall {c}" for c in CLASSES) + " | train time |",
             "|---|---|" + "---|" * len(CLASSES) + "---|"]
    for r in rows:
        rec = " | ".join("-" if r[f"recall_{c}"] is None else f"{r[f'recall_{c}']:.3f}" for c in CLASSES)
        t = "-" if r["seconds"] is None else f"{r['seconds'] / 60:.0f} min"
        lines.append(f"| {r['candidate']} | **{r['oof_accuracy']:.4f}** | {rec} | {t} |")
    lines += ["", f"**Best ensemble (greedy on OOF): {best_acc:.4f}**  weights: "
              + ", ".join(f"{n} {w:.2f}" for n, w in weights.items()),
              "", "Differences under ~0.01-0.02 are within noise for a few thousand images: treat them as ties "
              "and prefer the faster / simpler model."]
    (runs / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (runs / "ensemble_choice.json").write_text(json.dumps({"oof_accuracy": best_acc, "weights": weights}, indent=1))
    print("\n".join(lines))


def cmd_assemble(a):
    runs, out = Path(a.runs), Path(a.out)
    choice = json.loads((runs / "ensemble_choice.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    member_w = {}
    for name, w in choice["weights"].items():
        src = runs / name
        folds = sorted(src.glob("*.pt")) + sorted(d for d in src.iterdir() if (d / "vlm_meta.json").exists())
        if not folds:
            raise SystemExit(f"{name}: no checkpoints to copy")
        for f in folds:
            dst = out / f"{name}__{f.name}"
            if f.is_dir():
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(f, dst)
            member_w[dst.name] = w / len(folds)  # a candidate's folds share its weight
    (out / "ensemble.json").write_text(json.dumps({"weights": member_w, "oof_accuracy": choice["oof_accuracy"]}, indent=1))
    print(f"assembled {len(member_w)} members into {out} (OOF ensemble accuracy {choice['oof_accuracy']:.4f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--images", required=True)
    r.add_argument("--labels", required=True)
    r.add_argument("--runs", default="runs")
    r.add_argument("--candidates", default="all")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--epochs", type=int, default=25)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--group-regex")
    r.add_argument("--groups-file")
    r.add_argument("--force", action="store_true")
    r.add_argument("extra", nargs=argparse.REMAINDER, help="extra args passed to d2qual.train (after --)")
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--images", required=True)
    lb.add_argument("--labels", required=True)
    lb.add_argument("--runs", default="runs")
    lb.add_argument("--max-steps", type=int, default=8)
    asm = sub.add_parser("assemble")
    asm.add_argument("--runs", default="runs")
    asm.add_argument("--out", default="model/weights")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        a.extra = [x for x in a.extra if x != "--"]
        cmd_run(a)
    elif a.cmd == "leaderboard":
        cmd_leaderboard(a)
    else:
        cmd_assemble(a)


if __name__ == "__main__":
    main()

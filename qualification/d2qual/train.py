"""
Train the injury classifier on the DARPA Part 2 training set.

    python -m d2qual.train --images /data/train/images --labels /data/train/labels.json \
        --out model/weights --folds 5 --epochs 25

Each fold writes `fold{i}.pt`; inference ensembles every *.pt it finds in
model/weights. Reported metric is exactly the ICD's:
    overall accuracy = correct class-at-site predictions / all class-at-site predictions
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import InjuryDataset, class_prior, load_labels, match_images
from .model import InjuryNet, ModelConfig, flip_logits, mirror_inputs, save_checkpoint
from .schema import CLASSES, SITE_KEYS, build_document, make_prediction


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_folds(ids: list[str], labels: dict[str, list[str]], k: int, seed: int) -> list[list[str]]:
    """Greedy stratification on the joint 4-site label signature."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault("|".join(labels[i]), []).append(i)
    folds: list[list[str]] = [[] for _ in range(k)]
    for sig in sorted(groups, key=lambda s: -len(groups[s])):
        members = groups[sig][:]
        rng.shuffle(members)
        for m in members:
            min(folds, key=len).append(m)
    return folds


def load_groups(ids: list[str], group_regex: str | None = None, groups_file: str | None = None) -> dict[str, str] | None:
    """
    Map image_id -> group (scene / manikin / session). Images of the same group are
    never split across train and val, so near-duplicate shots can't inflate CV.

      --group-regex '^(scene\\d+)_'      group = first capture group (or whole match)
      --groups-file groups.csv          CSV with columns image_id,group
    Images the regex doesn't match, or missing from the CSV, become singleton groups.
    """
    if not group_regex and not groups_file:
        return None
    out: dict[str, str] = {}
    if groups_file:
        import csv

        with open(groups_file, newline="", encoding="utf-8-sig") as f:
            rows = {r["image_id"]: r["group"] for r in csv.DictReader(f)}
        by_stem = {Path(k).stem.lower(): v for k, v in rows.items()}
        for i in ids:
            g = rows.get(i) or by_stem.get(Path(i).stem.lower())
            out[i] = str(g) if g is not None else f"__single__{i}"
    else:
        pat = re.compile(group_regex)
        for i in ids:
            m = pat.search(i)
            out[i] = (m.group(1) if m and m.groups() else m.group(0)) if m else f"__single__{i}"
    return out


def grouped_folds(ids: list[str], labels: dict[str, list[str]], groups: dict[str, str],
                  k: int, seed: int) -> list[list[str]]:
    """
    Group-aware k-fold: whole groups are assigned to folds, largest first, each to the
    fold that best balances size and per-site label distribution (greedy).
    """
    rng = random.Random(seed)
    members: dict[str, list[str]] = {}
    for i in ids:
        members.setdefault(groups[i], []).append(i)
    order = sorted(members, key=lambda g: (-len(members[g]), rng.random()))

    def label_counts(img_ids):
        c = Counter()
        for i in img_ids:
            for s, lab in enumerate(labels[i]):
                c[(s, lab)] += 1
        return c

    total = label_counts(ids)
    target = {key: v / k for key, v in total.items()}
    folds: list[list[str]] = [[] for _ in range(k)]
    fold_counts = [Counter() for _ in range(k)]
    for g in order:
        gc = label_counts(members[g])

        def cost(f):
            size_pen = (len(folds[f]) + len(members[g])) / max(1, len(ids) / k)
            lab_pen = sum(abs(fold_counts[f][key] + gc[key] - target[key]) for key in target) / max(1, sum(target.values()))
            return size_pen + lab_pen

        best = min(range(k), key=cost)
        folds[best].extend(members[g])
        fold_counts[best].update(gc)
    return folds


def _unpack(batch, device):
    """(x, y) or (x, crops, kp_feats, y) -> (model inputs, targets)."""
    *inputs, y = [t.to(device, non_blocking=True) for t in batch]
    if len(inputs) == 1:
        inputs += [None, None]
    return inputs, y


def predict_probs(model, inputs, tta: bool = True):
    x, crops, kpf = inputs
    p = model(x, crops, kpf).float().softmax(-1)
    if tta:
        p = p + flip_logits(model(*mirror_inputs(x, crops, kpf)).float()).softmax(-1)
    return p


@torch.no_grad()
def evaluate(model, loader, device, tta: bool = True, return_probs: bool = False) -> dict:
    model.eval()
    correct = total = 0
    per_site = np.zeros(4)
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    all_probs = []
    for batch in loader:
        inputs, y = _unpack(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probs = predict_probs(model, inputs, tta)
        pred = probs.argmax(-1)
        if return_probs:
            all_probs.append((probs / probs.sum(-1, keepdim=True)).cpu())
        correct += (pred == y).sum().item()
        total += y.numel()
        per_site += (pred == y).sum(0).cpu().numpy()
        for t, p in zip(y.flatten().tolist(), pred.flatten().tolist()):
            conf[t, p] += 1
    n_img = max(1, total // 4)
    out = {
        "overall_accuracy": correct / max(1, total),
        "per_site_accuracy": {k: per_site[i] / n_img for i, k in enumerate(SITE_KEYS)},
        "confusion": {"labels": list(CLASSES), "matrix": conf.tolist()},
    }
    if return_probs:
        out["probs"] = torch.cat(all_probs).numpy() if all_probs else np.zeros((0, 4, 4))
    return out


def train_one(args, train_ids, val_ids, labels, out_path: Path, device, keypoints=None) -> dict:
    if args.backbone.startswith("vlm:"):  # fine-tuned vision-language candidate (d2qual/vlm_ft.py)
        from .vlm_ft import train_vlm

        fallback = class_prior({i: labels[i] for i in train_ids})
        return train_vlm(args, train_ids, val_ids, labels, out_path.parent / f"vlm_{out_path.stem}", device, fallback)
    cfg = ModelConfig(backbone=args.backbone, image_size=args.size,
                      limb_crops=keypoints is not None, crop_size=args.crop_size)
    model = InjuryNet(cfg, pretrained=not args.no_pretrained).to(device)
    data_cfg = getattr(model.backbone, "pretrained_cfg", {}) or {}
    cfg.mean = tuple(data_cfg.get("mean", cfg.mean))
    cfg.std = tuple(data_cfg.get("std", cfg.std))

    tr = InjuryDataset(args.images, labels, train_ids, cfg.image_size, True, cfg.mean, cfg.std, args.seed,
                       keypoints=keypoints, crop_size=cfg.crop_size)
    va = InjuryDataset(args.images, labels, val_ids, cfg.image_size, False, cfg.mean, cfg.std,
                       keypoints=keypoints, crop_size=cfg.crop_size)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=len(tr) > args.batch, pin_memory=device.type == "cuda",
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers) if val_ids else None

    params = [
        {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
        {"params": model.head.parameters(), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=args.wd)
    steps = args.epochs * max(1, len(tl))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = {"overall_accuracy": -1.0}
    fallback = class_prior({i: labels[i] for i in train_ids})
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for batch in tl:
            inputs, y = _unpack(batch, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(*inputs)
                loss = crit(logits.float().reshape(-1, len(CLASSES)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        msg = f"epoch {epoch + 1}/{args.epochs} loss {running / max(1, len(tl)):.4f} ({time.time() - t0:.0f}s)"
        if vl is not None:
            m = evaluate(model, vl, device, return_probs=True)
            probs = m.pop("probs")
            msg += f" val_acc {m['overall_accuracy']:.4f}"
            if m["overall_accuracy"] > best["overall_accuracy"]:
                best = {**m, "oof": {iid: pr.tolist() for iid, pr in zip(val_ids, probs)}}
                save_checkpoint(out_path, model, fallback, {**m, "epoch": epoch + 1})
                msg += " *"
        print(msg, flush=True)

    if vl is None:  # full-data run: keep last epoch
        best = {"overall_accuracy": None, "epoch": args.epochs}
        save_checkpoint(out_path, model, fallback, best)
    return best


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of training images (flat)")
    ap.add_argument("--labels", required=True, help="label file or directory (json/jsonl/csv)")
    ap.add_argument("--out", default="model/weights")
    ap.add_argument("--backbone", default=ModelConfig.backbone)
    ap.add_argument("--size", type=int, default=ModelConfig.image_size)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=5, help="k for k-fold; 1 = train on all data, no val")
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    ap.add_argument("--group-regex", default=None,
                    help="regex on image_id; first capture group = scene/manikin id (grouped CV)")
    ap.add_argument("--groups-file", default=None, help="CSV image_id,group for grouped CV")
    ap.add_argument("--limb-crops", action="store_true",
                    help="pose-guided limb crops (runs the pose model once and caches keypoints)")
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--vlm-size", type=int, default=448, help="square input size for vlm:<model> backbones")
    ap.add_argument("--keypoints", default=None,
                    help="keypoint cache JSON (default <out>/keypoints.json; built if missing)")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = match_images(args.images, load_labels(args.labels))
    if not labels:
        raise SystemExit("no labelled images matched files on disk")
    ids = list(labels)
    print(f"{len(ids)} labelled images on {device}")
    for s, key in enumerate(SITE_KEYS):
        print(f"  {key:24s} {dict(Counter(v[s] for v in labels.values()))}")

    out = Path(args.out)
    keypoints = None
    if args.limb_crops:
        from .pose import build_cache

        kp_path = Path(args.keypoints) if args.keypoints else out / "keypoints.json"
        keypoints = build_cache(args.images, ids, kp_path, device=device)
        found = [sum(float(keypoints[i][j, 2]) >= 0.35 for i in ids) / len(ids) for j in (9, 10, 15, 16)]
        print("limb crops: share of images with wrist/ankle found (L arm, R arm, L leg, R leg):",
              " ".join(f"{f:.2f}" for f in found))
    results = {}
    if args.folds <= 1:
        results["full"] = train_one(args, ids, [], labels, out / "full.pt", device, keypoints)
    else:
        groups = load_groups(ids, args.group_regex, args.groups_file)
        if groups:
            n_groups = len(set(groups.values()))
            singles = sum(1 for g in groups.values() if g.startswith("__single__"))
            print(f"grouped CV: {n_groups} groups ({singles} ungrouped images as singletons)")
            if n_groups < args.folds:
                raise SystemExit(f"only {n_groups} groups for {args.folds} folds")
            folds = grouped_folds(ids, labels, groups, args.folds, args.seed)
        else:
            print("WARNING: plain stratified CV - near-duplicate shots of one scene can leak "
                  "across folds; pass --group-regex or --groups-file for real data")
            folds = stratified_folds(ids, labels, args.folds, args.seed)
        for i in range(args.folds):
            if args.only_fold is not None and i != args.only_fold:
                continue
            val = folds[i]
            trn = [x for j, f in enumerate(folds) if j != i for x in f]
            print(f"-- fold {i} ({len(trn)} train / {len(val)} val)")
            results[f"fold{i}"] = train_one(args, trn, val, labels, out / f"fold{i}.pt", device, keypoints)
        accs = [r["overall_accuracy"] for r in results.values() if r.get("overall_accuracy") is not None]
        if accs:
            print(f"CV overall accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        # Out-of-fold predictions: every image predicted by the one fold that never trained on it.
        # d2qual.review turns these into contact sheets of the model's mistakes.
        oof = {iid: pr for r in results.values() for iid, pr in r.pop("oof", {}).items()}
        if oof:
            preds = [make_prediction(iid, [CLASSES[int(np.argmax(site))] for site in pr])
                     for iid, pr in sorted(oof.items())]
            doc = build_document(preds, "oof", "cv", "oof@example.com")
            (out / "oof_predictions.json").write_text(json.dumps(doc, indent=1))
            (out / "oof_probs.json").write_text(json.dumps({k: np.round(v, 4).tolist() for k, v in oof.items()}))
            print(f"out-of-fold predictions for {len(oof)} images -> {out / 'oof_predictions.json'}")
    for r in results.values():
        r.pop("oof", None)
    (out / "train_report.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()

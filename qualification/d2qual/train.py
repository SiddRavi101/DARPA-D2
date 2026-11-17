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
from .model import InjuryNet, ModelConfig, flip_logits, save_checkpoint
from .schema import CLASSES, SITE_KEYS


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


@torch.no_grad()
def evaluate(model, loader, device, tta: bool = True) -> dict:
    model.eval()
    correct = total = 0
    per_site = np.zeros(4)
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = logits.softmax(-1) + flip_logits(model(torch.flip(x, dims=[3])).float()).softmax(-1)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        per_site += (pred == y).sum(0).cpu().numpy()
        for t, p in zip(y.flatten().tolist(), pred.flatten().tolist()):
            conf[t, p] += 1
    n_img = max(1, total // 4)
    return {
        "overall_accuracy": correct / max(1, total),
        "per_site_accuracy": {k: per_site[i] / n_img for i, k in enumerate(SITE_KEYS)},
        "confusion": {"labels": list(CLASSES), "matrix": conf.tolist()},
    }


def train_one(args, train_ids, val_ids, labels, out_path: Path, device) -> dict:
    cfg = ModelConfig(backbone=args.backbone, image_size=args.size)
    model = InjuryNet(cfg, pretrained=not args.no_pretrained).to(device)
    data_cfg = getattr(model.backbone, "pretrained_cfg", {}) or {}
    cfg.mean = tuple(data_cfg.get("mean", cfg.mean))
    cfg.std = tuple(data_cfg.get("std", cfg.std))

    tr = InjuryDataset(args.images, labels, train_ids, cfg.image_size, True, cfg.mean, cfg.std, args.seed)
    va = InjuryDataset(args.images, labels, val_ids, cfg.image_size, False, cfg.mean, cfg.std)
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
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
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
            m = evaluate(model, vl, device)
            msg += f" val_acc {m['overall_accuracy']:.4f}"
            if m["overall_accuracy"] > best["overall_accuracy"]:
                best = m
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
    results = {}
    if args.folds <= 1:
        results["full"] = train_one(args, ids, [], labels, out / "full.pt", device)
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
            results[f"fold{i}"] = train_one(args, trn, val, labels, out / f"fold{i}.pt", device)
        accs = [r["overall_accuracy"] for r in results.values() if r.get("overall_accuracy") is not None]
        if accs:
            print(f"CV overall accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    (out / "train_report.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()

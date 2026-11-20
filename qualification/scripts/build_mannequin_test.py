"""
BLIND TEST set from the Northeastern ACLab "Mannequin In-Bed" IRS dataset (419 photos of a
manikin lying in bed, LSP/Leeds 14-joint labels). See PLAN.md "Blind test" for the rules:
never trained on, never used to pick models or weights; run ONCE on the locked ensemble.

Labels per limb site (same rule as scripts/build_coco_proxy.py):
    no_injury     >= 2 of the limb's 3 joints are annotated visible AND inside the crop
    not_testable  otherwise (occluded, or cut out of frame)
Leeds joint names are the manikin's own sides (checked visually: for a face-up body the
"right" joints sit on the viewer's left), i.e. DARPA's anatomical laterality.
Each photo gives --variants crops: a loose full-body crop, and tight / half-body crops
that push limbs out of frame; every crop is rotated 0/90/180/270 (sides unchanged).
No wounds or amputations exist here: it tests limb finding, side and visibility only.

    python scripts/build_mannequin_test.py --src <IRS_MANNE_rawN folder> --out data/mannequin_test
Source: https://ostadabbas.sites.northeastern.edu/?p=5153 (no license posted; confirm with
the ACLab before use). Output lives under qualification/data/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2qual.schema import SITE_KEYS, build_document, make_prediction  # noqa: E402

# Leeds order (0-based): 0 R ankle, 1 R knee, 2 R hip, 3 L hip, 4 L knee, 5 L ankle,
# 6 R wrist, 7 R elbow, 8 R shoulder, 9 L shoulder, 10 L elbow, 11 L wrist, 12 neck, 13 head top
# Chains in schema.SITES order: L arm, R arm, L leg, R leg.
CHAINS = ((9, 10, 11), (8, 7, 6), (3, 4, 5), (2, 1, 0))


def limb_label(joints, crop) -> str:
    """joints: [(x, y, invisible_flag)]; flag 1 = occluded."""
    x0, y0, x1, y1 = crop
    vis = sum(1 for x, y, inv in joints if not inv and x0 <= x < x1 and y0 <= y < y1)
    return "no_injury" if vis >= 2 else "not_testable"


def make_crop(kps, W, H, kind: str, rng: random.Random):
    xs = [x for x, _, _ in kps]
    ys = [y for _, y, _ in kps]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = bx1 - bx0, by1 - by0
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    if kind == "loose":
        f = rng.uniform(1.3, 1.8)
        w, h = bw * f, bh * f
    elif kind == "tight":
        f = rng.uniform(0.75, 1.0)
        w, h = bw * f, bh * f
    else:  # "half": keep the upper or lower part of the body (other limbs leave the frame)
        w, h = bw * rng.uniform(1.2, 1.5), bh * rng.uniform(0.5, 0.65)
        cy = by0 + h / 2 - bh * 0.05 if kind == "half_top" else by1 - h / 2 + bh * 0.05
    return (max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with images/ and joints_gt.mat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=2, help="crops per photo (first is always loose)")
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from scipy.io import loadmat

    rng = random.Random(a.seed)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    J = loadmat(str(src / "joints_gt.mat"))["joints_gt"]  # 3 x 16 x N
    n = J.shape[2]
    preds, counts = [], Counter()
    for i in range(n):
        path = src / "images" / f"{i + 1:06d}.jpg"
        if not path.exists():
            continue
        kps = [(float(J[0, k, i]), float(J[1, k, i]), int(J[2, k, i])) for k in range(14)]
        img0 = Image.open(path).convert("RGB")
        W, H = img0.size
        for v in range(a.variants):
            kind = "loose" if v == 0 else rng.choice(["tight", "half_top", "half_bottom"])
            crop = make_crop(kps, W, H, kind, rng)
            labels = [limb_label([kps[j] for j in chain], crop) for chain in CHAINS]
            img = img0.crop(tuple(round(c) for c in crop))
            s = min(1.0, a.max_side / max(img.size))
            if s < 1:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
            rot = rng.choice([0, 90, 180, 270])
            if rot:
                img = img.rotate(rot, expand=True)
            name = f"manne_{i + 1:06d}_{v}.jpg"
            img.save(out / name, quality=90)
            preds.append(make_prediction(name, labels))
            counts.update(f"{SITE_KEYS[s_]}={lab}" for s_, lab in enumerate(labels))
    (out / "labels.json").write_text(json.dumps(build_document(preds, "mannequin-blind", "1", "gt@example.com"), indent=1))
    print(f"{len(preds)} blind-test images from {n} photos -> {out}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()

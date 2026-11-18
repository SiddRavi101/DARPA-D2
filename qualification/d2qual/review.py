"""
Error review: see the model's mistakes in minutes instead of hours.

    python -m d2qual.review --images /data/train --labels labels.json \
        --pred model/weights/oof_predictions.json --probs model/weights/oof_probs.json \
        --keypoints model/weights/keypoints.json --out review/

Use the OUT-OF-FOLD predictions written by `d2qual.train` (each image predicted by the
fold that never trained on it), so the mistakes are honest.

Writes to --out:
  summary.txt      overall / per-site / per-class accuracy and the confusion matrix
  errors.csv       one row per wrong site: image, site, truth, prediction, confidence
  sheets/*.jpg     contact sheets per mistake type (e.g. wound->no_injury), most confident
                   mistakes first: those are often LABEL errors worth checking by eye.
                   The limb in question is boxed (when keypoints are given) and named.
  index.html       open in a browser to page through the sheets

Everything stays local: never upload DARPA images anywhere.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from .data import load_image, load_labels, match_images
from .schema import CLASSES, SITE_KEYS

THUMB = 300
COLS = 4


def _site_box(kps, site_idx, img_size):
    if kps is None:
        return None
    from .data import limb_boxes

    boxes, _ = limb_boxes(kps, img_size)
    return boxes[site_idx]


def _thumb(path: Path, site_idx: int, caption: str, kps=None) -> Image.Image:
    try:
        img = load_image(path)
    except Exception:
        img = Image.new("RGB", (THUMB, THUMB), (60, 0, 0))
    box = _site_box(kps, site_idx, img.size)
    s = THUMB / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
    canvas = Image.new("RGB", (THUMB, THUMB + 34), (25, 25, 25))
    canvas.paste(img, ((THUMB - img.width) // 2, 0))
    d = ImageDraw.Draw(canvas)
    if box is not None:
        ox = (THUMB - img.width) // 2
        d.rectangle([box[0] * s + ox, box[1] * s, box[2] * s + ox, box[3] * s], outline=(255, 220, 0), width=3)
    for k, line in enumerate(caption.split("\n")[:2]):
        d.text((4, THUMB + 3 + 15 * k), line, fill=(235, 235, 235))
    return canvas


def _sheet(tiles: list[Image.Image], title: str) -> Image.Image:
    rows = (len(tiles) + COLS - 1) // COLS
    tw, th = THUMB, THUMB + 34
    sheet = Image.new("RGB", (COLS * tw, rows * th + 28), (10, 10, 10))
    ImageDraw.Draw(sheet).text((6, 7), title, fill=(255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % COLS) * tw, 28 + (i // COLS) * th))
    return sheet


def review(images, labels_path, pred_path, out, probs_path=None, keypoints_path=None, per_sheet=24, max_sheets=4):
    out = Path(out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    truth = match_images(images, load_labels(labels_path))
    pred = match_images(images, load_labels(pred_path))
    probs = json.loads(Path(probs_path).read_text()) if probs_path else {}
    kps = {}
    if keypoints_path:
        from .pose import load_keypoint_cache

        kps = load_keypoint_cache(keypoints_path)
    ids = sorted(set(truth) & set(pred))
    if not ids:
        raise SystemExit("no images in common between labels and predictions")

    conf = [[0] * 4 for _ in range(4)]
    per_site = Counter()
    errors = []
    for iid in ids:
        for s in range(4):
            t, p = truth[iid][s], pred[iid][s]
            conf[CLASSES.index(t)][CLASSES.index(p)] += 1
            if t == p:
                per_site[s] += 1
                continue
            c = None
            if iid in probs:
                c = float(probs[iid][s][CLASSES.index(p)])
            errors.append(dict(image_id=iid, site=SITE_KEYS[s], site_idx=s, truth=t, pred=p, confidence=c))

    total = 4 * len(ids)
    acc = 1 - len(errors) / total
    lines = [f"images {len(ids)}   site predictions {total}   overall accuracy {acc:.4f}", "",
             "per site:"]
    lines += [f"  {SITE_KEYS[s]:24s} {per_site[s] / len(ids):.4f}" for s in range(4)]
    lines += ["", "per class (recall = correct / true count):"]
    for i, c in enumerate(CLASSES):
        n = sum(conf[i])
        lines.append(f"  {c:13s} n={n:5d} recall={conf[i][i] / max(1, n):.4f}")
    lines += ["", "confusion (rows = truth, cols = prediction):", "               " + " ".join(f"{c[:11]:>12s}" for c in CLASSES)]
    lines += [f"  {c:13s}" + " ".join(f"{v:12d}" for v in conf[i]) for i, c in enumerate(CLASSES)]
    buckets = defaultdict(list)
    for e in errors:
        buckets[(e["truth"], e["pred"])].append(e)
    lines += ["", "mistake types, most common first:"]
    lines += [f"  {t} -> {p}: {len(v)}" for (t, p), v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))]
    # left/right swaps: both arms (or both legs) wrong with each other's labels
    swaps = 0
    for iid in ids:
        for a, b in ((0, 1), (2, 3)):
            t, p = truth[iid], pred[iid]
            if t[a] != t[b] and p[a] == t[b] and p[b] == t[a]:
                swaps += 1
    lines += ["", f"left/right swaps (an arm or leg pair predicted with sides exchanged): {swaps}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out / "errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "site", "truth", "pred", "confidence"])
        w.writeheader()
        for e in sorted(errors, key=lambda e: -(e["confidence"] or 0)):
            w.writerow({k: e[k] for k in w.fieldnames})

    page = ["<!doctype html><meta charset='utf-8'><title>D2 error review</title>",
            "<body style='background:#111;color:#eee;font-family:sans-serif'>",
            f"<h2>Error review: accuracy {acc:.4f} on {len(ids)} images</h2><pre>{html.escape(chr(10).join(lines))}</pre>"]
    for (t, p), errs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        errs.sort(key=lambda e: -(e["confidence"] or 0))
        page.append(f"<h3>{t} &rarr; {p} ({len(errs)})</h3>")
        for k in range(min(max_sheets, (len(errs) + per_sheet - 1) // per_sheet)):
            chunk = errs[k * per_sheet:(k + 1) * per_sheet]
            tiles = [_thumb(Path(images) / e["image_id"], e["site_idx"],
                            f"{e['image_id'][:34]}\n{e['site']}  p={e['confidence'] if e['confidence'] is None else round(e['confidence'], 2)}",
                            kps.get(e["image_id"])) for e in chunk]
            name = f"{t}__as__{p}_{k}.jpg"
            _sheet(tiles, f"truth {t}  predicted {p}  (sheet {k + 1})").save(out / "sheets" / name, quality=85)
            page.append(f"<img src='sheets/{name}' style='max-width:100%'><br>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"review written to {out}/index.html")
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="ground truth")
    ap.add_argument("--pred", required=True, help="predictions (e.g. oof_predictions.json)")
    ap.add_argument("--probs", help="oof_probs.json, to sort mistakes by confidence")
    ap.add_argument("--keypoints", help="keypoints.json, to box the limb in each thumbnail")
    ap.add_argument("--out", default="review")
    a = ap.parse_args(argv)
    review(a.images, a.labels, a.pred, a.out, a.probs, a.keypoints)


if __name__ == "__main__":
    main()

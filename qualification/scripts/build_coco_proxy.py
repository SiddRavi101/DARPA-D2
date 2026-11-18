"""
Real-photo PROXY task from COCO val2017, used only to FILTER model candidates before
DARPA's data arrives (never to pick the final model, never shipped).

For each photo with one dominant person, every limb gets a label:
    no_injury     the limb is visible  (>= 2 of shoulder/elbow/wrist or hip/knee/ankle
                  are annotated visible AND inside the crop)
    not_testable  otherwise (hidden, or cut out of frame)
Left/right are COCO's person-centric sides = DARPA's anatomical laterality. Crops are
randomly loose or tight (tight ones push limbs out of frame), and photos are rotated
0/90/180/270 so people appear "lying down" at any angle, like casualties.

What it can tell you: left/right handling, visible-vs-not-testable, robustness on real
photos, speed. What it cannot: wound / amputation (COCO has none). It also flatters
limb-crop models (the pose model learned COCO-style keypoints), so only a limb-crop LOSS
here is meaningful.

    python scripts/build_coco_proxy.py --out data/coco_proxy

Downloads val2017 images (~0.8 GB) + annotations (~0.25 GB) to --coco-dir if missing.
COCO annotations are CC BY 4.0; images keep their Flickr licenses: internal research only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2qual.schema import SITE_KEYS, build_document, make_prediction  # noqa: E402

IMG_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# COCO keypoint index chains per site, in schema.SITES order (person's own sides)
CHAINS = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))


def fetch(coco_dir: Path) -> tuple[Path, Path]:
    coco_dir.mkdir(parents=True, exist_ok=True)
    ann = coco_dir / "annotations" / "person_keypoints_val2017.json"
    imgs = coco_dir / "val2017"
    for url, done in ((ANN_ZIP, ann), (IMG_ZIP, imgs)):
        if done.exists():
            continue
        z = coco_dir / Path(url).name
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(coco_dir)
        z.unlink()
    return imgs, ann


def limb_label(kps, crop) -> str:
    x0, y0, x1, y1 = crop
    vis = 0
    for x, y, v in kps:
        if v == 2 and x0 <= x < x1 and y0 <= y < y1:
            vis += 1
    return "no_injury" if vis >= 2 else "not_testable"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-dir", default="data/coco")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-area-frac", type=float, default=0.06)
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    imgs_dir, ann_path = fetch(Path(a.coco_dir))
    coco = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in coco["images"]}
    people: dict[int, list] = {}
    for an in coco["annotations"]:
        if an["category_id"] == 1 and not an.get("iscrowd"):
            people.setdefault(an["image_id"], []).append(an)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds, counts = [], Counter()
    for iid, anns in sorted(people.items()):
        im = images[iid]
        W, H = im["width"], im["height"]
        anns.sort(key=lambda x: -x["area"])
        top = anns[0]
        if top["num_keypoints"] < 5 or top["area"] < a.min_area_frac * W * H:
            continue
        if len(anns) > 1 and anns[1]["area"] > 0.4 * top["area"]:
            continue  # no single dominant "casualty"
        bx, by, bw, bh = top["bbox"]
        cx, cy = bx + bw / 2, by + bh / 2
        f = rng.uniform(0.85, 1.1) if rng.random() < 0.3 else rng.uniform(1.2, 1.9)  # tight crops cut limbs off
        side_w, side_h = bw * f, bh * f
        crop = (max(0, cx - side_w / 2), max(0, cy - side_h / 2), min(W, cx + side_w / 2), min(H, cy + side_h / 2))
        k = top["keypoints"]
        kps = [(k[3 * j], k[3 * j + 1], k[3 * j + 2]) for j in range(17)]
        labels = [limb_label([kps[j] for j in chain], crop) for chain in CHAINS]
        img = Image.open(imgs_dir / im["file_name"]).convert("RGB").crop(tuple(round(v) for v in crop))
        s = min(1.0, a.max_side / max(img.size))
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        rot = rng.choices([0, 90, 180, 270], weights=[0.4, 0.2, 0.2, 0.2])[0]  # rotation keeps anatomical sides
        if rot:
            img = img.rotate(rot, expand=True)
        name = f"coco_{iid:012d}.jpg"
        img.save(out / name, quality=90)
        preds.append(make_prediction(name, labels))
        counts.update(f"{SITE_KEYS[s]}={lab}" for s, lab in enumerate(labels))
    (out / "labels.json").write_text(json.dumps(build_document(preds, "coco-proxy", "1", "gt@example.com"), indent=1))
    print(f"{len(preds)} proxy images -> {out}")
    for kk in sorted(counts):
        print(f"  {kk}: {counts[kk]}")


if __name__ == "__main__":
    main()

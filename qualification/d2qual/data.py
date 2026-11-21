"""
Training-data loading for the Part 2 manikin-injury task.

The DARPA training set's label file format is not fixed by the ICD, so
`load_labels` accepts every reasonable layout and normalises to:

    {image_id: [label_UE_L, label_UE_R, label_LE_L, label_LE_R]}

Key domain rule baked in here: laterality is the CASUALTY's anatomical side,
so a horizontal mirror must swap left<->right labels (FLIP_PERMUTATION).
That turns flipping from a label-corrupting augmentation into a free 2x data
multiplier, and lets inference average the original and mirrored views.

Never use random crops: cropping a limb out of frame silently changes its
true label to "not_testable". We letterbox instead.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFile, ImageFilter, ImageOps

from .schema import CLASS_TO_IDX, CLASSES, FLIP_PERMUTATION, SITE_KEYS, SITES, list_images

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Label normalisation ───────────────────────────────────────────────────────

_CLASS_ALIASES = {
    "no_injury": "no_injury", "none": "no_injury", "noinjury": "no_injury",
    "intact": "no_injury", "healthy": "no_injury", "normal": "no_injury",
    "wound": "wound", "wounded": "wound", "injury": "wound", "injured": "wound",
    "amputation": "amputation", "amputated": "amputation", "amp": "amputation",
    "not_testable": "not_testable", "nottestable": "not_testable",
    "not_visible": "not_testable", "occluded": "not_testable", "nt": "not_testable",
}

_SITE_ALIASES = {
    # canonical "<region>:<laterality>" ← many spellings
    "upper_extremity:left": ["upper_extremity_left", "left_upper_extremity", "upper_left",
                             "left_upper", "lue", "left_arm", "l_arm", "arm_left"],
    "upper_extremity:right": ["upper_extremity_right", "right_upper_extremity", "upper_right",
                              "right_upper", "rue", "right_arm", "r_arm", "arm_right"],
    "lower_extremity:left": ["lower_extremity_left", "left_lower_extremity", "lower_left",
                             "left_lower", "lle", "left_leg", "l_leg", "leg_left"],
    "lower_extremity:right": ["lower_extremity_right", "right_lower_extremity", "lower_right",
                              "right_lower", "rle", "right_leg", "r_leg", "leg_right"],
}
_COL_TO_SITE = {alias: key for key, aliases in _SITE_ALIASES.items() for alias in aliases}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def normalize_class(raw: str) -> str:
    n = _norm(raw)
    if n in _CLASS_ALIASES:
        return _CLASS_ALIASES[n]
    n2 = n.replace("_", "")
    if n2 in _CLASS_ALIASES:
        return _CLASS_ALIASES[n2]
    raise ValueError(f"unknown injury class {raw!r}")


def _site_index(region: str, laterality: str) -> int:
    key = f"{_norm(region)}:{_norm(laterality)}"
    try:
        return SITE_KEYS.index(key)
    except ValueError:
        raise ValueError(f"unknown site {region!r}/{laterality!r}")


def _from_prediction_list(preds: list) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in preds:
        labels: list[Optional[str]] = [None] * 4
        for s in p["sites"]:
            labels[_site_index(s["body_region"], s["laterality"])] = normalize_class(s["injury_type"])
        if any(l is None for l in labels):
            raise ValueError(f"{p.get('image_id')}: missing site labels")
        out[p["image_id"]] = labels  # type: ignore[assignment]
    return out


def _from_csv(path: Path) -> dict[str, list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {_norm(c): c for c in rows[0].keys()}
    id_col = next((cols[c] for c in ("image_id", "image", "filename", "file", "file_name", "img",
                                     "filepath", "file_path", "path", "image_path")
                   if c in cols), None)
    if id_col is None:
        raise ValueError(f"{path}: no image id column in {list(rows[0].keys())}")
    # DARPA's train_dataset.csv mixes image and video rows (media_format): keep images only,
    # so a video sharing a stem with a photo can never overwrite the photo's labels.
    if "media_format" in cols:
        rows = [r for r in rows if _norm(r[cols["media_format"]]) in ("image", "")]

    # Long format: one row per (image, site)
    if {"body_region", "laterality", "injury_type"} <= set(cols):
        out: dict[str, list[Optional[str]]] = {}
        for r in rows:
            labels = out.setdefault(r[id_col], [None] * 4)
            idx = _site_index(r[cols["body_region"]], r[cols["laterality"]])
            labels[idx] = normalize_class(r[cols["injury_type"]])
        bad = [k for k, v in out.items() if any(x is None for x in v)]
        if bad:
            raise ValueError(f"{path}: images missing sites: {bad[:5]}")
        return out  # type: ignore[return-value]

    # Wide format: one column per site
    site_cols: dict[str, str] = {}
    for norm_c, orig_c in cols.items():
        if norm_c in _COL_TO_SITE:
            site_cols[_COL_TO_SITE[norm_c]] = orig_c
    if len(site_cols) != 4:
        raise ValueError(f"{path}: could not map 4 site columns from {list(rows[0].keys())}")
    return {
        r[id_col]: [normalize_class(r[site_cols[k]]) for k in SITE_KEYS] for r in rows
    }


GROUP_COLUMNS = ("casualty_id", "subject_id", "manikin_id", "patient_id", "scene_id", "session_id")


def load_label_groups(path: str | Path) -> dict[str, str] | None:
    """image id -> group from a label CSV that carries a casualty/subject id column, else None."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = {_norm(c): c for c in rows[0].keys()}
    gcol = next((cols[c] for c in GROUP_COLUMNS if c in cols), None)
    id_col = next((cols[c] for c in ("image_id", "image", "filename", "file", "file_name", "img",
                                     "filepath", "file_path", "path", "image_path") if c in cols), None)
    if gcol is None or id_col is None:
        return None
    if "media_format" in cols:
        rows = [r for r in rows if _norm(r[cols["media_format"]]) in ("image", "")]
    return {r[id_col]: str(r[gcol]) for r in rows}


def load_labels(path: str | Path) -> dict[str, list[str]]:
    """Accepts predictions.json-style docs, bare prediction lists, JSONL, or CSV."""
    path = Path(path)
    if path.is_dir():
        merged: dict[str, list[str]] = {}
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in {".json", ".jsonl", ".csv"}:
                merged.update(load_labels(p))
        return merged
    suf = path.suffix.lower()
    if suf == ".csv":
        return _from_csv(path)
    if suf == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return _from_prediction_list([json.loads(l) for l in f if l.strip()])
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        return _from_prediction_list(data["predictions"])
    if isinstance(data, list):
        return _from_prediction_list(data)
    if isinstance(data, dict):  # {image_id: {site_key: class}} or {image_id: [4 classes]}
        out = {}
        for iid, v in data.items():
            if isinstance(v, list):
                out[iid] = [normalize_class(x) for x in v]
                continue
            by_site = {}
            for k, val in v.items():
                nk = _norm(k)
                site = _COL_TO_SITE.get(nk) or (k if k in SITE_KEYS else None)
                if site:
                    by_site[site] = normalize_class(val)
            if len(by_site) != 4:
                raise ValueError(f"{path}: {iid} does not map to 4 sites: {list(v)}")
            out[iid] = [by_site[sk] for sk in SITE_KEYS]
        return out
    raise ValueError(f"unrecognised label file {path}")


def flip_labels(labels: list) -> list:
    return [labels[i] for i in FLIP_PERMUTATION]


def class_prior(labels: dict[str, list[str]]) -> list[str]:
    """Most frequent class per site — used as the crash fallback in inference."""
    out = []
    for s in range(4):
        counts = {c: 0 for c in CLASSES}
        for v in labels.values():
            counts[v[s]] += 1
        out.append(max(CLASSES, key=lambda c: counts[c]))
    return out


# ── Image preprocessing ───────────────────────────────────────────────────────

def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # phone photos: honour EXIF rotation
    return img.convert("RGB")


def letterbox(img: Image.Image, size: int, fill=(124, 116, 104)) -> Image.Image:
    """Resize longest side to `size`, pad to square. Never crops a limb out."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    from PIL import ImageEnhance

    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.5)))
    if rng.random() < 0.5:
        # mild rotation / zoom-out only (zoom-in could push limbs out of frame)
        angle = rng.uniform(-12, 12)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(124, 116, 104))
    return img


# ── Pose-guided limb crops ────────────────────────────────────────────────────
# For each site, a square close-up around shoulder->elbow->wrist (or hip->knee->ankle),
# extended past the wrist/ankle to include the hand/foot. Crops are cut from the ORIGINAL
# image (before global augmentation), so keypoints never need re-projecting. A limb the
# pose model can't see gets a blank crop and zero confidences: the model learns that a
# missing limb is evidence for not_testable / amputation, and falls back to the global view.

KP_CONF_MIN = 0.35
# Per-site pose evidence fed to the classifier (Approach B in Rohan's plan):
#   [conf proximal, conf middle, conf distal, crop present, anchor-only crop, fraction of crop outside image]
# "anchor-only" = only the shoulder/hip was found: likely a stump or a hidden limb. The out-of-frame
# fraction separates "limb runs off the image" (not_testable) from "limb ends at a stump" (amputation).
N_KP_FEATS = 6
LEFT_SITES = (0, 2)  # SITES order: left arm, right arm, left leg, right leg
_FILL = (124, 116, 104)


def _body_scale(kps) -> float:
    import numpy as np

    pairs = [(5, 11), (6, 12), (5, 6), (11, 12)]  # shoulders-hips, shoulder-shoulder, hip-hip
    d = [float(np.hypot(*(kps[a, :2] - kps[b, :2]))) for a, b in pairs
         if kps[a, 2] >= KP_CONF_MIN and kps[b, 2] >= KP_CONF_MIN]
    return max(d) if d else 0.0


def _outside_fraction(box, w, h) -> float:
    x0, y0, x1, y1 = box
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    ix = max(0.0, min(x1, w) - max(x0, 0.0))
    iy = max(0.0, min(y1, h) - max(y0, 0.0))
    return float(1.0 - ix * iy / area)


def limb_boxes(kps, img_size):
    """kps (17,3) in image pixels -> 4 boxes (x0,y0,x1,y1) or None, plus (4, N_KP_FEATS) features.

    Boxes are anchored on the proximal joint (shoulder / hip), which survives amputation: when only
    the anchor is found, the crop is a square around it big enough to hold a stump, so the
    classifier still sees blood / a severed end instead of a blank tile.
    """
    import numpy as np

    from .pose import KP, LIMB_CHAINS

    w, h = img_size
    scale = _body_scale(kps) or 0.25 * max(w, h)
    boxes, feats = [], np.zeros((4, N_KP_FEATS), np.float32)
    for s, chain in enumerate(LIMB_CHAINS):
        idx = [KP[n] for n in chain]
        conf = kps[idx, 2]
        feats[s, :3] = conf
        vis = [i for i, c in zip(idx, conf) if c >= KP_CONF_MIN]
        if len(vis) < 2:
            if conf[0] >= KP_CONF_MIN:  # anchor only: stump or hidden limb
                cx, cy = kps[idx[0], :2].astype(np.float64)
                side = 0.9 * scale
                box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
                boxes.append(box)
                feats[s, 3], feats[s, 4], feats[s, 5] = 1.0, 1.0, _outside_fraction(box, w, h)
            else:
                boxes.append(None)
            continue
        pts = kps[vis, :2].astype(np.float64)
        if kps[idx[2], 2] >= KP_CONF_MIN and kps[idx[1], 2] >= KP_CONF_MIN:  # add the hand / foot
            d = kps[idx[2], :2] - kps[idx[1], :2]
            pts = np.vstack([pts, kps[idx[2], :2] + d * 0.45])
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        side = max(x1 - x0, y1 - y0) + 0.5 * scale
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        boxes.append(box)
        feats[s, 3], feats[s, 5] = 1.0, _outside_fraction(box, w, h)
    return boxes, feats


def crop_box(img: Image.Image, box, size: int, rng: Optional[random.Random] = None) -> Image.Image:
    """Square crop (padded where it leaves the image), optionally jittered for training."""
    if box is None:
        return Image.new("RGB", (size, size), _FILL)
    x0, y0, x1, y1 = box
    if rng is not None:  # small shift / zoom; never large enough to lose the limb
        side = x1 - x0
        dx, dy = rng.uniform(-0.08, 0.08) * side, rng.uniform(-0.08, 0.08) * side
        z = rng.uniform(0.92, 1.12)
        cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
        half = side * z / 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    # PIL pads out-of-bounds regions with black, identically at train and test time
    return img.crop((round(x0), round(y0), round(x1), round(y1))).resize((size, size), Image.BICUBIC)


def limb_crops(img: Image.Image, kps, size: int, rng: Optional[random.Random] = None):
    """-> (list of 4 PIL crops in SITES order, (4, N_KP_FEATS) features)."""
    import numpy as np

    if kps is None:
        kps = np.zeros((17, 3), np.float32)
    boxes, feats = limb_boxes(kps, img.size)
    crops = [crop_box(img, b, size, rng) for b in boxes]
    # Canonical orientation: left-limb crops are mirrored to look like right limbs, so the shared
    # classifier only learns "arm" / "leg"; laterality comes from pose, not the classifier.
    crops = [ImageOps.mirror(c) if s in LEFT_SITES else c for s, c in enumerate(crops)]
    return crops, feats


def mirror_limb_inputs(crops: list, feats):
    """Mirror the photo => anatomical sides swap. With canonical (left-mirrored) crops this is a
    pure site permutation: the new left crop is mirror(mirror(old right)) = the old right crop."""
    return [crops[i] for i in FLIP_PERMUTATION], feats[list(FLIP_PERMUTATION)]


def to_tensor(img: Image.Image, mean, std):
    import numpy as np
    import torch

    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


# ── Torch dataset ─────────────────────────────────────────────────────────────

class InjuryDataset:
    """
    Map-style dataset; torch is imported lazily so schema/validator stay torch-free.
    Returns (x, y), or (x, crops, kp_feats, y) when `keypoints` is given (limb-crop models).
    """

    def __init__(self, image_dir, labels: dict[str, list[str]], image_ids: list[str],
                 size: int, train: bool, mean, std, seed: int = 0,
                 keypoints: Optional[dict] = None, crop_size: int = 224):
        self.image_dir = Path(image_dir)
        self.labels = labels
        self.ids = image_ids
        self.size = size
        self.train = train
        self.mean, self.std = mean, std
        self.rng = random.Random(seed)
        self.keypoints = keypoints
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        import torch

        iid = self.ids[i]
        orig = load_image(self.image_dir / iid)
        lab = list(self.labels[iid])
        crops = feats = None
        if self.keypoints is not None:
            crops, feats = limb_crops(orig, self.keypoints.get(iid), self.crop_size,
                                      self.rng if self.train else None)
        img = orig
        if self.train:
            img = _augment(img, self.rng)
            if crops is not None:
                crops = [_augment_crop(c, self.rng) for c in crops]
            if self.rng.random() < 0.5:
                img = ImageOps.mirror(img)
                lab = flip_labels(lab)
                if crops is not None:
                    crops, feats = mirror_limb_inputs(crops, feats)
        x = to_tensor(letterbox(img, self.size), self.mean, self.std)
        y = torch.tensor([CLASS_TO_IDX[c] for c in lab], dtype=torch.long)
        if crops is None:
            return x, y
        c = torch.stack([to_tensor(cr, self.mean, self.std) for cr in crops])
        return x, c, torch.from_numpy(feats), y


def _augment_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    from PIL import ImageEnhance

    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    return img


def match_images(image_dir, labels: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Re-key labels by the exact on-disk filename (case-insensitive, and tolerant
    of label files that omit the extension). Unmatched labels are dropped.
    """
    on_disk = [p.name for p in list_images(image_dir)]
    by_lower = {n.lower(): n for n in on_disk}
    by_stem = {Path(n).stem.lower(): n for n in on_disk}
    out: dict[str, list[str]] = {}
    for iid, lab in labels.items():
        name = by_lower.get(iid.lower()) or by_stem.get(Path(iid).stem.lower())
        if name:
            out[name] = lab
    return dict(sorted(out.items()))

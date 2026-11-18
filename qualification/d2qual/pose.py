"""
Casualty pose: find shoulders/elbows/wrists/hips/knees/ankles and which ANATOMICAL side
each belongs to, so the classifier can look at a close-up of every limb.

Backend: torchvision Keypoint R-CNN (COCO keypoints, BSD-licensed code, no extra pip
dependency: torchvision ships in the PyTorch base image). COCO's left/right are the
person's own sides, which is exactly the ICD's laterality.

Casualties lie on the ground at any angle and pose models are trained mostly on upright
people, so each image is tried at 0/90/270 degrees and the most confident result wins.

The container never downloads anything: weights are baked into model/pose/ with
    python -m d2qual.pose --save-weights model/pose/keypointrcnn.pth
Training images are processed once and cached:
    python -m d2qual.pose --images /data/train --out keypoints.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger("d2qual")

N_KP = 17
# COCO indices (person's own left/right)
KP = dict(nose=0, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
          l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)
# Site order matches schema.SITES: (UE,L), (UE,R), (LE,L), (LE,R)
LIMB_CHAINS = (("l_sh", "l_el", "l_wr"), ("r_sh", "r_el", "r_wr"),
               ("l_hip", "l_kn", "l_an"), ("r_hip", "r_kn", "r_an"))
WEIGHTS_NAME = "keypointrcnn.pth"


def empty_keypoints() -> np.ndarray:
    return np.zeros((N_KP, 3), np.float32)


class PoseExtractor:
    def __init__(self, device, weights: str | Path | None = None, rotations=(0, 90, 270), max_side: int = 800):
        import torch
        import torchvision
        from torchvision.models.detection import keypointrcnn_resnet50_fpn

        self.torch = torch
        self.device = device
        self.rotations = rotations
        self.max_side = max_side
        if weights and Path(weights).exists():
            model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        else:  # training box with internet only; the container always passes baked weights
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
            model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model = model.eval().to(device)
        log.info("pose model ready (torchvision %s)", torchvision.__version__)

    @staticmethod
    def try_load(weights_dir: str | Path, device) -> "PoseExtractor | None":
        w = Path(weights_dir) / "pose" / WEIGHTS_NAME
        if not w.exists():
            w2 = Path(weights_dir).parent / "pose" / WEIGHTS_NAME  # model/weights -> model/pose
            w = w2 if w2.exists() else w
        if not w.exists():
            log.error("pose weights not found next to %s; limb crops disabled", weights_dir)
            return None
        try:
            if getattr(device, "type", str(device)) == "cuda":
                return PoseExtractor(device, w)
            # CPU fallback: one orientation, smaller input, so the 1-hour budget still holds
            return PoseExtractor(device, w, rotations=(0,), max_side=512)
        except Exception as e:  # never fatal: the model also has a whole-image path
            log.exception("pose model failed to load: %s", e)
            return None

    def _run(self, imgs):
        torch = self.torch
        ts = [torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).to(self.device) for im in imgs]
        with torch.inference_mode():
            return self.model(ts)

    def extract(self, images) -> list[np.ndarray]:
        """PIL images -> list of (17, 3) arrays [x, y, conf in 0..1] in each image's own pixel frame."""
        from PIL import Image

        best = [(-1.0, empty_keypoints()) for _ in images]
        smalls, scales = [], []
        for im in images:
            s = min(1.0, self.max_side / max(im.size))
            smalls.append(im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR) if s < 1 else im)
            scales.append(s)
        for rot in self.rotations:
            rotated = [sm.rotate(rot, expand=True) if rot else sm for sm in smalls]
            outs = self._run(rotated)
            for i, (o, sm, rim) in enumerate(zip(outs, smalls, rotated)):
                if len(o["boxes"]) == 0:
                    continue
                boxes = o["boxes"].cpu().numpy()
                det = o["scores"].cpu().numpy()
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                j = int(np.argmax(det * np.sqrt(area)))  # the casualty: confident and large
                kp = o["keypoints"][j].cpu().numpy()[:, :2]
                conf = 1 / (1 + np.exp(-o["keypoints_scores"][j].cpu().numpy() / 2))  # logits -> 0..1
                quality = float(det[j]) * float(np.mean(conf))
                if quality <= best[i][0]:
                    continue
                pts = _unrotate(kp, rot, sm.size, rim.size) / scales[i]
                best[i] = (quality, np.concatenate([pts, conf[:, None]], 1).astype(np.float32))
        return [b[1] for b in best]


def _unrotate(pts: np.ndarray, rot: int, orig_size, rot_size) -> np.ndarray:
    """Map points from an image rotated counter-clockwise by `rot` (PIL, expand=True) back."""
    if rot == 0:
        return pts.copy()
    w, h = orig_size
    W, Hh = rot_size
    th = math.radians(rot)
    # PIL rotates CCW about the centre; inverse is a CW rotation about the centre.
    c, s = math.cos(th), math.sin(th)
    x = pts[:, 0] - W / 2
    y = pts[:, 1] - Hh / 2
    ox = c * x - s * y + w / 2
    oy = s * x + c * y + h / 2
    return np.stack([ox, oy], 1)


def save_weights(path: str | Path) -> None:
    import torch
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

    m = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), path)
    print(f"saved {path} ({Path(path).stat().st_size / 1e6:.0f} MB)")


def load_keypoint_cache(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.asarray(v, np.float32).reshape(N_KP, 3) for k, v in raw.items()}


def build_cache(image_dir, image_ids, out_path, device=None, batch: int = 8, weights=None) -> dict[str, np.ndarray]:
    """Run pose once over training images; reuse an existing cache for images already done."""
    import torch

    from .data import load_image

    out_path = Path(out_path)
    cache = load_keypoint_cache(out_path) if out_path.exists() else {}
    todo = [i for i in image_ids if i not in cache]
    if todo:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ext = PoseExtractor(device, weights)
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            ims = [load_image(Path(image_dir) / i) for i in chunk]
            for i, kp in zip(chunk, ext.extract(ims)):
                cache[i] = kp
            if (k // batch) % 25 == 0:
                print(f"pose {min(k + batch, len(todo))}/{len(todo)}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({k: np.round(v, 2).tolist() for k, v in cache.items()}))
    return cache


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images")
    ap.add_argument("--out", help="keypoint cache JSON")
    ap.add_argument("--save-weights", help="download COCO Keypoint R-CNN weights to this path and exit")
    args = ap.parse_args(argv)
    if args.save_weights:
        save_weights(args.save_weights)
        return
    from .schema import list_images

    ids = [p.name for p in list_images(args.images)]
    build_cache(args.images, ids, args.out)
    print(f"cached keypoints for {len(ids)} images -> {args.out}")


if __name__ == "__main__":
    main()

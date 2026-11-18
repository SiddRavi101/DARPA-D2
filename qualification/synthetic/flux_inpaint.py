"""
Photorealistic synthetic casualties with labels that are correct by construction.

Text-to-image alone gets the injuries wrong (amputations come out as wounds, sides get
swapped). So each image is built in steps and the label comes from the edit we applied,
not from the prompt:

  1. FLUX.1-schnell draws an UNINJURED full-body manikin in a random scene/viewpoint.
  2. A pose model (YOLO11-pose, COCO keypoints) finds shoulders/elbows/wrists/hips/
     knees/ankles. COCO left/right are the PERSON's left/right = the ICD's anatomical side.
     Bases where any limb keypoint is missing or low-confidence are thrown away.
  3. Each limb gets a class and a masked FLUX inpaint:
       wound        -> small mask on one limb segment, "moulage laceration/burn/..." prompt
       amputation   -> (a) erase the distal limb with a background-only prompt,
                       (b) paint a stump at the cut point
       not_testable -> cover the whole limb with a blanket
       no_injury    -> untouched
  4. The pose model runs again on the result as an automatic check (an amputated or
     covered limb should lose its distal keypoint; others should keep theirs).

    python synthetic/flux_inpaint.py --out data/flux --n 20 --seed 0

Writes <out>/<id>.jpg, labels.json (ICD format), meta.json (prompts, masks, checks),
and <out>/debug/<id>_base.jpg for side-by-side review.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2qual.schema import SITE_KEYS, build_document, make_prediction  # noqa: E402

W, H = 1024, 768
MODEL = "black-forest-labs/FLUX.1-schnell"  # Apache-2.0, not gated

# COCO keypoint indices (person's own left/right)
KP = dict(l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10, l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)
# site key -> (proximal, middle, distal) keypoints
LIMBS = {
    "upper_extremity:left": ("l_sh", "l_el", "l_wr"),
    "upper_extremity:right": ("r_sh", "r_el", "r_wr"),
    "lower_extremity:left": ("l_hip", "l_kn", "l_an"),
    "lower_extremity:right": ("r_hip", "r_kn", "r_an"),
}

GROUNDS = ["dry dirt and gravel", "trampled grass", "a cracked concrete floor", "sand", "a muddy forest floor with leaves",
           "asphalt", "broken concrete rubble", "a dusty wooden floor", "a gym floor mat"]
CLOTHES = ["a torn camouflage uniform", "camouflage trousers and a tan t-shirt", "a tactical vest over a camouflage uniform",
           "shorts and a grey t-shirt", "camouflage trousers, bare chest", "a dark civilian jacket and jeans"]
VIEWS = ["photographed from directly overhead", "photographed from standing height at a slight angle",
         "photographed from standing height near the feet", "photographed from a low oblique angle beside the body",
         "photographed from an elevated drone-like angle"]
LIGHTS = ["overcast daylight", "bright midday sun with hard shadows", "late afternoon light", "dim indoor light with camera flash",
          "fluorescent indoor lighting", "hazy dusty daylight"]
POSES = ["lying flat on its back, arms slightly away from the body, legs apart",
         "lying on its back with one knee bent, arms at its sides",
         "lying on its back, arms spread out, legs straight"]
WOUNDS = {
    "upper": ["a deep bloody laceration", "a severe burn with blistered charred skin", "multiple bloody shrapnel wounds",
              "an open compound fracture with bone protruding through bloody skin"],
    "lower": ["a deep bloody gash", "a severe burn with blistered charred skin", "multiple bloody shrapnel wounds",
              "an open compound fracture of the shin with bone protruding", "a large bloody gunshot wound"],
}
BLANKETS = ["olive drab wool blanket", "grey wool blanket", "silver emergency foil blanket", "dark green poncho", "brown tarp"]
CLASS_W = {"no_injury": 0.35, "wound": 0.3, "amputation": 0.22, "not_testable": 0.13}


def load_pipes(device):
    from diffusers import FluxInpaintPipeline, FluxPipeline

    t2i = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    inp = FluxInpaintPipeline(**t2i.components)  # shares the same weights, no extra VRAM
    return t2i, inp


def detect(pose_model, img: Image.Image):
    """Return (17,3) keypoints [x,y,conf] for the largest person, or None."""
    r = pose_model(np.ascontiguousarray(np.array(img)[:, :, ::-1]), verbose=False)[0]  # RGB -> BGR
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    boxes = r.boxes.xywh.cpu().numpy()
    i = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
    xy = r.keypoints.xy[i].cpu().numpy()
    conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
    return np.concatenate([xy, conf[:, None]], 1)


def pt(kps, name):
    return np.array(kps[KP[name], :2], dtype=float)


def body_scale(kps) -> float:
    sh = (pt(kps, "l_sh") + pt(kps, "r_sh")) / 2
    hip = (pt(kps, "l_hip") + pt(kps, "r_hip")) / 2
    return float(np.linalg.norm(sh - hip)) or 100.0


def capsule_mask(points, width, extend_end=0.0, blur=6) -> Image.Image:
    """Thick polyline mask along points; optionally extend past the last point (hands/feet)."""
    pts = [np.array(p, float) for p in points]
    if extend_end:
        d = pts[-1] - pts[-2]
        pts[-1] = pts[-1] + d / (np.linalg.norm(d) + 1e-6) * extend_end
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    w = int(max(8, width))
    dr.line([tuple(p) for p in pts], fill=255, width=w, joint="curve")
    for p in (pts[0], pts[-1]):
        dr.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=255)
    return m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(blur))


def disc_mask(center, r, blur=5) -> Image.Image:
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=255)
    return m.filter(ImageFilter.GaussianBlur(blur))


def inpaint(pipe, img, mask, prompt, seed, strength=0.95, steps=8):
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(prompt=prompt, image=img, mask_image=mask, height=H, width=W, strength=strength,
               num_inference_steps=steps, guidance_scale=0.0, max_sequence_length=256, generator=g).images[0]
    return out


def make_base(t2i, pose_model, rng, seed, max_tries=6):
    for t in range(max_tries):
        scene = dict(ground=rng.choice(GROUNDS), clothes=rng.choice(CLOTHES), view=rng.choice(VIEWS),
                     light=rng.choice(LIGHTS), pose=rng.choice(POSES))
        prompt = (f"Realistic documentary photo of an adult full-body medical CPR trauma training manikin {scene['pose']} on "
                  f"{scene['ground']}, wearing {scene['clothes']}. The whole body from head to feet is in frame, both arms "
                  f"and both legs fully visible, intact and uninjured. {scene['view']}, {scene['light']}, DSLR photo, sharp focus.")
        g = torch.Generator("cuda").manual_seed(seed * 100 + t)
        img = t2i(prompt=prompt, height=H, width=W, num_inference_steps=4, guidance_scale=0.0,
                  max_sequence_length=256, generator=g).images[0]
        kps = detect(pose_model, img)
        if kps is None:
            continue
        need = [KP[k] for k in KP]
        if kps[need, 2].min() < 0.5:
            continue
        return img, kps, scene, prompt
    return None


def sample_classes(rng):
    while True:
        cls = {k: rng.choices(list(CLASS_W), weights=list(CLASS_W.values()))[0] for k in SITE_KEYS}
        if any(v != "no_injury" for v in cls.values()) or rng.random() < 0.15:
            return cls


def apply_limb(inp, img, kps, site, cls, scene, rng, seed, log):
    prox, mid, dist = (pt(kps, n) for n in LIMBS[site])
    upper = site.startswith("upper")
    part = "arm" if upper else "leg"
    s = body_scale(kps)
    width = s * (0.22 if upper else 0.30)
    ground = scene["ground"]

    if cls == "wound":
        seg = rng.choice([(prox, mid), (mid, dist)])
        c = seg[0] + (seg[1] - seg[0]) * rng.uniform(0.35, 0.65)
        kind = rng.choice(WOUNDS["upper" if upper else "lower"])
        prompt = (f"close-up of a medical training manikin {part} with {kind}, realistic moulage special-effects makeup, "
                  f"glossy fake blood, torn clothing around the wound, photo")
        img = inpaint(inp, img, disc_mask(c, width * 0.85), prompt, seed, strength=0.9)
        log[site] = dict(cls=cls, kind=kind, center=c.round(1).tolist())

    elif cls == "amputation":
        # cut between the middle joint and the distal joint (below elbow/knee) or above it
        if rng.random() < 0.65:
            cut = mid + (dist - mid) * rng.uniform(0.25, 0.55)
            removed = [cut, dist]
            level = "below the elbow" if upper else "below the knee"
        else:
            cut = prox + (mid - prox) * rng.uniform(0.45, 0.7)
            removed = [cut, mid, dist]
            level = "above the elbow" if upper else "above the knee"
        extend = s * (0.35 if upper else 0.45)  # hand / foot beyond the distal keypoint
        erase = capsule_mask(removed, width * 1.6, extend_end=extend, blur=8)
        img = inpaint(inp, img, erase, f"empty {ground}, bare ground with nothing on it, photo", seed, strength=1.0)
        stump = disc_mask(cut, width * 0.95)
        prompt = (f"severed manikin {part} amputated {level}, ending in a ragged bloody amputation stump with exposed bone and torn "
                  f"flesh, realistic moulage, fake blood pooling on the {ground}, tourniquet above the stump, photo")
        img = inpaint(inp, img, stump, prompt, seed + 1, strength=0.92)
        log[site] = dict(cls=cls, level=level, cut=cut.round(1).tolist())

    elif cls == "not_testable":
        extend = s * (0.35 if upper else 0.45)
        cover = capsule_mask([prox, mid, dist], width * 2.2, extend_end=extend, blur=10)
        blanket = rng.choice(BLANKETS)
        img = inpaint(inp, img, cover, f"a crumpled {blanket} lying on the {ground}, completely covering, photo", seed, strength=1.0)
        log[site] = dict(cls=cls, blanket=blanket)
    else:
        log[site] = dict(cls=cls)
    return img


def auto_check(pose_model, img, classes) -> dict:
    """Distal keypoint should vanish for amputation/not_testable and stay for the rest."""
    kps = detect(pose_model, img)
    res = {}
    for site, cls in classes.items():
        d = KP[LIMBS[site][2]]
        conf = float(kps[d, 2]) if kps is not None else 0.0
        present = conf >= 0.5
        res[site] = dict(distal_conf=round(conf, 3),
                         ok=(not present) if cls in ("amputation", "not_testable") else present)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="flux_")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    out = Path(args.out)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t2i, inp = load_pipes("cuda")
    pose_model = YOLO("yolo11x-pose.pt")

    preds, meta = [], []
    i = made = 0
    t0 = time.time()
    while made < args.n and i < args.n * 4:
        seed = args.seed * 10000 + i
        i += 1
        base = make_base(t2i, pose_model, rng, seed)
        if base is None:
            print(f"[{i}] no usable base (pose check failed)", flush=True)
            continue
        img, kps, scene, prompt = base
        classes = sample_classes(rng)
        name = f"{args.prefix}{made:05d}.jpg"
        img.save(out / "debug" / name.replace(".jpg", "_base.jpg"), quality=88)
        log = {}
        for k, site in enumerate(SITE_KEYS):
            img = apply_limb(inp, img, kps, site, classes[site], scene, rng, seed * 10 + k, log)
        check = auto_check(pose_model, img, classes)
        img.save(out / name, quality=90)
        labels = [classes[k] for k in SITE_KEYS]
        preds.append(make_prediction(name, labels))
        meta.append(dict(image_id=name, base_prompt=prompt, scene=scene, edits=log, auto_check=check,
                         base_keypoints=kps.round(1).tolist()))
        made += 1
        n_ok = sum(v["ok"] for v in check.values())
        print(f"[{made}/{args.n}] {name} {labels} auto_check {n_ok}/4  ({time.time() - t0:.0f}s)", flush=True)

    (out / "labels.json").write_text(json.dumps(build_document(preds, "synthetic-gt", "1", "gt@example.com"), indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    total = sum(len(m["auto_check"]) for m in meta)
    ok = sum(v["ok"] for m in meta for v in m["auto_check"].values())
    print(f"done: {made} images, auto-check {ok}/{total} limbs consistent, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

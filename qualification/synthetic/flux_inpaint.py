"""
Photorealistic synthetic casualties with labels that are correct by construction.

Text-to-image alone gets the injuries wrong (amputations come out as wounds, sides get
swapped). So each image is built in steps and the label comes from the edit we applied,
not from the prompt:

  1. A photoreal text-to-image model (RealVisXL by default, or FLUX.1-schnell) draws an UNINJURED full-body manikin in a random scene/viewpoint.
  2. A pose model (YOLO11-pose, COCO keypoints) finds shoulders/elbows/wrists/hips/
     knees/ankles. COCO left/right are the PERSON's left/right = the ICD's anatomical side.
     Bases where any limb keypoint is missing or low-confidence are thrown away.
  3. Each limb gets a class and a masked inpaint:
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
# Backends. "sdxl" (default) needs no Hugging Face token: RealVisXL is ungated, openrail++.
# "flux" needs a token with the FLUX.1-schnell terms accepted (the repo is gated).
FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
SDXL_T2I = "SG161222/RealVisXL_V5.0"
SDXL_INPAINT = "OzzyGT/RealVisXL_V4.0_inpainting"
NEG = ("cartoon, drawing, illustration, 3d render, cgi, deformed, extra limbs, missing limbs, fused fingers, "
       "text, watermark, blurry")
BACKEND = "sdxl"

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
    if BACKEND == "flux":
        from diffusers import FluxInpaintPipeline, FluxPipeline

        t2i = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16).to(device)
        inp = FluxInpaintPipeline(**t2i.components)  # shares the same weights, no extra VRAM
        return t2i, inp
    from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline

    def load(cls, repo):
        try:
            return cls.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16").to(device)
        except (OSError, ValueError):  # repo has no fp16 variant files
            return cls.from_pretrained(repo, torch_dtype=torch.float16).to(device)

    return load(StableDiffusionXLPipeline, SDXL_T2I), load(StableDiffusionXLInpaintPipeline, SDXL_INPAINT)


def text2img(pipe, prompt, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    if BACKEND == "flux":
        return pipe(prompt=prompt, height=H, width=W, num_inference_steps=4, guidance_scale=0.0,
                    max_sequence_length=256, generator=g).images[0]
    return pipe(prompt=prompt, negative_prompt=NEG, height=H, width=W, num_inference_steps=30,
                guidance_scale=5.5, generator=g).images[0]


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


def inpaint(pipe, img, mask, prompt, seed, strength=0.95):
    g = torch.Generator("cuda").manual_seed(seed)
    if BACKEND == "flux":
        return pipe(prompt=prompt, image=img, mask_image=mask, height=H, width=W, strength=strength,
                    num_inference_steps=8, guidance_scale=0.0, max_sequence_length=256, generator=g).images[0]
    out = pipe(prompt=prompt, negative_prompt=NEG, image=img, mask_image=mask, height=H, width=W,
               strength=min(strength, 0.99), num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
    # keep pixels outside the mask exactly as they were (SDXL's VAE round-trip shifts them slightly)
    return Image.composite(out, img, mask)


def make_base(t2i, pose_model, rng, seed, max_tries=6):
    for t in range(max_tries):
        scene = dict(ground=rng.choice(GROUNDS), clothes=rng.choice(CLOTHES), view=rng.choice(VIEWS),
                     light=rng.choice(LIGHTS), pose=rng.choice(POSES))
        # SDXL's text encoder reads ~77 tokens, so the important words come first
        prompt = (f"RAW photo, full body medical trauma training manikin {scene['pose']} on {scene['ground']}, "
                  f"wearing {scene['clothes']}, entire body from head to feet in frame, both arms and both legs visible, "
                  f"{scene['view']}, {scene['light']}, DSLR")
        img = text2img(t2i, prompt, seed * 100 + t)
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


# --------------------------------------------------------------------------- crude "paint" priors
# A diffusion inpaint on its own redraws whatever the surroundings suggest (an intact limb), so
# each edit is first painted crudely into the pixels, then made photorealistic at partial
# strength. At strength < 1 the model keeps the painted layout (no hand where we erased it,
# a red stump where we drew one) and only fixes texture and lighting.

def _np(img):
    return np.array(img).astype(np.float32)


def _img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _blend(img, layer_rgb, alpha):
    a = _np(img)
    al = (np.asarray(alpha).astype(np.float32) / 255.0)[..., None]
    return _img(a * (1 - al) + layer_rgb * al)


def _blob(center, r, rng, n=14, jitter=0.35):
    pts = []
    for k in range(n):
        ang = 2 * math.pi * k / n + rng.uniform(-0.15, 0.15)
        rr = r * rng.uniform(1 - jitter, 1 + jitter * 0.4)
        pts.append((float(center[0] + rr * math.cos(ang)), float(center[1] + rr * math.sin(ang))))
    return pts


def _noise(rng, scale=18.0):
    g = np.random.default_rng(rng.randrange(1 << 30))
    small = g.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
    big = np.array(Image.fromarray((small * 40 + 128).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC))
    smooth = (big.astype(np.float32) - 128) / 40 * scale
    return smooth[..., None] + g.normal(0, scale * 0.4, (H, W, 1)).astype(np.float32)


def _solid(rgb):
    return np.broadcast_to(np.array(rgb, np.float32), (H, W, 3))


def paint_wound(img, c, r, kind, rng):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(_blob(c, r, rng), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(3))
    base = (70, 38, 28) if "burn" in kind else (120, 12, 16)
    img = _blend(img, _solid(base) + _noise(rng, 22), m)
    core = Image.new("L", (W, H), 0)
    ImageDraw.Draw(core).polygon(_blob(c, r * 0.45, rng), fill=200)
    img = _blend(img, _solid((55, 0, 6)) + _noise(rng, 10), core.filter(ImageFilter.GaussianBlur(2)))
    d = ImageDraw.Draw(img)
    if "fracture" in kind:
        a = rng.uniform(0, math.pi)
        d.line([(c[0] - r * 0.5 * math.cos(a), c[1] - r * 0.5 * math.sin(a)),
                (c[0] + r * 0.6 * math.cos(a), c[1] + r * 0.6 * math.sin(a))],
               fill=(228, 220, 200), width=max(4, int(r * 0.22)))
    if "shrapnel" in kind:
        for _ in range(rng.randint(4, 8)):
            p = (c[0] + rng.uniform(-r, r), c[1] + rng.uniform(-r, r))
            q = rng.uniform(r * 0.08, r * 0.18)
            d.ellipse([p[0] - q, p[1] - q, p[0] + q, p[1] + q], fill=(60, 0, 5))
    return img, m


def body_hull_mask(kps, pad) -> np.ndarray:
    """Boolean mask of the casualty's body (convex hull of all keypoints, padded)."""
    import cv2

    pts = kps[kps[:, 2] > 0.3, :2].astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    k = max(3, int(pad) | 1)
    return cv2.dilate(m, np.ones((k, k), np.uint8)) > 0


def remove_region(img, mask_l, kps, pad, rng):
    """
    Wipe a limb out by pasting ground copied from elsewhere in the same photo (shifted so the
    source patch lies off the body). Falls back to classical Telea inpainting if no clean patch
    fits. Either way the limb pixels are gone before diffusion sees them.
    """
    import cv2

    a = np.array(img)
    hard = np.array(mask_l) > 40
    ys, xs = np.nonzero(hard)
    if len(xs) == 0:
        return img
    body = body_hull_mask(kps, pad)
    shifts = [(dx, dy) for dx in range(-700, 701, 50) for dy in range(-600, 601, 50) if abs(dx) + abs(dy) >= 150]
    rng.shuffle(shifts)
    al = (np.array(mask_l.filter(ImageFilter.GaussianBlur(4))).astype(np.float32) / 255)[..., None]
    for dx, dy in shifts:  # one clean shift for the whole region (most coherent)
        sx, sy = xs + dx, ys + dy
        if sx.min() < 0 or sy.min() < 0 or sx.max() >= W or sy.max() >= H or body[sy, sx].any():
            continue
        out = a.copy()
        out[ys, xs] = a[sy, sx]
        return _img(a * (1 - al) + out * al)
    # region too big for one shift: fill tile by tile, reusing the last good shift when possible
    out, left = a.copy(), hard.copy()
    T, last = 48, None
    for ty in range(0, H, T):
        for tx in range(0, W, T):
            tm = hard[ty:ty + T, tx:tx + T]
            if not tm.any():
                continue
            yy, xx = np.nonzero(tm)
            yy, xx = yy + ty, xx + tx
            for dx, dy in ([last] if last else []) + shifts:
                sx, sy = xx + dx, yy + dy
                if sx.min() < 0 or sy.min() < 0 or sx.max() >= W or sy.max() >= H or body[sy, sx].any():
                    continue
                out[yy, xx] = a[sy, sx]
                left[yy, xx] = False
                last = (dx, dy)
                break
    if left.any():  # whatever no clean tile could cover
        bgr = np.ascontiguousarray(out[:, :, ::-1])
        out = cv2.inpaint(bgr, left.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)[:, :, ::-1]
    return _img(a * (1 - al) + out.astype(np.float32) * al)


def paint_stump(img, cut, direction, width, rng):
    u = direction / (np.linalg.norm(direction) + 1e-6)
    ang = math.degrees(math.atan2(u[1], u[0]))
    pool = Image.new("L", (W, H), 0)
    pc = cut + u * width * 0.7
    ImageDraw.Draw(pool).polygon(_blob(pc, width * rng.uniform(0.8, 1.2), rng, jitter=0.5), fill=230)
    pool = pool.filter(ImageFilter.GaussianBlur(4))
    img = _blend(img, _solid((95, 5, 10)) + _noise(rng, 14), pool)
    sz = int(width * 2)
    face = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    fd = ImageDraw.Draw(face)
    cx = cy = width
    fd.ellipse([cx - width * 0.28, cy - width * 0.52, cx + width * 0.28, cy + width * 0.52], fill=(150, 20, 28, 255))
    fd.ellipse([cx - width * 0.10, cy - width * 0.14, cx + width * 0.10, cy + width * 0.14], fill=(232, 222, 205, 255))
    face = face.rotate(-ang, resample=Image.BICUBIC, expand=False)
    out = img.copy()
    out.paste(face, (int(cut[0] - cx), int(cut[1] - cy)), face)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse([cut[0] - width * 0.9, cut[1] - width * 0.9, cut[0] + width * 0.9, cut[1] + width * 0.9], fill=255)
    m = Image.composite(Image.new("L", (W, H), 255), m, pool)  # stump region + blood pool
    return out, m.filter(ImageFilter.GaussianBlur(6))


def paint_cover(img, mask_l, rgb, rng):
    return _blend(img, _solid(rgb) + _noise(rng, 26), mask_l)


BLANKET_RGB = {"olive drab wool blanket": (92, 96, 60), "grey wool blanket": (120, 120, 118),
               "silver emergency foil blanket": (190, 190, 196), "dark green poncho": (48, 66, 46),
               "brown tarp": (110, 84, 56)}


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
        r = width * rng.uniform(0.8, 1.15)
        img, m = paint_wound(img, c, r, kind, rng)
        mask = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5))
        prompt = f"{kind}, realistic moulage trauma makeup on a training manikin {part}, glossy fake blood, torn fabric, photo"
        img = inpaint(inp, img, mask, prompt, seed, strength=0.6)
        log[site] = dict(cls=cls, kind=kind, center=c.round(1).tolist())

    elif cls == "amputation":
        if rng.random() < 0.65:  # below elbow / knee
            cut = mid + (dist - mid) * rng.uniform(0.25, 0.5)
            removed, direction = [cut, dist], dist - mid
            level = "below the elbow" if upper else "below the knee"
        else:
            cut = prox + (mid - prox) * rng.uniform(0.5, 0.7)
            removed, direction = [cut, mid, dist], mid - prox
            level = "above the elbow" if upper else "above the knee"
        extend = s * (0.5 if upper else 0.65)  # hand / boot beyond the distal keypoint
        u = direction / (np.linalg.norm(direction) + 1e-6)
        start = cut + u * width * 0.25  # keep the stump itself
        erase = capsule_mask([start] + removed[1:], width * 2.4, extend_end=extend, blur=4)
        img = remove_region(img, erase, kps, width * 2, rng)
        img = inpaint(inp, img, erase, f"{ground}, empty ground, nothing there, photo", seed, strength=0.45)
        img, m = paint_stump(img, cut, direction, width, rng)
        prompt = f"bloody amputation stump {level}, exposed bone, torn flesh, moulage on training manikin, blood pool on {ground}, photo"
        img = inpaint(inp, img, m, prompt, seed + 1, strength=0.55)
        log[site] = dict(cls=cls, level=level, cut=cut.round(1).tolist())

    elif cls == "not_testable":
        extend = s * (0.4 if upper else 0.5)
        cover = capsule_mask([prox, mid, dist], width * 2.3, extend_end=extend, blur=8)
        blanket = rng.choice(BLANKETS)
        img = paint_cover(img, cover, BLANKET_RGB[blanket], rng)
        img = inpaint(inp, img, cover, f"crumpled {blanket} with folds lying on {ground}, photo", seed, strength=0.7)
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
    ap.add_argument("--prefix", default="syn_")
    ap.add_argument("--backend", choices=["sdxl", "flux"], default="sdxl")
    args = ap.parse_args(argv)
    global BACKEND
    BACKEND = args.backend

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

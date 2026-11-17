"""
Procedural "simulated casualty" image generator (non-HSR synthetic data).

Mimics what the D2 Part 2 set is described as: photos of medical manikins
with traumatic limb injuries, labelled per limb as
no_injury / wound / amputation / not_testable, with laterality defined by the
CASUALTY's anatomical side (not the camera's).

What it deliberately varies, because these are the things that make the real
task hard:
  * view: supine (face up → casualty's left on image right) vs prone
    (face down → casualty's left on image left), plus in-plane rotation up to
    ±180°, so laterality must be inferred from body cues (face vs back of head)
  * pose: limb angles, figure scale/position, partial off-centre framing
  * injuries: blood-soaked wounds, lacerations, burns, shrapnel peppering;
    amputations at random levels with ragged stumps and blood pools
  * hard negatives: blood pools / splatter near intact limbs, tourniquets
  * not_testable: limb fully hidden under a blanket, litter, kit bag
  * appearance: manikin skin tones, uniform colours, backgrounds (dirt,
    grass, concrete, sand, floor), lighting gradients, blur, noise, JPEG

Usage:
    python synthetic/generate.py --out data/synth/train --n 3000 --seed 1
    python synthetic/generate.py --out data/synth/test  --n 600  --seed 999

Writes images + labels.json in the exact ICD predictions.json format.
This is a proof-of-concept and pre-training source, NOT a stand-in for the
real DARPA photos.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2qual.schema import build_document, make_prediction  # noqa: E402

W, H = 640, 480
LW, LH = 440, 600                      # local figure canvas (head at top)

SKIN = [(236, 200, 170), (214, 170, 135), (180, 135, 100), (140, 100, 70), (225, 190, 160), (200, 160, 120)]
UNIFORM = [(96, 104, 72), (120, 110, 85), (85, 95, 80), (140, 125, 95), (70, 80, 70), (60, 70, 90), (110, 118, 100)]
BOOT = [(40, 35, 30), (90, 75, 55), (60, 50, 40)]
BACKGROUNDS = [(120, 95, 70), (95, 115, 60), (140, 140, 135), (190, 170, 130), (160, 150, 140), (80, 85, 70)]
BLANKETS = [(170, 170, 175), (95, 100, 70), (60, 70, 100), (150, 40, 40), (200, 190, 120)]
BLOOD = [(120, 10, 15), (150, 20, 25), (95, 5, 10), (170, 30, 30)]

P_CLASS = {"no_injury": 0.48, "wound": 0.27, "amputation": 0.12, "not_testable": 0.13}


def pick(rng, d):
    r, acc = rng.random(), 0.0
    for k, p in d.items():
        acc += p
        if r <= acc:
            return k
    return k


def jitter(c, rng, a=18):
    return tuple(max(0, min(255, v + rng.randint(-a, a))) for v in c)


# ── background ────────────────────────────────────────────────────────────────

def background(rng: random.Random, nrng: np.random.Generator) -> Image.Image:
    base = np.array(jitter(rng.choice(BACKGROUNDS), rng, 25), dtype=np.float32)
    small = nrng.normal(0, 1, (H // 16 + 1, W // 16 + 1, 1)).astype(np.float32)
    low = np.array(Image.fromarray(((small[..., 0] + 3) * 40).clip(0, 255).astype(np.uint8))
                   .resize((W, H), Image.BICUBIC), dtype=np.float32)[..., None] / 255.0 - 0.5
    fine = nrng.normal(0, 1, (H, W, 1)).astype(np.float32)
    img = base + low * 60 + fine * rng.uniform(4, 14)
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(5, 30)):  # stones / leaves / debris
        x, y, r = rng.randint(0, W), rng.randint(0, H), rng.randint(2, 10)
        d.ellipse([x - r, y - r, x + r, y + r], fill=jitter(tuple(int(v) for v in base), rng, 50))
    if rng.random() < 0.3:  # floor tiles / planks
        step = rng.randint(50, 120)
        for x in range(0, W, step):
            d.line([(x, 0), (x, H)], fill=jitter(tuple(int(v) for v in base * 0.7), rng, 10), width=2)
    return im


# ── figure ───────────────────────────────────────────────────────────────────

def _limb_points(start, angle_deg, l1, l2, bend_deg):
    a1 = math.radians(angle_deg)
    j = (start[0] + l1 * math.sin(a1), start[1] + l1 * math.cos(a1))
    a2 = math.radians(angle_deg + bend_deg)
    e = (j[0] + l2 * math.sin(a2), j[1] + l2 * math.cos(a2))
    return [start, j, e]


def _lerp(p, q, f):
    return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)


def _along(pts, f):
    """Point at fraction f along a 2-segment polyline + cumulative split."""
    l1 = math.dist(pts[0], pts[1])
    l2 = math.dist(pts[1], pts[2])
    s = f * (l1 + l2)
    if s <= l1:
        return _lerp(pts[0], pts[1], s / l1), 0
    return _lerp(pts[1], pts[2], (s - l1) / l2), 1


def _thick(d, pts, w, fill):
    d.line(pts, fill=fill, width=int(w), joint="curve")
    for p in pts:
        d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=fill)


def _blob(d, c, r, color, rng, n=10):
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        rr = r * rng.uniform(0.6, 1.25)
        pts.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
    d.polygon(pts, fill=color)


def draw_wound(d, pts, w, rng):
    f = rng.uniform(0.15, 0.9)
    c, _ = _along(pts, f)
    kind = rng.choice(["soak", "soak", "lac", "burn", "frag", "open_fx"])
    if kind == "soak":
        _blob(d, c, w * rng.uniform(0.45, 0.8), rng.choice(BLOOD), rng)
        _blob(d, c, w * rng.uniform(0.15, 0.3), (60, 0, 5), rng)
    elif kind == "lac":
        a = rng.uniform(0, math.pi)
        L = w * rng.uniform(0.6, 1.1)
        p, q = (c[0] - L * math.cos(a) / 2, c[1] - L * math.sin(a) / 2), (c[0] + L * math.cos(a) / 2, c[1] + L * math.sin(a) / 2)
        _blob(d, c, w * 0.35, rng.choice(BLOOD), rng)
        d.line([p, q], fill=(70, 0, 5), width=max(3, int(w * 0.15)))
    elif kind == "burn":
        _blob(d, c, w * rng.uniform(0.5, 0.8), (45, 30, 25), rng)
        _blob(d, c, w * 0.3, (210, 110, 110), rng)
    elif kind == "frag":
        for _ in range(rng.randint(4, 9)):
            o = (c[0] + rng.uniform(-w * 0.6, w * 0.6), c[1] + rng.uniform(-w * 0.6, w * 0.6))
            _blob(d, o, w * rng.uniform(0.08, 0.18), rng.choice(BLOOD), rng, 6)
    else:  # open fracture: blood + white bone fragment
        _blob(d, c, w * 0.6, rng.choice(BLOOD), rng)
        d.ellipse([c[0] - w * 0.15, c[1] - w * 0.1, c[0] + w * 0.15, c[1] + w * 0.1], fill=(235, 230, 215))
    if rng.random() < 0.4:
        draw_tourniquet(d, pts, w, max(0.05, f - 0.15), rng)


def draw_tourniquet(d, pts, w, f, rng):
    c, seg = _along(pts, f)
    a, b = pts[seg], pts[seg + 1]
    ang = math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2
    L = w * 0.65
    p, q = (c[0] - L * math.cos(ang), c[1] - L * math.sin(ang)), (c[0] + L * math.cos(ang), c[1] + L * math.sin(ang))
    d.line([p, q], fill=rng.choice([(20, 20, 20), (200, 120, 30), (190, 175, 140)]), width=max(5, int(w * 0.28)))


def draw_figure(rng: random.Random, labels_local: list[str], front: bool):
    """
    labels_local order: [img-left arm, img-right arm, img-left leg, img-right leg] in the
    UNROTATED local canvas (head at top). Returns RGBA figure + occluder list.
    """
    fig = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
    d = ImageDraw.Draw(fig)
    skin = jitter(rng.choice(SKIN), rng, 10)
    uni = jitter(rng.choice(UNIFORM), rng, 12)
    boot = rng.choice(BOOT)
    cx = LW // 2
    s = rng.uniform(0.9, 1.1)

    sh_l, sh_r = (cx - 60 * s, 150), (cx + 60 * s, 150)
    hip_l, hip_r = (cx - 33 * s, 345), (cx + 33 * s, 345)
    arm_w, leg_w = rng.uniform(26, 34) * s, rng.uniform(36, 46) * s
    limbs = [
        # angle 0 = straight down the body; negative swings toward image-left (outward for left limbs)
        _limb_points(sh_l, -rng.uniform(15, 75), 95 * s, 90 * s, rng.uniform(-30, 35)),
        _limb_points(sh_r, rng.uniform(15, 75), 95 * s, 90 * s, rng.uniform(-35, 30)),
        _limb_points(hip_l, -rng.uniform(0, 22), 115 * s, 115 * s, rng.uniform(-12, 12)),
        _limb_points(hip_r, rng.uniform(0, 22), 115 * s, 115 * s, rng.uniform(-12, 12)),
    ]
    widths = [arm_w, arm_w, leg_w, leg_w]

    # ground blood pools first (under the body)
    pools = []
    for i, lab in enumerate(labels_local):
        if lab == "amputation" or (lab == "wound" and rng.random() < 0.3) or (lab == "no_injury" and rng.random() < 0.08):
            pools.append(_along(limbs[i], rng.uniform(0.5, 1.0))[0])
    for p in pools:
        _blob(d, (p[0] + rng.uniform(-30, 30), p[1] + rng.uniform(-20, 30)), rng.uniform(25, 55),
              (*rng.choice(BLOOD), 230), rng, 14)

    # limbs
    for i, (pts, w, lab) in enumerate(zip(limbs, widths, labels_local)):
        is_leg = i >= 2
        if lab == "amputation":
            f = rng.uniform(0.25, 0.75)
            stump, seg = _along(pts, f)
            cut = pts[: seg + 1] + [stump]
            _thick(d, cut, w, uni)
            _blob(d, stump, w * 0.62, rng.choice(BLOOD), rng)
            d.ellipse([stump[0] - w * 0.16, stump[1] - w * 0.16, stump[0] + w * 0.16, stump[1] + w * 0.16],
                      fill=(230, 225, 205))
            if rng.random() < 0.7:
                draw_tourniquet(d, pts, w, max(0.05, f - 0.12), rng)
        else:
            _thick(d, pts, w, uni)
            end = pts[2]
            if is_leg:
                d.ellipse([end[0] - w * 0.6, end[1] - w * 0.35, end[0] + w * 0.6, end[1] + w * 0.75], fill=boot)
            else:
                d.ellipse([end[0] - w * 0.5, end[1] - w * 0.5, end[0] + w * 0.5, end[1] + w * 0.5], fill=skin)
            if lab == "wound":
                draw_wound(d, pts, w, rng)
                if rng.random() < 0.25:
                    draw_wound(d, pts, w, rng)
            elif lab == "no_injury" and rng.random() < 0.05:
                draw_tourniquet(d, pts, w, rng.uniform(0.1, 0.4), rng)  # rare TQ on intact limb

    # torso + head (drawn after limbs so shoulders/hips overlap nicely)
    d.rounded_rectangle([cx - 70 * s, 120, cx + 70 * s, 360], radius=30, fill=uni)
    d.rectangle([cx - 14, 95, cx + 14, 125], fill=skin)
    hx, hy, hr = cx, 62, 42 * s
    d.ellipse([hx - hr, hy - hr * 1.1, hx + hr, hy + hr * 1.1], fill=skin)
    if front:
        e = hr * 0.38
        for sx in (-1, 1):
            d.ellipse([hx + sx * e - 6, hy - 10, hx + sx * e + 6, hy - 1], fill=(40, 40, 50))
        d.polygon([(hx, hy - 2), (hx - 6, hy + 14), (hx + 6, hy + 14)], fill=jitter(skin, rng, 25))
        d.line([(hx - 14, hy + 25), (hx + 14, hy + 25)], fill=(140, 60, 60), width=4)
        d.line([(cx, 130), (cx, 350)], fill=jitter(uni, rng, 30), width=4)       # zipper
        for sx in (-1, 1):                                                       # chest pockets
            d.rectangle([cx + sx * 45 - 20, 160, cx + sx * 45 + 20, 195], outline=jitter(uni, rng, 35), width=3)
    else:
        hair = rng.choice([(40, 30, 20), (90, 70, 40), (20, 20, 20), (130, 100, 60)])
        d.ellipse([hx - hr, hy - hr * 1.1, hx + hr, hy + hr * 1.1], fill=hair)
        d.ellipse([hx - hr * 1.05, hy - 6, hx - hr * 0.8, hy + 14], fill=skin)    # ears
        d.ellipse([hx + hr * 0.8, hy - 6, hx + hr * 1.05, hy + 14], fill=skin)
        d.line([(cx - 60, 230), (cx + 60, 230)], fill=jitter(uni, rng, 30), width=5)  # belt line / yoke
        d.rectangle([cx - 40, 135, cx + 40, 150], fill=jitter(uni, rng, 30))

    occluders = []
    for i, lab in enumerate(labels_local):
        if lab == "not_testable":
            xs = [p[0] for p in limbs[i]]
            ys = [p[1] for p in limbs[i]]
            m = widths[i] * 0.9 + 12
            occluders.append([min(xs) - m, min(ys) - m * 0.6, max(xs) + m, max(ys) + m * 1.2])
    return fig, occluders


def draw_occluders(fig, occluders, rng):
    d = ImageDraw.Draw(fig)
    for box in occluders:
        x0, y0, x1, y1 = box
        col = jitter(rng.choice(BLANKETS), rng, 15)
        pts = [(x0 + rng.uniform(-8, 8), y0 + rng.uniform(-8, 8)), (x1 + rng.uniform(-8, 8), y0 + rng.uniform(-8, 8)),
               (x1 + rng.uniform(-8, 8), y1 + rng.uniform(-8, 8)), (x0 + rng.uniform(-8, 8), y1 + rng.uniform(-8, 8))]
        d.polygon(pts, fill=(*col, 255))
        for _ in range(4):  # folds
            yy = rng.uniform(y0, y1)
            d.line([(x0, yy), (x1, yy + rng.uniform(-15, 15))], fill=(*jitter(col, rng, 40), 255), width=3)


# ── compose ──────────────────────────────────────────────────────────────────

def make_sample(rng: random.Random, nrng: np.random.Generator):
    # Labels in ANATOMICAL order: [L arm, R arm, L leg, R leg] — matches schema.SITES.
    labels = [pick(rng, P_CLASS) for _ in range(4)]
    if rng.random() < 0.06:                       # lower body covered entirely
        labels[2] = labels[3] = "not_testable"
    front = rng.random() < 0.6                    # supine (face up) vs prone
    # Local canvas: image-left limbs. Supine → image-left is casualty RIGHT.
    if front:
        local = [labels[1], labels[0], labels[3], labels[2]]
    else:
        local = [labels[0], labels[1], labels[2], labels[3]]

    fig, occ = draw_figure(rng, local, front)
    draw_occluders(fig, occ, rng)
    angle = rng.choice([rng.uniform(-35, 35)] * 3 + [rng.uniform(-180, 180)])
    fig = fig.rotate(angle, resample=Image.BICUBIC, expand=True)
    scale = rng.uniform(0.7, 0.95) * H / max(fig.size)
    fig = fig.resize((max(1, int(fig.width * scale)), max(1, int(fig.height * scale))), Image.BICUBIC)

    bg = background(rng, nrng)
    x = int((W - fig.width) / 2 + rng.uniform(-0.12, 0.12) * W)
    y = int((H - fig.height) / 2 + rng.uniform(-0.08, 0.08) * H)
    shadow = Image.new("RGBA", fig.size, (0, 0, 0, 0))
    shadow.putalpha(fig.getchannel("A").point(lambda a: int(a * 0.35)))
    bg.paste(shadow.filter(ImageFilter.GaussianBlur(8)), (x + 8, y + 10), shadow.filter(ImageFilter.GaussianBlur(8)))
    bg.paste(fig, (x, y), fig)

    # distractors: medic hands/gear near casualty
    d = ImageDraw.Draw(bg)
    for _ in range(rng.randint(0, 2)):
        cx, cy = rng.randint(0, W), rng.choice([rng.randint(0, 60), rng.randint(H - 60, H)])
        if rng.random() < 0.5:
            _blob(d, (cx, cy), rng.uniform(18, 35), jitter(rng.choice(SKIN), rng, 10), rng)
        else:
            d.rounded_rectangle([cx - 40, cy - 25, cx + 40, cy + 25], radius=8, fill=jitter(rng.choice(UNIFORM), rng, 20))

    # photometric: lighting gradient, colour cast, blur, noise
    arr = np.asarray(bg, dtype=np.float32)
    gx = np.linspace(rng.uniform(0.75, 1.15), rng.uniform(0.75, 1.15), W)[None, :, None]
    gy = np.linspace(rng.uniform(0.8, 1.1), rng.uniform(0.8, 1.1), H)[:, None, None]
    arr = arr * gx * gy * np.array([rng.uniform(0.9, 1.1) for _ in range(3)])[None, None, :]
    arr += nrng.normal(0, rng.uniform(2, 9), arr.shape)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    return img, labels, {"front": front, "angle": round(angle, 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="image_")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng, nrng = random.Random(a.seed), np.random.default_rng(a.seed)
    preds, meta = [], {}
    for i in range(a.n):
        img, labels, m = make_sample(rng, nrng)
        name = f"{a.prefix}{i + 1:06d}." + ("png" if rng.random() < 0.15 else "jpg")
        if name.endswith("jpg"):
            img.save(out / name, quality=rng.randint(70, 95))
        else:
            img.save(out / name)
        preds.append(make_prediction(name, labels))
        meta[name] = m
    (out / "labels.json").write_text(json.dumps(build_document(preds, "synthetic-gt", "1", "gt@example.com"), indent=1))
    (out / "render_meta.json").write_text(json.dumps(meta))
    print(f"wrote {a.n} images -> {out}")


if __name__ == "__main__":
    main()

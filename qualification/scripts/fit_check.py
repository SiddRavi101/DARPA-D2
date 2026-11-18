"""
Will each candidate backbone fit DARPA's box? (ICD: 1 GPU < 16 GB, 1 hour total.)

For each candidate: checks the pretrained tag exists in the pinned timm, then measures
  * training peak GPU memory for a batch (does it train on our GPU?)
  * inference peak memory and images/sec in fp16, with mirror TTA, x N folds
  * projected container time for a test set of --test-images images
No pretrained weights are downloaded (timing does not depend on weight values).

    python scripts/fit_check.py --test-images 2000 --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2qual.model import InjuryNet, ModelConfig, mirror_inputs  # noqa: E402

CANDIDATES = [
    # (name, global size, limb crops)
    ("convnext_small.fb_in22k_ft_in1k_384", 512, False),
    ("convnext_small.fb_in22k_ft_in1k_384", 512, True),
    ("convnext_base.fb_in22k_ft_in1k_384", 512, False),
    ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", 448, False),
    ("vit_base_patch14_dinov2.lvd142m", 518, False),
    ("vit_base_patch14_dinov2.lvd142m", 518, True),
    ("vit_large_patch14_dinov2.lvd142m", 518, False),
    ("vit_base_patch16_siglip_384.webli", 384, False),
]


def measure(name, size, limb, train_batch, infer_batch, device):
    import timm

    tag_ok = name in timm.list_pretrained(name.split(".")[0] + "*")
    cfg = ModelConfig(backbone=name, image_size=size, limb_crops=limb, crop_size=224)
    model = InjuryNet(cfg, pretrained=False).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    def batch(b):
        x = torch.randn(b, 3, size, size, device=device)
        if not limb:
            return x, None, None
        return x, torch.randn(b, 4, 3, 224, 224, device=device), torch.rand(b, 4, 4, device=device)

    # training step (bf16 autocast like train.py)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        with torch.autocast("cuda"):
            loss = model(*batch(train_batch)).float().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    train_gb = torch.cuda.max_memory_allocated() / 1e9
    del opt
    torch.cuda.empty_cache()

    # inference: fp16, mirror TTA, like predict.py
    model.eval().half()
    torch.cuda.reset_peak_memory_stats()
    inputs = [t.half() if t is not None else None for t in batch(infer_batch)]
    with torch.inference_mode():
        for _ in range(2):
            model(*inputs)
        torch.cuda.synchronize()
        t0 = time.time()
        n_iter = 5
        for _ in range(n_iter):
            model(*inputs)
            model(*mirror_inputs(*inputs))
        torch.cuda.synchronize()
    ips = infer_batch * n_iter / (time.time() - t0)
    infer_gb = torch.cuda.max_memory_allocated() / 1e9
    del model
    torch.cuda.empty_cache()
    return dict(backbone=name, size=size, limb_crops=limb, pretrained_tag_found=tag_ok, params_m=round(params, 1),
                train_peak_gb=round(train_gb, 1), train_batch=train_batch,
                infer_peak_gb=round(infer_gb, 1), infer_img_per_s_per_model=round(ips, 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-images", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    rows = []
    for name, size, limb in CANDIDATES:
        try:
            r = measure(name, size, limb, a.train_batch, a.infer_batch, dev)
            # projected container time: N images x folds members; pose model adds ~0.15 s/img for limb models
            per_img = a.folds / r["infer_img_per_s_per_model"] + (0.15 if limb else 0)
            r["projected_minutes"] = round(a.test_images * per_img / 60, 1)
            r["fits_16gb"] = r["infer_peak_gb"] < 14
        except Exception as e:  # e.g. out of memory: record it and free the GPU for the next one
            import gc

            r = dict(backbone=name, size=size, limb_crops=limb, error=str(e)[:200])
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

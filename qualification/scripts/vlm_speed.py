"""
GPU check for the VLM fast path (d2qual/vlm_ft.py):
  1. equivalence: fast path vs the original slow path on the same images (base model)
  2. inference speed of a 3-adapter VLMEnsemble with/without mirror TTA, and with 1 adapter
  3. one LoRA training step (forward+backward) at batch 4

    python scripts/vlm_speed.py --n 300 --out /root/work/out/vlm_speed.json
Adapters here are freshly initialised (speed does not depend on the trained values).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fake_images(n, seed=0):
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w, h = [(640, 480), (480, 640), (1024, 768), (800, 800)][i % 4]
        im = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        d = ImageDraw.Draw(im)
        for _ in range(6):
            x, y = rng.integers(0, w - 50), rng.integers(0, h - 50)
            d.rectangle([x, y, x + rng.integers(30, 200), y + rng.integers(30, 200)], fill=tuple(rng.integers(0, 255, 3).tolist()))
        out.append(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="vlm_speed.json")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model

    from d2qual.vlm_ft import META, VLMClassifier, VLMEnsemble, lora_targets

    dev = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(0)}

    # 1. equivalence on the base model (Qwen fast path only; other bases already use the plain forward)
    clf = VLMClassifier(a.base, dev, train=True)  # train=True: no CPU offload while adapters are made
    res["base_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    ims = fake_images(4, seed=1)
    with torch.inference_mode():
        res["probs_base"] = clf.letter_logits(ims).softmax(-1)[0].tolist()
    if clf.qwen:
      with torch.inference_mode():
        fast = clf.letter_logits(ims).softmax(-1)
        ref = clf.letter_logits_reference(ims).softmax(-1)
      res["max_abs_prob_diff_fast_vs_reference"] = float((fast - ref).abs().max())
      res["argmax_agreement"] = float((fast.argmax(-1) == ref.argmax(-1)).float().mean())
    print("equivalence", res, flush=True)

    # 2. adapters
    tmp = Path(tempfile.mkdtemp())
    dirs = []
    pm = get_peft_model(clf.model, LoraConfig(r=16, lora_alpha=32, target_modules=lora_targets(clf.model), init_lora_weights="gaussian"))
    for k in range(3):
        d = tmp / f"vlm_fold{k}"
        pm.save_pretrained(d)
        (d / META).write_text(json.dumps({"format": "d2qual-vlm-v1", "base": a.base, "image_size": 448,
                                          "fallback": ["no_injury"] * 4}))
        dirs.append(d)
    del pm, clf
    torch.cuda.empty_cache()

    images = fake_images(a.n, seed=2)

    def bench(member_dirs, tta, label):
        os.environ["D2_VLM_TTA"] = "1" if tta else "0"
        ens = VLMEnsemble(member_dirs, dev, None)
        ens.member_probs(images[:2])  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for s in range(0, len(images), a.batch):
            ens.member_probs(images[s:s + a.batch])
        torch.cuda.synchronize()
        dt = time.time() - t0
        r = {"seconds": round(dt, 1), "s_per_image": round(dt / len(images), 3),
             "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "projected_min_per_2000": round(dt / len(images) * 2000 / 60, 1)}
        print(label, r, flush=True)
        res[label] = r
        del ens
        torch.cuda.empty_cache()

    bench(dirs, True, "3_adapters_tta")
    bench(dirs, False, "3_adapters_no_tta")
    bench(dirs[:1], True, "1_adapter_tta")

    # 3. one training step (forward+backward), batch 4
    clf = VLMClassifier(a.base, dev, train=True)
    m = clf.model
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    clf.model = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=lora_targets(clf.model)))
    y = torch.zeros(16, dtype=torch.long, device=dev)
    times = []
    for i in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = clf.letter_logits(images[i * 4:(i + 1) * 4]).view(-1, 4)
        torch.nn.functional.cross_entropy(lg.float(), y).backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    res["train_step_batch4_s"] = round(float(np.median(times[1:])), 3)
    res["train_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print("train", res["train_step_batch4_s"], res["train_peak_gb"], flush=True)

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("=== VLM_SPEED_DONE", flush=True)


if __name__ == "__main__":
    main()

"""
Fine-tuned small vision-language model as an ensemble candidate (the PRONTO-style approach:
their best limb-injury model was a compact VLM asked short per-limb questions).

Model: Qwen2-VL-2B-Instruct (Apache-2.0, ungated), LoRA-fine-tuned.
For every image we ask four questions, one per anatomical limb:

    "...Look at the casualty's LEFT ARM (the casualty's own left, not the viewer's left).
     A) intact  B) wound ...  C) amputated  D) not visible.  Answer with one letter."

and read the probabilities of the tokens A/B/C/D at the answer position: one forward
pass, no text generation, so it is fast and always returns a valid class. Training is
plain cross-entropy over those four letters. Mirror TTA works as for the CNNs: mirror
the photo and ask about the opposite side.

Images are letterboxed to a fixed square so every prompt has the same length (no padding).

Training (called by d2qual.train when --backbone vlm:<hf-id>):
    python -m d2qual.train ... --backbone vlm:Qwen/Qwen2-VL-2B-Instruct --epochs 3 --batch 4 --lr 1e-4
Container: the base model must be baked into model/vlm_base/ (scripts/submit.sh does it),
adapters live in model/weights/<name>/ next to vlm_meta.json.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np

from .schema import CLASSES, FLIP_PERMUTATION

log = logging.getLogger("d2qual")

DEFAULT_BASE = "Qwen/Qwen2-VL-2B-Instruct"
META = "vlm_meta.json"
FORMAT = "d2qual-vlm-v1"
LETTERS = ("A", "B", "C", "D")  # same order as CLASSES
SITE_TEXT = ("LEFT ARM", "RIGHT ARM", "LEFT LEG", "RIGHT LEG")  # schema.SITES order
QUESTION = (
    "This is a photo of a casualty (often a medical training manikin with simulated injuries). "
    "Look only at the casualty's {site} (the casualty's own {side}, not the viewer's {side}). Which is true?\n"
    "A) intact, no visible injury\n"
    "B) wound: laceration, burn, shrapnel or open fracture, limb still attached\n"
    "C) amputated: the limb is severed\n"
    "D) not visible in the photo\n"
    "Answer with one letter."
)
LORA_TARGETS = r".*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"  # LLM only, not the vision tower


def _question(site: int) -> str:
    text = SITE_TEXT[site]
    return QUESTION.format(site=text, side=text.split()[0].lower())


class VLMClassifier:
    def __init__(self, base: str, device, image_size: int = 448, dtype=None, train: bool = False):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.image_size = image_size
        self.dtype = dtype or (torch.bfloat16 if getattr(device, "type", "") == "cuda" else torch.float32)
        px = image_size * image_size
        self.processor = AutoProcessor.from_pretrained(base, min_pixels=px, max_pixels=px)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=self.dtype, attn_implementation="sdpa").to(device)
        tok = self.processor.tokenizer
        ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
        assert all(len(i) == 1 for i in ids), f"answer letters must be single tokens: {ids}"
        self.letter_ids = [i[0] for i in ids]
        self.prompts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _question(s)}]}],
            tokenize=False, add_generation_prompt=True) for s in range(4)]
        if not train:
            self.model.eval()

    # ── core ────────────────────────────────────────────────────────────────────
    def _inputs(self, images):
        from .data import letterbox

        sq = [letterbox(im, self.image_size) for im in images]
        texts = [p for _ in sq for p in self.prompts]          # image-major: 4 questions per image
        ims = [im for im in sq for _ in range(4)]
        enc = self.processor(text=texts, images=ims, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def letter_logits(self, images):
        """-> tensor (N, 4 sites, 4 classes) of logits at the answer position."""
        enc = self._inputs(images)
        try:
            out = self.model(**enc, logits_to_keep=1, use_cache=False)
        except TypeError:  # older transformers without logits_to_keep
            out = self.model(**enc, use_cache=False)
        logits = out.logits[:, -1, :][:, self.letter_ids].float()
        return logits.view(len(images), 4, len(LETTERS))

    def predict_probs(self, images, tta: bool = True):
        from PIL import ImageOps

        torch = self.torch
        with torch.inference_mode():
            p = self.letter_logits(images).softmax(-1)
            if tta:
                pm = self.letter_logits([ImageOps.mirror(im) for im in images]).softmax(-1)
                p = p + pm[:, list(FLIP_PERMUTATION)]
                p = p / 2
        return p


# ── training (one fold) ─────────────────────────────────────────────────────────

def train_vlm(args, train_ids, val_ids, labels, out_dir: Path, device, fallback) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model

    from .data import CLASS_TO_IDX, _augment, flip_labels, load_image

    base = args.backbone.split(":", 1)[1] or DEFAULT_BASE
    clf = VLMClassifier(base, device, image_size=args.vlm_size, train=True)
    model = clf.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32, base stays bf16
    clf.model = model
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"vlm {base}: {n_train / 1e6:.1f}M trainable LoRA params", flush=True)

    rng = random.Random(args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(train_ids) / args.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    cache: dict = {}

    def get(iid):  # decode once, keep a downsized copy (VLM sees 448 px anyway)
        if iid not in cache:
            im = load_image(Path(args.images) / iid)
            s = min(1.0, 2 * args.vlm_size / max(im.size))
            cache[iid] = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s)))) if s < 1 else im
        return cache[iid]

    def evaluate():
        model.eval()
        probs = []
        for k in range(0, len(val_ids), args.batch):
            probs.append(clf.predict_probs([get(i) for i in val_ids[k:k + args.batch]]).cpu())
        model.train()
        probs = torch.cat(probs) if probs else torch.zeros(0, 4, 4)
        y = torch.tensor([[CLASS_TO_IDX[c] for c in labels[i]] for i in val_ids])
        pred = probs.argmax(-1)
        conf = np.zeros((4, 4), int)
        for t, p in zip(y.flatten().tolist(), pred.flatten().tolist()):
            conf[t, p] += 1
        acc = (pred == y).float().mean().item() if len(val_ids) else None
        return acc, probs.numpy(), conf

    best = {"overall_accuracy": -1.0}
    model.train()
    for epoch in range(args.epochs):
        order = list(train_ids)
        rng.shuffle(order)
        t0, running, n = time.time(), 0.0, 0
        for k in range(0, len(order), args.batch):
            ims, ys = [], []
            for iid in order[k:k + args.batch]:
                im, lab = _augment(get(iid), rng), list(labels[iid])
                if rng.random() < 0.5:
                    from PIL import ImageOps
                    im, lab = ImageOps.mirror(im), flip_labels(lab)
                ims.append(im)
                ys.append([CLASS_TO_IDX[c] for c in lab])
            y = torch.tensor(ys, device=device).view(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = clf.letter_logits(ims).view(-1, len(LETTERS))
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n += 1
        msg = f"epoch {epoch + 1}/{args.epochs} loss {running / max(1, n):.4f} ({time.time() - t0:.0f}s)"
        if val_ids:
            acc, probs, conf = evaluate()
            msg += f" val_acc {acc:.4f}"
            if acc > best["overall_accuracy"]:
                best = {"overall_accuracy": acc, "epoch": epoch + 1,
                        "confusion": {"labels": list(CLASSES), "matrix": conf.tolist()},
                        "oof": {iid: pr.tolist() for iid, pr in zip(val_ids, probs)}}
                _save(model, out_dir, base, args, fallback, acc)
                msg += " *"
        print(msg, flush=True)
    if not val_ids:
        _save(model, out_dir, base, args, fallback, None)
        best = {"overall_accuracy": None, "epoch": args.epochs}
    return best


def _save(model, out_dir: Path, base: str, args, fallback, acc):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)  # adapter only (~tens of MB)
    (out_dir / META).write_text(json.dumps({
        "format": FORMAT, "base": base, "image_size": args.vlm_size, "classes": list(CLASSES),
        "fallback": list(fallback), "metrics": {"overall_accuracy": acc}}, indent=1))


# ── inference: all VLM members share one base model ─────────────────────────────

class VLMEnsemble:
    """Loads the base once and switches LoRA adapters per member."""

    def __init__(self, member_dirs: list[Path], device, vlm_base_dir: Path | None):
        from peft import PeftModel

        metas = [json.loads((d / META).read_text()) for d in member_dirs]
        base = str(vlm_base_dir) if vlm_base_dir and (vlm_base_dir / "config.json").exists() else metas[0]["base"]
        self.clf = VLMClassifier(base, device, image_size=metas[0].get("image_size", 448))
        self.names = []
        pm = None
        for d in member_dirs:
            name = d.name.replace(".", "_")
            if pm is None:
                pm = PeftModel.from_pretrained(self.clf.model, str(d), adapter_name=name)
            else:
                pm.load_adapter(str(d), adapter_name=name)
            self.names.append(name)
        self.clf.model = pm.eval()
        self.fallback = metas[0].get("fallback")
        log.info("vlm members %s on base %s", self.names, base)

    def member_probs(self, images):
        """-> list of (name, probs tensor (N,4,4)) one per adapter."""
        out = []
        for name in self.names:
            self.clf.model.set_adapter(name)
            out.append((name, self.clf.predict_probs(images)))
        return out

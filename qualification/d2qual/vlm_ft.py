"""
Fine-tuned small vision-language model as an ensemble candidate (the PRONTO-style approach:
their best limb-injury model was a compact VLM asked short per-limb questions).

Models: Qwen2-VL-2B-Instruct or Qwen3-VL-2B/4B-Instruct (all Apache-2.0, ungated), LoRA-fine-tuned.
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
adapters live in model/weights/<name>/ next to vlm_meta.json. Each distinct base is baked into
model/vlm_base/<hub id with / -> __>/ (scripts/submit.sh does it).
"""

from __future__ import annotations

import json
import logging
import math
import os
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
        from transformers import AutoProcessor

        try:  # transformers >= 4.57: one class for Qwen2-VL, Qwen2.5-VL and Qwen3-VL
            from transformers import AutoModelForImageTextToText as ModelCls
        except ImportError:  # older pins
            from transformers import Qwen2VLForConditionalGeneration as ModelCls

        self.torch = torch
        self.device = device
        self.image_size = image_size
        self.dtype = dtype or (torch.bfloat16 if getattr(device, "type", "") == "cuda" else torch.float32)
        px = image_size * image_size
        self.processor = AutoProcessor.from_pretrained(base, min_pixels=px, max_pixels=px)
        self.processor.tokenizer.padding_side = "left"
        self.model = ModelCls.from_pretrained(
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
    # Fast path (2026-11-19). The naive `model(**processor(text, images))` call was ~10x too slow:
    #  * transformers 4.51 Qwen2-VL "sdpa" vision attention builds ONE (P x P) mask over the patches
    #    of every image in the batch, so cost grew with the square of the batch size;
    #  * each photo went through the vision tower 4x (once per limb question) and again per adapter;
    #  * qwen2_vl has no `logits_to_keep` in 4.51, so full-vocabulary logits were computed for every
    #    token (~6 GB per batch).
    # Now: vision tower once per photo (one photo per call), image embeddings reused for the 4
    # questions and for every adapter, and only the 4 answer-letter logits at the last position.
    def _core(self):
        m = self.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def _parts(self):
        """(causal-LM wrapper, vision tower, text decoder, rope fn) across transformers layouts."""
        core = self._core()
        inner = core.model
        if hasattr(inner, "language_model") and hasattr(inner, "visual"):  # transformers >= 4.52
            return core, inner.visual, inner.language_model, inner.get_rope_index
        return core, core.visual, inner, core.get_rope_index  # transformers 4.51 Qwen2-VL

    def _text_template(self):
        """Tokenized 4 prompts with the image placeholder already expanded (fixed image size)."""
        if not hasattr(self, "_tmpl"):
            from PIL import Image

            dummy = Image.new("RGB", (self.image_size, self.image_size))
            enc = self.processor(text=list(self.prompts), images=[dummy] * 4, padding=True, return_tensors="pt")
            self._tmpl = (enc["input_ids"], enc["attention_mask"], enc["image_grid_thw"][:1])
        return self._tmpl

    def encode(self, images):
        """Vision tower once per photo -> dict reusable across questions, adapters (no LoRA in vision)."""
        from .data import letterbox

        torch = self.torch
        _, visual, _, _ = self._parts()
        sq = [letterbox(im, self.image_size) for im in images]
        pix = self.processor.image_processor(images=sq, return_tensors="pt")
        thw = pix["image_grid_thw"].to(self.device)
        pv = pix["pixel_values"].to(self.device, next(visual.parameters()).dtype)
        per = int(thw[0].prod())
        embs, deep = [], []
        with torch.no_grad():  # vision tower is frozen (LoRA only targets the LLM)
            for i in range(len(sq)):
                out = visual(pv[i * per:(i + 1) * per], grid_thw=thw[i:i + 1])
                if isinstance(out, tuple):  # Qwen3-VL: (embeds, deepstack features per injection layer)
                    out, ds = out
                    deep.append(ds)
                embs.append(out)
        emb = torch.stack(embs)                                    # (N, tokens, d)
        enc = {"n": len(sq), "image_embeds": emb, "thw": thw}
        if deep:
            enc["deepstack"] = [torch.stack([d[k] for d in deep]) for k in range(len(deep[0]))]
        return enc

    def logits_from(self, enc):
        """-> tensor (N, 4 sites, 4 classes) of answer-letter logits."""
        torch = self.torch
        core, _, lm, rope = self._parts()
        ids, mask, thw1 = self._text_template()
        n = enc["n"]
        input_ids = ids.repeat(n, 1).to(self.device)            # image-major: 4 questions per image
        attn = mask.repeat(n, 1).to(self.device)
        thw = thw1.to(self.device).repeat(4 * n, 1)
        tok = lm.embed_tokens(input_ids)
        d = tok.shape[-1]
        img = enc["image_embeds"].to(tok.dtype).repeat_interleave(4, dim=0).reshape(-1, d)
        is_img = input_ids == core.config.image_token_id
        tok = tok.masked_scatter(is_img.unsqueeze(-1).expand_as(tok), img)
        pos, _ = rope(input_ids, thw, None, attn)
        extra = {}
        if "deepstack" in enc:  # Qwen3-VL adds multi-level vision features inside early decoder layers
            extra = {"visual_pos_masks": is_img,
                     "deepstack_visual_embeds": [x.to(tok.dtype).repeat_interleave(4, dim=0).reshape(-1, d)
                                                 for x in enc["deepstack"]]}
        cache_pos = torch.arange(input_ids.shape[1], device=self.device)
        hidden = lm(input_ids=None, attention_mask=attn, position_ids=pos, inputs_embeds=tok, use_cache=False,
                    cache_position=cache_pos, **extra).last_hidden_state[:, -1]
        w = core.lm_head.weight[self.letter_ids]
        logits = (hidden.to(w.dtype) @ w.T).float()
        return logits.view(n, 4, len(LETTERS))

    def letter_logits(self, images):
        return self.logits_from(self.encode(images))

    def letter_logits_reference(self, images):
        """Original slow path, kept only to check the fast path gives the same numbers."""
        from .data import letterbox

        sq = [letterbox(im, self.image_size) for im in images]
        texts = [p for _ in sq for p in self.prompts]
        ims = [im for im in sq for _ in range(4)]
        enc = {k: v.to(self.device) for k, v in self.processor(text=texts, images=ims, padding=True,
                                                               return_tensors="pt").items()}
        out = self.model(**enc, use_cache=False)
        return out.logits[:, -1, :][:, self.letter_ids].float().view(len(images), 4, len(LETTERS))

    def encode_views(self, images, tta: bool = True):
        from PIL import ImageOps

        views = [self.encode(images)]
        if tta:
            views.append(self.encode([ImageOps.mirror(im) for im in images]))
        return views

    def probs_from(self, views):
        p = self.logits_from(views[0]).softmax(-1)
        if len(views) > 1:
            pm = self.logits_from(views[1]).softmax(-1)
            p = (p + pm[:, list(FLIP_PERMUTATION)]) / 2
        return p

    def predict_probs(self, images, tta: bool = True):
        with self.torch.inference_mode():
            return self.probs_from(self.encode_views(images, tta))


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
        select = getattr(args, "select", "last")
        last = epoch + 1 == args.epochs
        if val_ids and (last or select == "best" or getattr(args, "val_curve", False)):
            acc, probs, conf = evaluate()
            msg += f" val_acc {acc:.4f}"
            if (last if select == "last" else acc > best["overall_accuracy"]):
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

def base_dir_name(base: str) -> str:
    """Folder name under model/vlm_base/ for a hub id, e.g. Qwen/Qwen3-VL-2B-Instruct -> Qwen__Qwen3-VL-2B-Instruct."""
    return base.strip("/").replace("/", "__")


def resolve_base(base: str, vlm_base_dir: Path | None) -> str:
    """Baked-in local copy if present (container has no network), else the hub id (dev boxes)."""
    if vlm_base_dir:
        sub = vlm_base_dir / base_dir_name(base)
        if (sub / "config.json").exists():
            return str(sub)
        if (vlm_base_dir / "config.json").exists():  # legacy single-base layout
            return str(vlm_base_dir)
    return base


class _BaseGroup:
    """One base model with all its LoRA adapters; vision features shared across adapters."""

    def __init__(self, base: str, member_dirs: list[Path], device, vlm_base_dir: Path | None, image_size: int):
        from peft import PeftModel

        path = resolve_base(base, vlm_base_dir)
        self.clf = VLMClassifier(path, device, image_size=image_size)
        self.names = []
        pm = None
        for d in member_dirs:
            name = d.name.replace(".", "_")
            if pm is None:
                pm = PeftModel.from_pretrained(self.clf.model, str(d), adapter_name=name)
            else:
                pm.load_adapter(str(d), adapter_name=name)
            self.names.append((name, d.name))
        self.clf.model = pm.eval()
        log.info("vlm members %s on base %s", [n for _, n in self.names], path)


class VLMEnsemble:
    """VLM members grouped by base model (e.g. Qwen2-VL-2B and Qwen3-VL-2B adapters side by side)."""

    def __init__(self, member_dirs: list[Path], device, vlm_base_dir: Path | None):
        metas = {d: json.loads((d / META).read_text()) for d in member_dirs}
        groups: dict[str, list[Path]] = {}
        for d in member_dirs:
            groups.setdefault(metas[d]["base"], []).append(d)
        self.groups = [_BaseGroup(b, ds, device, vlm_base_dir, metas[ds[0]].get("image_size", 448))
                       for b, ds in groups.items()]
        self.fallback = metas[member_dirs[0]].get("fallback")
        self.tta = os.environ.get("D2_VLM_TTA", "1") != "0"

    def member_probs(self, images):
        """-> list of (member dir name, probs tensor (N,4,4)) one per adapter."""
        out = []
        for g in self.groups:
            with g.clf.torch.inference_mode():
                views = g.clf.encode_views(images, self.tta)  # vision once, shared by every adapter
                for adapter, dname in g.names:
                    g.clf.model.set_adapter(adapter)
                    out.append((dname, g.clf.probs_from(views)))
        return out

"""
Fine-tuned small vision-language model as an ensemble candidate (the PRONTO-style approach:
their best limb-injury model was a compact VLM asked short per-limb questions).

Models: Qwen2-VL-2B-Instruct, Qwen3-VL-2B/4B-Instruct or Google Gemma 4 E4B-it (all Apache-2.0,
ungated), LoRA-fine-tuned. Qwen uses a hand-written fast path; other bases use a plain forward.
Gemma 4 needs requirements-vlm-gemma.txt (transformers 5.x), not requirements-vlm.txt.
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


def torch_cat(xs, dim):
    import torch

    return torch.cat(xs, dim)

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
CROP_INTRO = ("Second image: an automatic close-up of the casualty's {site}, cut out by a pose detector "
              "(it can be off-target, or plain grey when the limb was not found).")
LORA_TARGETS = r".*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"  # LLM only, not the vision tower
# Non-Qwen bases (Gemma 4) name their vision/audio encoder blocks `layers` too: restrict to the text decoder.
LORA_TARGETS_GENERIC = r".*language_model.*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def is_qwen(model_or_config) -> bool:
    cfg = getattr(model_or_config, "config", model_or_config)
    return str(getattr(cfg, "model_type", "")).startswith("qwen")


def lora_targets(model) -> str:
    return LORA_TARGETS if is_qwen(model) else LORA_TARGETS_GENERIC


def _offload_lookup_tables(model, device) -> int:
    """Move the model to `device` but keep Gemma's per-layer embedding (PLE) tables in CPU RAM.

    Gemma 4 E4B is 8B params with embeddings (16 GB bf16) but 4.5B "effective": ~2.8B of it is
    PLE lookup tables, read a few rows per token. ICD: the eval GPU has < 16 GB, so the tables stay
    on CPU and only the looked-up rows travel to the GPU. Returns the number of params kept on CPU."""
    import torch

    found = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.Embedding) and "per_layer" in n.split(".")[-1]]
    parents = []
    for name, mod in found:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        setattr(parent, attr, torch.nn.Identity())  # detach so .to(device) never puts it on the GPU
        parents.append((parent, attr, mod))
    model.to(device)
    for parent, attr, mod in parents:
        mod.to("cpu")
        mod.register_forward_pre_hook(lambda m, args: tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args))
        mod.register_forward_hook(lambda m, args, out: out.to(device))
        setattr(parent, attr, mod)
    return sum(m.weight.numel() for _, _, m in parents)


def _question(site: int) -> str:
    text = SITE_TEXT[site]
    return QUESTION.format(site=text, side=text.split()[0].lower())


def _content(site: int, crops: bool) -> list:
    """Chat content for one limb question: whole photo [+ that limb's close-up] + question."""
    if not crops:
        return [{"type": "image"}, {"type": "text", "text": _question(site)}]
    return [{"type": "image"}, {"type": "image"},
            {"type": "text", "text": CROP_INTRO.format(site=SITE_TEXT[site]) + "\n" + _question(site)}]


def vlm_crops(img, kps, size: int, rng=None) -> list:
    """4 limb close-ups for the VLM, in the photo's own orientation (the question names the side)."""
    from .data import limb_crops

    return limb_crops(img, kps, size, rng, canonical=False)[0]


def mirror_view_crops(crops: list) -> list:
    """Crops for the mirrored photo: the mirrored casualty's LEFT arm is the original right arm, mirrored."""
    from PIL import ImageOps

    return [[ImageOps.mirror(four[i]) for i in FLIP_PERMUTATION] for four in crops]


class VLMClassifier:
    def __init__(self, base: str, device, image_size: int = 448, dtype=None, train: bool = False):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        self.torch = torch
        self.device = device
        self.image_size = image_size
        self.dtype = dtype or (torch.bfloat16 if getattr(device, "type", "") == "cuda" else torch.float32)
        self.qwen = is_qwen(AutoConfig.from_pretrained(base))
        px = image_size * image_size
        # Qwen: fixed pixel budget so every prompt has the same token count (fast path below).
        # Others (Gemma 4): the processor's own resize; images are still letterboxed to a square.
        self.processor = (AutoProcessor.from_pretrained(base, min_pixels=px, max_pixels=px) if self.qwen
                          else AutoProcessor.from_pretrained(base))
        self.processor.tokenizer.padding_side = "left"
        dkw = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": self.dtype}
        model = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM",
                         "Qwen2VLForConditionalGeneration"):  # 4.57+: one class for Qwen2/2.5/3-VL; Gemma 4: MultimodalLM
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(base, attn_implementation="sdpa", **dkw)
                break
            except ValueError:  # this auto class has no mapping for the architecture
                continue
        assert model is not None, f"no transformers class loads {base}"
        # Inference on the < 16 GB eval GPU: Gemma's PLE tables stay in CPU RAM (D2_VLM_OFFLOAD=0 to disable)
        if not self.qwen and not train and getattr(device, "type", "") == "cuda" and os.environ.get("D2_VLM_OFFLOAD", "1") != "0":
            n_cpu = _offload_lookup_tables(model, device)
            log.info("vlm %s: %.2fB lookup-table params kept on CPU", base, n_cpu / 1e9)
        else:
            model.to(device)
        self.model = model
        tok = self.processor.tokenizer
        ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
        assert all(len(i) == 1 for i in ids), f"answer letters must be single tokens: {ids}"
        self.letter_ids = [i[0] for i in ids]
        tkw = {} if self.qwen else {"enable_thinking": False}  # Gemma 4: answer directly, no thinking block
        self.prompts = {c: [self.processor.apply_chat_template(
            [{"role": "user", "content": _content(s, c)}], tokenize=False, add_generation_prompt=True, **tkw)
            for s in range(4)] for c in (False, True)}
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

    def _text_template(self, crops: bool = False):
        """Tokenized 4 prompts with the image placeholders already expanded (fixed image size)."""
        if not hasattr(self, "_tmpl"):
            self._tmpl = {}
        if crops not in self._tmpl:
            from PIL import Image

            dummy = Image.new("RGB", (self.image_size, self.image_size))
            per = 2 if crops else 1
            enc = self.processor(text=list(self.prompts[crops]), images=[dummy] * (4 * per), padding=True,
                                 return_tensors="pt")
            self._tmpl[crops] = (enc["input_ids"], enc["attention_mask"], enc["image_grid_thw"][:per])
        return self._tmpl[crops]

    def _vision(self, ims):
        """Square PIL images -> (embeds (N, T, d), deepstack list of (N, T, d) or None), one image per call."""
        torch = self.torch
        _, visual, _, _ = self._parts()
        pix = self.processor.image_processor(images=ims, return_tensors="pt")
        thw = pix["image_grid_thw"].to(self.device)
        pv = pix["pixel_values"].to(self.device, next(visual.parameters()).dtype)
        per = int(thw[0].prod())
        embs, deep = [], []
        with torch.no_grad():  # vision tower is frozen (LoRA only targets the LLM)
            for i in range(len(ims)):
                out = visual(pv[i * per:(i + 1) * per], grid_thw=thw[i:i + 1])
                if isinstance(out, tuple):  # Qwen3-VL: (embeds, deepstack features per injection layer)
                    out, ds = out
                    deep.append(ds)
                embs.append(out)
        ds = [torch.stack([d[k] for d in deep]) for k in range(len(deep[0]))] if deep else None
        return torch.stack(embs), ds

    def encode(self, images, crops=None):
        """Vision tower once per photo (and per limb close-up) -> dict reusable across questions and
        adapters (no LoRA in vision). crops: optional list (per photo) of 4 PIL close-ups."""
        from .data import letterbox

        sq = [letterbox(im, self.image_size) for im in images]
        if not self.qwen:  # generic path: images go through the full model in logits_from
            enc = {"n": len(sq), "images": sq}
            if crops is not None:
                enc["crops"] = [[letterbox(c, self.image_size) for c in four] for four in crops]
            return enc
        emb, ds = self._vision(sq)
        enc = {"n": len(sq), "image_embeds": emb}
        if ds is not None:
            enc["deepstack"] = ds
        if crops is not None:
            flat = [letterbox(c, self.image_size) for four in crops for c in four]
            cemb, cds = self._vision(flat)
            enc["crop_embeds"] = cemb.view(len(sq), 4, *cemb.shape[1:])
            if cds is not None:
                enc["crop_deepstack"] = [x.view(len(sq), 4, *x.shape[1:]) for x in cds]
        return enc

    @staticmethod
    def _per_prompt(full, crop=None):
        """(N, T, d) [+ (N, 4, T, d)] -> (N*4*tokens, d) in prompt order: photo tokens then crop tokens."""
        n, t, d = full.shape
        x = full[:, None].expand(n, 4, t, d)
        if crop is not None:
            x = torch_cat([x, crop.to(full.dtype)], 2)
        return x.reshape(-1, d)

    def logits_from(self, enc, crops: bool = False):
        """-> tensor (N, 4 sites, 4 classes) of answer-letter logits. crops=True uses the
        photo + limb close-up prompts (needs encode(..., crops=...))."""
        if not self.qwen:
            return self._logits_generic(enc, bool(crops and "crops" in enc))
        torch = self.torch
        core, _, lm, rope = self._parts()
        crops = bool(crops and "crop_embeds" in enc)
        ids, mask, thw_p = self._text_template(crops)
        n = enc["n"]
        input_ids = ids.repeat(n, 1).to(self.device)            # image-major: 4 questions per image
        attn = mask.repeat(n, 1).to(self.device)
        thw = thw_p.to(self.device).repeat(4 * n, 1)
        tok = lm.embed_tokens(input_ids)
        d = tok.shape[-1]
        img = self._per_prompt(enc["image_embeds"].to(tok.dtype), enc["crop_embeds"] if crops else None)
        is_img = input_ids == core.config.image_token_id
        tok = tok.masked_scatter(is_img.unsqueeze(-1).expand_as(tok), img)
        pos, _ = rope(input_ids, thw, None, attn)
        extra = {}
        if "deepstack" in enc:  # Qwen3-VL adds multi-level vision features inside early decoder layers
            cds = enc.get("crop_deepstack") if crops else None
            extra = {"visual_pos_masks": is_img,
                     "deepstack_visual_embeds": [self._per_prompt(x.to(tok.dtype), cds[k] if cds else None)
                                                 for k, x in enumerate(enc["deepstack"])]}
        cache_pos = torch.arange(input_ids.shape[1], device=self.device)
        hidden = lm(input_ids=None, attention_mask=attn, position_ids=pos, inputs_embeds=tok, use_cache=False,
                    cache_position=cache_pos, **extra).last_hidden_state[:, -1]
        w = core.lm_head.weight[self.letter_ids]
        logits = (hidden.to(w.dtype) @ w.T).float()
        return logits.view(n, 4, len(LETTERS))

    def _logits_generic(self, enc, crops: bool):
        """Any HF image-text-to-text model (Gemma 4): plain forward over the 4 limb prompts per photo,
        answer-letter logits at the last position. Slower than the Qwen fast path (the vision encoder
        runs once per prompt), but Gemma 4 E4B's encoder is small (~150M params)."""
        n = enc["n"]
        texts = [p for _ in range(n) for p in self.prompts[crops]]  # image-major: 4 questions per photo
        ims = [[enc["images"][i]] + ([enc["crops"][i][s]] if crops else []) for i in range(n) for s in range(4)]
        if "_inputs" not in enc:
            enc["_inputs"] = {}
        if crops not in enc["_inputs"]:  # processor work done once per view, reused by every adapter
            enc["_inputs"][crops] = self.processor(text=texts, images=ims, padding=True, return_tensors="pt")
        inp = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
               for k, v in enc["_inputs"][crops].items() if hasattr(v, "to")}
        try:
            out = self.model(**inp, use_cache=False, logits_to_keep=1)
        except TypeError:  # model without logits_to_keep
            out = self.model(**inp, use_cache=False)
        return out.logits[:, -1, self.letter_ids].float().view(n, 4, len(LETTERS))

    def letter_logits(self, images, crops=None):
        return self.logits_from(self.encode(images, crops), crops is not None)

    def letter_logits_reference(self, images):
        """Original slow path, kept only to check the fast path gives the same numbers."""
        from .data import letterbox

        sq = [letterbox(im, self.image_size) for im in images]
        texts = [p for _ in sq for p in self.prompts[False]]
        ims = [im for im in sq for _ in range(4)]
        enc = {k: v.to(self.device) for k, v in self.processor(text=texts, images=ims, padding=True,
                                                               return_tensors="pt").items()}
        out = self.model(**enc, use_cache=False)
        return out.logits[:, -1, :][:, self.letter_ids].float().view(len(images), 4, len(LETTERS))

    def encode_views(self, images, tta: bool = True, crops=None):
        from PIL import ImageOps

        views = [self.encode(images, crops)]
        if tta:
            views.append(self.encode([ImageOps.mirror(im) for im in images],
                                     mirror_view_crops(crops) if crops is not None else None))
        return views

    def probs_from(self, views, crops: bool = False):
        p = self.logits_from(views[0], crops).softmax(-1)
        if len(views) > 1:
            pm = self.logits_from(views[1], crops).softmax(-1)
            p = (p + pm[:, list(FLIP_PERMUTATION)]) / 2
        return p

    def predict_probs(self, images, tta: bool = True, crops=None):
        with self.torch.inference_mode():
            return self.probs_from(self.encode_views(images, tta, crops), crops is not None)


# ── training (one fold) ─────────────────────────────────────────────────────────

def train_vlm(args, train_ids, val_ids, labels, out_dir: Path, device, fallback, keypoints=None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model

    from .data import CLASS_TO_IDX, _augment, _augment_crop, flip_labels, load_image

    base = args.backbone.split(":", 1)[1] or DEFAULT_BASE
    clf = VLMClassifier(base, device, image_size=args.vlm_size, train=True)
    model = clf.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_targets(model)))
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
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # ignore_index -100 = unknown label
    use_crops = keypoints is not None
    image_root = getattr(args, "image_root_of", {})  # extra (video-frame) ids live in another folder
    cache: dict = {}

    def get(iid):
        """decode once; keep a downsized photo (VLM sees 448 px anyway) and, in crop mode, the 4
        limb close-ups cut from the FULL-resolution photo (that is where the wound detail is)."""
        if iid not in cache:
            im = load_image(Path(image_root.get(iid, args.images)) / iid)
            crops = vlm_crops(im, keypoints.get(iid), args.vlm_size) if use_crops else None
            s = min(1.0, 2 * args.vlm_size / max(im.size))
            small = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s)))) if s < 1 else im
            cache[iid] = (small, crops)
        return cache[iid]

    def batch_crops(iids):
        return [get(i)[1] for i in iids] if use_crops else None

    def evaluate():
        model.eval()
        probs = []
        for k in range(0, len(val_ids), args.batch):
            chunk = val_ids[k:k + args.batch]
            probs.append(clf.predict_probs([get(i)[0] for i in chunk], crops=batch_crops(chunk)).cpu())
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
            ims, ys, crs = [], [], []
            for iid in order[k:k + args.batch]:
                small, crops = get(iid)
                im, lab = _augment(small, rng), list(labels[iid])
                if crops is not None:
                    crops = [_augment_crop(c, rng) for c in crops]
                if rng.random() < 0.5:
                    from PIL import ImageOps
                    im, lab = ImageOps.mirror(im), flip_labels(lab)
                    if crops is not None:
                        crops = mirror_view_crops([crops])[0]
                ims.append(im)
                crs.append(crops)
                ys.append([CLASS_TO_IDX.get(c, -100) for c in lab])  # None (unknown) -> ignored
            y = torch.tensor(ys, device=device).view(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = clf.letter_logits(ims, crs if use_crops else None).view(-1, len(LETTERS))
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
                _save(model, out_dir, base, args, fallback, acc, use_crops)
                msg += " *"
        print(msg, flush=True)
    if not val_ids:
        _save(model, out_dir, base, args, fallback, None, use_crops)
        best = {"overall_accuracy": None, "epoch": args.epochs}
    return best


def _save(model, out_dir: Path, base: str, args, fallback, acc, limb_crops: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)  # adapter only (~tens of MB)
    (out_dir / META).write_text(json.dumps({
        "format": FORMAT, "base": base, "image_size": args.vlm_size, "classes": list(CLASSES),
        "limb_crops": bool(limb_crops),
        "fallback": list(fallback), "metrics": {"overall_accuracy": acc},
        "data_source": getattr(args, "data_source", "unspecified")}, indent=1))


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
            crops = bool(json.loads((d / META).read_text()).get("limb_crops"))
            if pm is None:
                pm = PeftModel.from_pretrained(self.clf.model, str(d), adapter_name=name)
            else:
                pm.load_adapter(str(d), adapter_name=name)
            self.names.append((name, d.name, crops))
        self.clf.model = pm.eval()
        log.info("vlm members %s on base %s", [n for _, n, _ in self.names], path)


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
        self.needs_pose = any(c for g in self.groups for _, _, c in g.names)

    def member_probs(self, images, keypoints=None):
        """-> list of (member dir name, probs tensor (N,4,4)) one per adapter.
        keypoints (per image, or None) feed the limb close-ups of crop-mode adapters."""
        out = []
        for g in self.groups:
            crops = None
            if any(c for _, _, c in g.names):
                kps = keypoints or [None] * len(images)
                crops = [vlm_crops(im, kp, g.clf.image_size) for im, kp in zip(images, kps)]
            with g.clf.torch.inference_mode():
                views = g.clf.encode_views(images, self.tta, crops)  # vision once, shared by every adapter
                for adapter, dname, use_crops in g.names:
                    g.clf.model.set_adapter(adapter)
                    out.append((dname, g.clf.probs_from(views, use_crops)))
        return out

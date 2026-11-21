"""
Inference: /app/input/*.{jpg,jpeg,png} → /app/output/predictions.json

Guarantees (each one is a way teams get disqualified):
  * every input image gets exactly one prediction with all four sites
  * output is always schema-valid, even if the model, CUDA or an image fails
  * hard wall-clock budget (ICD: 1 h) — remaining images get the fallback prior
  * no network access required

Backends, in priority order:
  1. ensemble of trained checkpoints in model/weights/*.pt (+ mirror TTA); limb-crop
     checkpoints also get pose keypoints from model/pose/ (baked in, never downloaded)
  2. Gemma 4 vision zero-shot (only if D2_ENABLE_VLM=1 and weights are baked in)
  3. per-site class prior saved in the checkpoint (or DEFAULT_FALLBACK)
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .schema import (CLASSES, DEFAULT_FALLBACK, build_document, list_images, make_prediction,
                     validate_document, write_document)

log = logging.getLogger("d2qual")


class Ensemble:
    """
    Members: every *.pt checkpoint (CNN / ViT, optionally limb-crop) and every sub-folder
    with vlm_meta.json (LoRA adapter on a shared, baked-in VLM base in model/vlm_base/).
    Weights come from weights_dir/ensemble.json ({"weights": {member: w}}) written by
    scripts/leaderboard.py; without it every member counts equally.
    """

    def __init__(self, weights_dir: Path, device):
        import json

        import torch

        from .model import load_checkpoint

        self.torch = torch
        self.device = device
        self.members = []
        self.fallback: list[str] = list(DEFAULT_FALLBACK)
        self.pose = None
        self.vlm = None
        weights_dir = Path(weights_dir)
        wfile = weights_dir / "ensemble.json"
        edoc = json.loads(wfile.read_text()) if wfile.exists() else {}
        self.weights = edoc.get("weights")
        self.stacker = edoc.get("stacker")  # per-label combiner over candidates (d2qual/stack.py)
        for p in sorted(weights_dir.glob("*.pt")):
            if self.weights is not None and self.weights.get(p.name, 0) <= 0:
                continue
            try:
                model, ckpt = load_checkpoint(p, device)
                if device.type == "cuda":
                    model = model.half()
                self.members.append((p.name, model))
                self.fallback = ckpt.get("fallback", self.fallback)
                log.info("loaded %s (cv acc %s)", p.name, ckpt.get("metrics", {}).get("overall_accuracy"))
            except Exception as e:  # a corrupt member must not take down the run
                log.error("skipping checkpoint %s: %s", p, e)
        vlm_dirs = [d for d in sorted(weights_dir.iterdir()) if (d / "vlm_meta.json").exists()
                    and (self.weights is None or self.weights.get(d.name, 0) > 0)]
        if vlm_dirs:
            if device.type != "cuda":  # a 2B VLM on CPU would blow the 1-hour budget
                log.error("skipping %d VLM members: no GPU", len(vlm_dirs))
            else:
                try:
                    from .vlm_ft import VLMEnsemble

                    self.vlm = VLMEnsemble(vlm_dirs, device, weights_dir.parent / "vlm_base")
                    if not self.members and self.vlm.fallback:
                        self.fallback = list(self.vlm.fallback)
                except Exception as e:
                    log.exception("VLM members failed to load (%s); continuing without them", e)
        if any(m.cfg.limb_crops for _, m in self.members) or (self.vlm is not None and self.vlm.needs_pose):
            from .pose import PoseExtractor

            self.pose = PoseExtractor.try_load(weights_dir, device)
            if self.pose is None:  # limb models still run, on blank crops (global view only)
                log.error("limb-crop checkpoints without a pose model: accuracy will drop")

    def __bool__(self) -> bool:
        return bool(self.members) or self.vlm is not None

    def _w(self, name: str) -> float:
        return 1.0 if self.weights is None else float(self.weights.get(name, 0.0))

    def predict_probs(self, images) -> "list[list[list[float]]]":
        """images: list[PIL.Image] -> probs [N][4][4], weighted mean over members x {orig, mirror}."""
        from .data import letterbox, limb_crops, to_tensor
        from .model import flip_logits, mirror_inputs

        torch = self.torch
        half = self.device.type == "cuda"
        keypoints = None
        if any(m.cfg.limb_crops for _, m in self.members) or (self.vlm is not None and self.vlm.needs_pose):
            keypoints = [None] * len(images)
            if self.pose is not None:
                try:
                    keypoints = self.pose.extract(images)
                except Exception as e:  # pose failure must not kill the batch
                    log.error("pose extraction failed (%s); using blank limb crops", e)
        total, wsum = None, 0.0
        per_member = {}  # member name -> (N,4,4) probs, for the stacker
        for name, model in self.members:
            cfg = model.cfg
            x = torch.stack([to_tensor(letterbox(im, cfg.image_size), cfg.mean, cfg.std) for im in images])
            crops = kpf = None
            if cfg.limb_crops:
                cs, fs = zip(*(limb_crops(im, kp, cfg.crop_size) for im, kp in zip(images, keypoints)))
                crops = torch.stack([torch.stack([to_tensor(c, cfg.mean, cfg.std) for c in four]) for four in cs])
                kpf = torch.stack([torch.from_numpy(f) for f in fs])
            x = x.to(self.device).half() if half else x.to(self.device)
            if crops is not None:
                crops = crops.to(self.device).half() if half else crops.to(self.device)
                kpf = kpf.to(self.device).half() if half else kpf.to(self.device)
            with torch.inference_mode():
                p = model(x, crops, kpf).float().softmax(-1)
                p = (p + flip_logits(model(*mirror_inputs(x, crops, kpf)).float()).softmax(-1)) / 2
            per_member[name] = p
            w = self._w(name)
            total = p * w if total is None else total + p * w
            wsum += w
        if self.vlm is not None:
            try:
                for name, p in self.vlm.member_probs(images, keypoints):
                    w = self._w(name)
                    p = p.to(total.device if total is not None else p.device).float()
                    per_member[name] = p
                    total = p * w if total is None else total + p * w
                    wsum += w
            except Exception as e:  # keep the CNN result if the VLM fails mid-run
                log.exception("VLM inference failed on a batch: %s", e)
        if total is None or wsum <= 0:
            raise RuntimeError("no ensemble member produced output")
        stacked = self._stack(per_member)
        if stacked is not None:
            return stacked.tolist()
        return (total / wsum).cpu().tolist()

    def embed(self, images):
        """L2-normalised global features from the first whole-image CNN/ViT member (for pooling)."""
        from .data import letterbox, to_tensor

        torch = self.torch
        name, model = next((n, m) for n, m in self.members if not m.cfg.limb_crops)
        cfg = model.cfg
        x = torch.stack([to_tensor(letterbox(im, cfg.image_size), cfg.mean, cfg.std) for im in images]).to(self.device)
        if self.device.type == "cuda":
            x = x.half()
        with torch.inference_mode():
            f = model.backbone(x).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def _stack(self, per_member):
        """Average each candidate's fold members, then apply the stacker. None if it can't run
        (a candidate failed to load) -> caller uses the weighted average instead."""
        if not self.stacker:
            return None
        import numpy as np

        from .stack import apply

        by_cand: dict = {}
        for name, p in per_member.items():
            by_cand.setdefault(name.split("__")[0], []).append(p.float().cpu().numpy())
        try:
            per = [np.mean(by_cand[c], 0) for c in self.stacker["candidates"]]
        except KeyError as e:
            log.error("stacker candidate %s missing; using weighted average", e)
            return None
        return apply(self.stacker, per)


def cluster(emb, sim: float) -> list[int]:
    """Average-linkage agglomerative clustering on cosine similarity; merge while the best pair of
    clusters is at least `sim` similar. emb: (N, D) L2-normalised. -> cluster id per row."""
    import numpy as np

    emb = np.asarray(emb, np.float64)
    s = emb @ emb.T
    clusters = [[i] for i in range(len(emb))]
    while len(clusters) > 1:
        best, pair = -2.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                v = s[np.ix_(clusters[a], clusters[b])].mean()
                if v > best:
                    best, pair = v, (a, b)
        if best < sim:
            break
        a, b = pair
        clusters[a] += clusters.pop(b)
    out = [0] * len(emb)
    for k, c in enumerate(clusters):
        for i in c:
            out[i] = k
    return out


def _pool_predictions(all_probs: dict, embeds: dict, labels: dict, sim: float, alpha: float) -> None:
    import numpy as np

    from .stack import pool_by_group

    names = [n for n in embeds if n in all_probs]
    groups = cluster(np.stack([embeds[n] for n in names]), sim)
    pooled = pool_by_group(np.asarray([all_probs[n] for n in names]), groups, alpha)
    for n, pr in zip(names, pooled):
        labels[n] = [CLASSES[int(np.argmax(site))] for site in pr]
    log.info("pooling: %d photos in %d clusters (sim %.2f, alpha %.2f)", len(names), len(set(groups)), sim, alpha)


def _select_device():
    import torch

    if os.environ.get("D2_FORCE_CPU") != "1" and torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            log.error("CUDA visible but unusable (%s) — falling back to CPU", e)
    return torch.device("cpu")


def _safe_load(path: Path):
    from .data import load_image

    try:
        return load_image(path)
    except Exception as e:
        log.error("cannot decode %s: %s", path.name, e)
        return None


def run(input_dir: str | Path, output_path: str | Path, weights_dir: str | Path,
        team_name: str, version: str, email: str,
        time_budget_s: float = 50 * 60, batch_size: int = 16) -> dict:
    t_start = time.monotonic()
    images = list_images(input_dir)
    log.info("%d images in %s", len(images), input_dir)

    labels: dict[str, list[str]] = {}
    fallback = list(DEFAULT_FALLBACK)

    # Write a valid all-fallback file FIRST: if anything later is killed
    # (OOM, timeout), a schema-valid predictions.json is already on disk.
    def _flush():
        preds = [make_prediction(p.name, labels.get(p.name, fallback)) for p in images]
        doc = build_document(preds, team_name, version, email)
        write_document(doc, output_path)
        return doc

    _flush()

    # D2_POOL=1: share injury evidence across photos that look like the same casualty (off by default;
    # only enable once DARPA confirms predictions may depend on other test images).
    pool = os.environ.get("D2_POOL") == "1"
    all_probs: dict = {}
    embeds: dict = {}
    ensemble: Optional[Ensemble] = None
    vlm = None
    try:
        device = _select_device()
        ensemble = Ensemble(Path(weights_dir), device)
        if ensemble:
            fallback = list(ensemble.fallback)
        elif os.environ.get("D2_ENABLE_VLM") == "1":
            from .vlm import GemmaVisionClassifier
            vlm = GemmaVisionClassifier.try_load(device)
        log.info("backend: %s on %s", "ensemble" if ensemble else ("vlm" if vlm else "prior"), device)
    except Exception as e:
        log.exception("backend init failed, using prior: %s", e)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        for start in range(0, len(images), batch_size):
            if time.monotonic() - t_start > time_budget_s:
                log.error("time budget hit at %d/%d images — rest use prior", start, len(images))
                break
            chunk = images[start:start + batch_size]
            pil = list(pool.map(_safe_load, chunk))
            ok = [(p, im) for p, im in zip(chunk, pil) if im is not None]
            for p, im in zip(chunk, pil):
                if im is None:
                    labels[p.name] = ["not_testable"] * 4  # undecodable == nothing visible
            if not ok:
                continue
            try:
                if ensemble:
                    probs = ensemble.predict_probs([im for _, im in ok])
                    for (p, _), pr in zip(ok, probs):
                        labels[p.name] = [CLASSES[max(range(4), key=site.__getitem__)] for site in pr]
                        all_probs[p.name] = pr
                    if pool:
                        try:
                            for (p, _), e in zip(ok, ensemble.embed([im for _, im in ok])):
                                embeds[p.name] = e
                        except Exception as e:
                            log.error("embedding failed (%s); pooling skips this batch", e)
                elif vlm is not None:
                    for p, im in ok:
                        labels[p.name] = vlm.classify(im) or fallback
            except Exception as e:
                log.exception("inference failed on batch at %d: %s", start, e)
                if ensemble and getattr(ensemble.device, "type", "") == "cuda" and batch_size > 1:
                    # most likely OOM: retry this chunk one image at a time on the same device
                    for p, im in ok:
                        try:
                            pr = ensemble.predict_probs([im])[0]
                            labels[p.name] = [CLASSES[max(range(4), key=s.__getitem__)] for s in pr]
                        except Exception:
                            pass
            if (start // batch_size) % 20 == 0:
                _flush()
                log.info("%d/%d done (%.0fs)", min(start + batch_size, len(images)), len(images),
                         time.monotonic() - t_start)

    if pool and len(embeds) > 1:
        try:
            _pool_predictions(all_probs, embeds, labels, float(os.environ.get("D2_POOL_SIM", "0.9")),
                              float(os.environ.get("D2_POOL_ALPHA", "1.0")))
        except Exception as e:  # pooling is an optional refinement; keep per-photo labels
            log.exception("pooling failed: %s", e)
    doc = _flush()
    errs = validate_document(doc, [p.name for p in images])
    if errs:  # should be impossible; never leave an invalid file behind
        log.error("self-validation failed: %s", errs[:10])
        doc = build_document([make_prediction(p.name, fallback) for p in images], team_name, version, email)
        write_document(doc, output_path)
    log.info("wrote %s (%d predictions) in %.1fs", output_path, len(images), time.monotonic() - t_start)
    return doc

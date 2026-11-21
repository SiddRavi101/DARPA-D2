"""
Per-label stacking combiner and casualty-level pooling.

Stacker: a small multinomial logistic regression that takes every candidate model's 4 class
probabilities for one limb (as log-probabilities) plus a site one-hot, and outputs the final
4 class probabilities. Unlike one weight per model (Caruana selection), it can learn
"trust the VLM for not_testable, trust the limb-crop CNN for wound / amputation".
It is fitted on out-of-fold (OOF) probabilities only, and scored with a second, nested
grouped split so its reported accuracy is honest.

Pooling: a casualty's injuries do not change between photos, but visibility does. Given a
grouping of photos (true casualty ids for the OOF diagnostic, or clusters at test time) it
shares the injury evidence (no_injury / wound / amputation) across the group, weighted by how
visible the limb is in each photo, and keeps each photo's own not_testable probability.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-4
NT = 3  # index of not_testable in CLASSES


def features(per_cand: list[np.ndarray]) -> np.ndarray:
    """per_cand: list over candidates of (N, 4 sites, 4 classes) probs -> (N*4, 4*C + 4) features."""
    n = per_cand[0].shape[0]
    logp = [np.log(np.clip(p, EPS, 1.0)).reshape(n * 4, 4) for p in per_cand]
    site = np.tile(np.eye(4, dtype=np.float64), (n, 1))
    return np.concatenate(logp + [site], 1)


def fit(x: np.ndarray, y: np.ndarray, l2: float, iters: int = 300) -> dict:
    """Multinomial logistic regression, L2 on weights (not bias). y: int class per row (-100 = skip)."""
    import torch

    keep = y >= 0
    xt = torch.tensor(x[keep], dtype=torch.float64)
    yt = torch.tensor(y[keep], dtype=torch.long)
    w = torch.zeros(x.shape[1], 4, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"W": w.detach().numpy().tolist(), "b": b.detach().numpy().tolist(), "l2": l2}


def apply(model: dict, per_cand: list[np.ndarray]) -> np.ndarray:
    """-> (N, 4, 4) probabilities."""
    x = features(per_cand)
    z = x @ np.asarray(model["W"]) + np.asarray(model["b"])
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p.reshape(-1, 4, 4)


def nested_accuracy(per_cand: list[np.ndarray], y: np.ndarray, folds: list[np.ndarray], l2: float):
    """Fit on all folds but one, predict the held-out fold; -> (accuracy, OOF stacked probs)."""
    out = np.zeros(per_cand[0].shape)
    for held in folds:
        tr = np.setdiff1d(np.arange(len(y)), held)
        m = fit(features([p[tr] for p in per_cand]), y[tr].reshape(-1), l2)
        out[held] = apply(m, [p[held] for p in per_cand])
    return float((out.argmax(-1) == y).mean()), out


def pool_by_group(probs: np.ndarray, groups: list, alpha: float = 1.0) -> np.ndarray:
    """probs (N,4,4); groups: group key per row (None = alone). alpha 0 = off, 1 = full pooling.

    Per group and site: injury mix = visibility-weighted mean of p[:3]/sum(p[:3]); each photo keeps
    its own p(not_testable) and gets (1 - p_nt) * blend(own mix, group mix)."""
    out = probs.copy()
    by: dict = {}
    for i, g in enumerate(groups):
        if g is not None:
            by.setdefault(g, []).append(i)
    for idx in by.values():
        if len(idx) < 2:
            continue
        p = probs[idx]                                   # (k, 4, 4)
        vis = 1.0 - p[..., NT]                           # (k, 4)
        mix = p[..., :NT] / np.clip(p[..., :NT].sum(-1, keepdims=True), EPS, None)
        group_mix = (mix * vis[..., None]).sum(0) / np.clip(vis.sum(0)[..., None], EPS, None)  # (4, 3)
        new_mix = (1 - alpha) * mix + alpha * group_mix[None]
        out[idx, :, :NT] = new_mix * vis[..., None]
        out[idx, :, NT] = p[..., NT]
    return out

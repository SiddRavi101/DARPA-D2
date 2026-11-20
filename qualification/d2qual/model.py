"""
Shared-backbone, four-head injury classifier.

Global mode (default): one timm backbone -> pooled features -> Linear(4 sites x 4 classes).
Sites share visual evidence (a manikin lying on its side hides a limb), so a joint model
beats four independent ones and costs one forward pass.

Limb-crop mode (`limb_crops=True`): the same backbone also encodes a close-up crop of each
limb, found by the pose model (d2qual/pose.py). Each site's head sees
    [global features | that limb's crop features | keypoint confidences | site one-hot]
so small wounds get pixels and left/right comes from the pose model's anatomical sides,
while the global view still covers limbs the pose model misses.

Checkpoints are self-contained (config + weights + fallback prior) so the
container loads them with `pretrained=False` and never touches the network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .schema import CLASSES, FLIP_PERMUTATION, SITES

N_SITES = len(SITES)
N_CLASSES = len(CLASSES)


@dataclass
class ModelConfig:
    backbone: str = "convnext_small.fb_in22k_ft_in1k_384"
    image_size: int = 512
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    limb_crops: bool = False
    crop_size: int = 224
    extra: dict = field(default_factory=dict)


def _create_backbone(cfg: "ModelConfig", pretrained: bool):
    import timm

    kw = dict(pretrained=pretrained, num_classes=0, drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate)
    name = cfg.backbone.lower()
    if any(t in name for t in ("vit", "eva", "deit", "beit", "siglip", "dinov")):
        # transformers: accept the global size and the crop size in the same model
        kw.update(img_size=cfg.image_size, dynamic_img_size=True)
    try:
        return timm.create_model(cfg.backbone, **kw)
    except TypeError:  # backbone without dynamic_img_size support
        kw.pop("dynamic_img_size", None)
        return timm.create_model(cfg.backbone, **kw)


class InjuryNet(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = False):
        super().__init__()
        self.cfg = cfg
        self.backbone = _create_backbone(cfg, pretrained)
        d = self.backbone.num_features
        if cfg.limb_crops:
            from .data import N_KP_FEATS

            self.register_buffer("site_onehot", torch.eye(N_SITES), persistent=False)
            self.head = nn.Sequential(
                nn.Dropout(cfg.drop_rate),
                nn.Linear(2 * d + N_KP_FEATS + N_SITES, 512),
                nn.GELU(),
                nn.Dropout(cfg.drop_rate),
                nn.Linear(512, N_CLASSES),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(cfg.drop_rate),
                nn.Linear(d, N_SITES * N_CLASSES),
            )

    def forward(self, x: torch.Tensor, crops: torch.Tensor | None = None,
                kp_feats: torch.Tensor | None = None) -> torch.Tensor:
        g = self.backbone(x)
        if not self.cfg.limb_crops:
            return self.head(g).view(-1, N_SITES, N_CLASSES)
        b = x.shape[0]
        c = self.backbone(crops.flatten(0, 1)).view(b, N_SITES, -1)  # (B, 4, D)
        site = self.site_onehot.to(c.dtype).expand(b, -1, -1)
        z = torch.cat([g[:, None, :].expand(-1, N_SITES, -1), c, kp_feats.to(c.dtype), site], -1)
        return self.head(z)  # (B, 4, 4)


def mirror_inputs(x: torch.Tensor, crops: torch.Tensor | None = None, kp_feats: torch.Tensor | None = None):
    """Mirror a batch: flip the image; for limb inputs also swap sites and flip each crop."""
    xm = torch.flip(x, dims=[3])
    if crops is None:
        return xm, None, None
    perm = list(FLIP_PERMUTATION)
    return xm, torch.flip(crops[:, perm], dims=[4]), kp_feats[:, perm]


def flip_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map logits predicted on a mirrored image back to anatomical sites."""
    return logits[:, list(FLIP_PERMUTATION), :]


def save_checkpoint(path: str | Path, model: InjuryNet, fallback: list[str], metrics: dict,
                    data_source: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "d2qual-v1",
            "config": asdict(model.cfg),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "fallback": list(fallback),
            "metrics": metrics,
            "data_source": data_source,  # e.g. "darpa"; scripts/submit.sh ships only "darpa"
            "classes": list(CLASSES),
            "sites": [list(s) for s in SITES],
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[InjuryNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "d2qual-v1":
        raise ValueError(f"{path}: not a d2qual checkpoint")
    if list(ckpt["classes"]) != list(CLASSES):
        raise ValueError(f"{path}: class order mismatch {ckpt['classes']}")
    known = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: tuple(v) if k in ("mean", "std") else v for k, v in ckpt["config"].items() if k in known})
    model = InjuryNet(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt

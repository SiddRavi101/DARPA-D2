"""
Shared-backbone, four-head injury classifier.

One timm backbone → pooled features → Linear(4 sites × 4 classes).
Sites share visual evidence (a manikin lying on its side hides a limb), so a
joint model beats four independent ones and costs one forward pass.

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
    extra: dict = field(default_factory=dict)


class InjuryNet(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = False):
        super().__init__()
        import timm

        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=pretrained, num_classes=0,
            drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.drop_rate),
            nn.Linear(self.backbone.num_features, N_SITES * N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).view(-1, N_SITES, N_CLASSES)


def flip_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map logits predicted on a mirrored image back to anatomical sites."""
    return logits[:, list(FLIP_PERMUTATION), :]


def save_checkpoint(path: str | Path, model: InjuryNet, fallback: list[str], metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "d2qual-v1",
            "config": asdict(model.cfg),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "fallback": list(fallback),
            "metrics": metrics,
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
    cfg = ModelConfig(**{k: tuple(v) if k in ("mean", "std") else v for k, v in ckpt["config"].items()})
    model = InjuryNet(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt

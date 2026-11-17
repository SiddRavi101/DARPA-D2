"""
Optional zero-shot backend: Gemma 4 E4B (vision) via Hugging Face transformers.

Ported from SecondLife's on-device pipeline: same 768x768 RGB frame contract
and the same "JSON-only answer, regex-extract the first object" parsing that
survived LiteRT-LM's chatty outputs. On the server GPU we use transformers
instead of LiteRT-LM (LiteRT-LM targets mobile NPUs/CPUs).

Only used when no trained checkpoint exists AND D2_ENABLE_VLM=1 AND weights
are baked into the image at $D2_VLM_PATH (default /app/model/gemma). A
fine-tuned classifier will beat this; it exists so the container produces
better-than-prior answers on day one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from .data import letterbox
from .data import normalize_class

log = logging.getLogger("d2qual.vlm")

PROMPT = (
    "You are documenting a TCCC casualty card from a photo of a medical training manikin.\n"
    "For each of the casualty's four limbs, classify the injury state. LEFT and RIGHT mean the "
    "CASUALTY'S OWN anatomical left/right, not the viewer's. If the casualty faces the camera, "
    "their left limb appears on the right side of the image.\n"
    "Classes: no_injury (limb visible, intact), wound (visible trauma: laceration, burn, "
    "shrapnel, open fracture; limb still attached), amputation (limb completely severed), "
    "not_testable (limb not visible in the image).\n"
    'Respond ONLY with JSON: {"left_arm":"<class>","right_arm":"<class>",'
    '"left_leg":"<class>","right_leg":"<class>"}'
)
_KEYS = ("left_arm", "right_arm", "left_leg", "right_leg")  # = SITES order


def parse_answer(raw: str) -> Optional[list[str]]:
    text = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        return [normalize_class(obj[k]) for k in _KEYS]
    except Exception:
        return None


class GemmaVisionClassifier:
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device

    @classmethod
    def try_load(cls, device) -> Optional["GemmaVisionClassifier"]:
        path = Path(os.environ.get("D2_VLM_PATH", "/app/model/gemma"))
        if not path.exists():
            log.warning("VLM enabled but %s missing", path)
            return None
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            proc = AutoProcessor.from_pretrained(path, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(
                path, local_files_only=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ).to(device).eval()
            return cls(model, proc, device)
        except Exception as e:
            log.error("VLM load failed: %s", e)
            return None

    def classify(self, image) -> Optional[list[str]]:
        import torch

        img = letterbox(image, 768)  # SecondLife frame contract: 768x768 RGB
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": PROMPT}]}]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
            raw = self.processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return parse_answer(raw)
        except Exception as e:
            log.error("VLM inference failed: %s", e)
            return None

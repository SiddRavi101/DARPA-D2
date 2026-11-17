"""
Optional Gemma 4 E4B layer (ported from SecondLife's Military Medic mode).

On the D2 eval GPU we load Gemma through Hugging Face transformers instead of
LiteRT-LM (LiteRT-LM targets phones). Kept from SecondLife:
  * the MILITARY_MEDIC role prompt (TCCC terminology, MARCH order, decisive)
  * the JSON-only "panic mode" contract and first-JSON-object parsing
  * the lesson that a short context window crashes the runtime: we budget
    generously (prompt is truncated, not the window)

The LLM is advisory only: it rewrites the top recommendations into one terse
radio-style line ("speak") for TTS / heads-up display. It never adds or
removes interventions — the rules engine owns clinical content.
Enabled with D2_ENABLE_LLM=1 and weights baked at $D2_LLM_PATH.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("d2sprint.llm")

MILITARY_MEDIC = (
    "You are a Tactical Combat Casualty Care assistant for a military medic. "
    "Use TCCC terminology. Assume austere environment with minimal equipment. "
    "Follow MARCH order: Massive hemorrhage, Airway, Respiration, Circulation, Hypothermia. "
    "Be direct and decisive. No hedging."
)


def parse_json(raw: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", raw), re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


class GemmaAdvisor:
    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device

    @classmethod
    def try_load(cls) -> Optional["GemmaAdvisor"]:
        if os.environ.get("D2_ENABLE_LLM") != "1":
            return None
        path = Path(os.environ.get("D2_LLM_PATH", "/app/models/gemma"))
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
            return cls(model, tok, device)
        except Exception as e:
            log.warning("Gemma unavailable (%s) — rules-only decision support", e)
            return None

    def _chat(self, user: str, max_new_tokens: int = 96) -> str:
        import torch

        msgs = [{"role": "system", "content": MILITARY_MEDIC}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    def polish(self, recs, card) -> None:
        if not recs:
            return
        top = "\n".join(f"{i + 1}. [{r.march}] {r.action}" for i, r in enumerate(recs[:3]))
        raw = self._chat(
            f"Casualty injuries: {json.dumps(card.card['injuries'])[:800]}\n"
            f"Outstanding interventions in priority order:\n{top}\n\n"
            'Respond ONLY with JSON: {"speak":"<25 words max, most critical action first>"}'
        )
        obj = parse_json(raw)
        if obj and obj.get("speak"):
            recs[0].rationale += f" | speak: {str(obj['speak'])[:200]}"

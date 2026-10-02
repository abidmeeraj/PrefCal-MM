"""Pillar 2: image-text alignment S_relevance.

A multimodal LLM judge receives the generated summary together with all
images the system *selected* (V_sel, not the source images) and returns a
1-5 score, normalised to [0, 1].
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import load_config, load_prompts
from ..models import load_mllm
from ..normalisation import normalise
from ..parsing import extract_first_int, strip_chat_artifacts

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ("srelevance", "raw_srelevance", "n_selected_images")


def compute(summary: str, images: List[Any]) -> Tuple[float, Optional[int]]:
    """Return ``(srelevance, raw_score)``.

    With no selected images there is nothing to judge: returns ``(0.0, None)``.
    """
    if not images:
        logger.warning("image_text_relevance: no selected images; returning 0.0")
        return 0.0, None

    import torch

    processor, model = load_mllm("image_text_relevance")
    prompt = load_prompts()["image_text_relevance"].format(summary=summary)
    max_new_tokens = int(load_config().get("decoding", {}).get("max_new_tokens", 512))

    # All selected images in one turn, followed by the text prompt.
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text_prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = processor(images=images, text=text_prompt, return_tensors="pt").to(device, torch.float16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    raw = strip_chat_artifacts(processor.decode(out_ids[0][input_len:], skip_special_tokens=True))

    score = extract_first_int(raw, allowed=range(1, 6))
    if score is None:
        logger.warning("image_text_relevance: no 1-5 score in %r; using midpoint 3", raw[:120])
        score = 3
    return normalise(score, "image_text_relevance"), int(score)


def score(record: Dict[str, Any]) -> Dict[str, Any]:
    images = record["summary"].get("selected_images") or []
    srelevance, raw = compute(record["summary"]["text"], images)
    return {
        "srelevance": float(srelevance),
        "raw_srelevance": "" if raw is None else int(raw),
        "n_selected_images": len(images),
    }

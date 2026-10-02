"""Pillar 1b: G-Eval coherence, fluency and relevance.

One LLM judge rates each dimension on a 1-5 scale; scores are mapped to
[0, 1] by (x - 1) / 4. By default the integer the judge generates under
greedy decoding is used; ``decoding.geval_scoring_mode: probability_weighted``
switches to the expectation over the score tokens (original G-Eval).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ..config import load_config, load_prompts
from ..generation import generate_text, score_prompt_weighted
from ..models import load_llm
from ..parsing import extract_first_int

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ("srel", "scoh", "sflu")

# Coherence and relevance are judged against the source; fluency is not.
_DIMENSIONS = (("coherence", True), ("fluency", False), ("relevance", True))


def _format(template: str, source: Optional[str], summary: str) -> str:
    if source is not None:
        return template.format(source=source, summary=summary)
    return template.format(summary=summary)


def _score_greedy(tokenizer, model, prompt: str, name: str) -> int:
    allowed_range = range(1, 6)
    raw = generate_text(tokenizer, model, prompt)
    score = extract_first_int(raw, allowed=list(allowed_range))
    if score is None:
        any_score = extract_first_int(raw)
        if any_score is not None:
            clipped = max(min(allowed_range), min(max(allowed_range), any_score))
            if clipped != any_score:
                logger.warning("[%s] out-of-range integer %s in %r; clipping to %s",
                               name, any_score, raw[:80], clipped)
                return clipped
        logger.warning("[%s] no score in 1-5 found in %r; using midpoint 3", name, raw[:80])
        return 3
    return score


def _score_weighted(tokenizer, model, prompt: str, name: str) -> float:
    labels = ("1", "2", "3", "4", "5")
    try:
        weighted, _probs, _ = score_prompt_weighted(tokenizer, model, prompt, labels=labels)
        return weighted
    except ValueError as e:
        logger.warning("[%s] probability-weighted scoring failed (%s); using midpoint", name, e)
        return 3.0


def compute(summary: str, source_text: str) -> Tuple[float, float, float]:
    """Return ``(srel, scoh, sflu)`` in [0, 1]."""
    tokenizer, model = load_llm("qualitative")
    prompts = load_prompts()["qualitative"]
    mode = load_config().get("decoding", {}).get("geval_scoring_mode", "greedy_integer")
    scorer = _score_weighted if mode == "probability_weighted" else _score_greedy

    raw_scores = {}
    for name, uses_source in _DIMENSIONS:
        prompt = _format(prompts[name], source_text if uses_source else None, summary)
        raw_scores[name] = scorer(tokenizer, model, prompt, name)

    srel = round((float(raw_scores["relevance"]) - 1) / 4.0, 6)
    scoh = round((float(raw_scores["coherence"]) - 1) / 4.0, 6)
    sflu = round((float(raw_scores["fluency"]) - 1) / 4.0, 6)
    return srel, scoh, sflu


def score(record: Dict[str, Any]) -> Dict[str, Any]:
    srel, scoh, sflu = compute(record["summary"]["text"], record["article"]["source_text"])
    return {"srel": srel, "scoh": scoh, "sflu": sflu}

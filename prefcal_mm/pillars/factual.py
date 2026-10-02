"""Pillar 1a: unified factual consistency S_fact†.

Atomic facts decomposed from the summary are verified by a cascade:

  Stage A  every fact is checked against the source text (text-only LLM);
  Stage B  facts not supported by text are checked against *all* source images
           together by the Visual Atomic Fact Verifier (VAFV, multimodal LLM).

The union verdict is supported if either stage supports the fact. Skipping
Stage B for text-supported facts is exact under the OR rule. Facts whose
verdict cannot be parsed ('unk') are excluded from the denominator:

    S_fact† = #supported / (|A| - |A_unk|)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from ..config import load_config, load_prompts
from ..generation import generate_text
from ..models import load_llm, load_mllm
from ..parsing import parse_batched_verdicts

logger = logging.getLogger(__name__)

SCORE_COLUMNS = (
    "sfact_unified",
    "n_atomic_facts",
    "n_supported_text_only",   # supported by Stage A
    "n_supported_image_only",  # rejected by Stage A, supported by Stage B (VAFV)
    "n_supported_both",        # always 0 under the cascade; kept for verify-all runs
    "n_supported_neither",     # rejected by both stages
    "n_unk",                   # unparseable; excluded from the denominator
    "n_supported_total",
    "atomic_facts_json",       # per-fact verdicts, for auditing
)


def _verifier_budget(n_facts: int) -> int:
    override = load_config().get("decoding", {}).get("verifier_max_new_tokens")
    return int(override) if override else 16 * n_facts + 32


# ── decomposition ──────────────────────────────────────────────────────────

def _parse_facts(raw: str) -> List[str]:
    """Parse facts from bulleted ('- ', '* ', '• ') or numbered ('1.', '1)', '(1)') lines."""
    facts: List[str] = []
    seen: set = set()
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        fact = None
        if s.startswith(("- ", "* ", "• ")):
            fact = s[2:].strip()
        elif s.startswith(("-", "*", "•")) and len(s) > 1:
            fact = s[1:].strip()
        elif re.match(r"^\d+[\.\)]\s*", s):
            fact = re.sub(r"^\d+[\.\)]\s*", "", s).strip()
        elif re.match(r"^\(\d+\)\s*", s):
            fact = re.sub(r"^\(\d+\)\s*", "", s).strip()
        if fact and fact not in seen and len(fact) > 5:
            facts.append(fact)
            seen.add(fact)
    return facts


def decompose(summary: str, pair_id: str = "") -> List[Dict[str, Any]]:
    """Decompose ``summary`` into atomic facts: ``[{"fact": str}, ...]``."""
    tokenizer, model = load_llm("factual_consistency")
    template = load_prompts()["factual_consistency"]["decompose"]
    raw = generate_text(tokenizer, model, template.format(summary=summary))
    facts = _parse_facts(raw)
    if not facts:
        logger.warning(
            "No atomic facts produced for pair_id=%s (summary length=%d)",
            pair_id or "unknown", len(summary),
        )
    return [{"fact": f} for f in facts]


# ── Stage A: text verification ─────────────────────────────────────────────

def verify_text(atomic_facts: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
    """Add ``text_verdict`` to each fact using one batched prompt per summary."""
    if not atomic_facts:
        return atomic_facts

    tokenizer, model = load_llm("factual_consistency")
    template = load_prompts()["factual_consistency"]["verify_batch"]
    claims = "\n".join(f"{i + 1}. {f['fact']}" for i, f in enumerate(atomic_facts))
    prompt = template.format(source=source_text, claims=claims, n=len(atomic_facts))

    raw = generate_text(tokenizer, model, prompt, max_new_tokens=_verifier_budget(len(atomic_facts)))
    verdicts = parse_batched_verdicts(raw, len(atomic_facts))

    n_unk = sum(1 for v in verdicts if v == "unk")
    if n_unk:
        logger.warning("Stage A: %d/%d facts unparseable from %r", n_unk, len(atomic_facts), raw[:240])
    for fact, verdict in zip(atomic_facts, verdicts):
        fact["text_verdict"] = verdict
    return atomic_facts


# ── Stage B: visual verification (VAFV) ───────────────────────────────────

def verify_visual(atomic_facts: List[Dict[str, Any]], source_images: List[Any]) -> List[Dict[str, Any]]:
    """Add ``visual_verdict`` to each fact, presenting all source images at once."""
    if not atomic_facts:
        return atomic_facts

    if not source_images:
        # Nothing to ground against: 'unk' keeps these facts out of the
        # denominator instead of forcing them to not_supported.
        logger.warning("VAFV: no source images; visual_verdict='unk' for %d facts", len(atomic_facts))
        for fact in atomic_facts:
            fact["visual_verdict"] = "unk"
        return atomic_facts

    import torch

    processor, model = load_mllm("vafv")
    template = load_prompts()["vafv_batch"]
    claims = "\n".join(f"{i + 1}. {f['fact']}" for i, f in enumerate(atomic_facts))
    prompt = template.format(claims=claims, n=len(atomic_facts))

    content = [{"type": "image"} for _ in source_images] + [{"type": "text", "text": prompt}]
    text_prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = processor(images=source_images, text=text_prompt, return_tensors="pt").to(
        device, torch.float16
    )
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=_verifier_budget(len(atomic_facts)),
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    raw = processor.decode(out_ids[0][input_len:], skip_special_tokens=True)
    verdicts = parse_batched_verdicts(raw, len(atomic_facts), strip_artifacts=True)

    n_unk = sum(1 for v in verdicts if v == "unk")
    if n_unk:
        logger.warning("VAFV: %d/%d facts unparseable from %r", n_unk, len(atomic_facts), raw[:240])
    for fact, verdict in zip(atomic_facts, verdicts):
        fact["visual_verdict"] = verdict
    return atomic_facts


# ── union aggregation ──────────────────────────────────────────────────────

def unified_score(
    verified_facts: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]], Dict[str, int]]:
    """OR-merge text and visual verdicts and compute S_fact†.

    Per fact:
      supported      if text == supported OR visual == supported
      not_supported  if text == not_supported AND visual == not_supported
      unk            otherwise (one side unparseable, no support evidence)

    Returns ``(score, facts, stats)``. With no facts, or all facts 'unk', the
    score is 1.0 (nothing unsupported was asserted).
    """
    stats = {
        "n_total": len(verified_facts),
        "n_supported_text_only": 0,
        "n_supported_image_only": 0,
        "n_supported_both": 0,
        "n_supported_neither": 0,
        "n_unk": 0,
        "n_supported_total": 0,
    }
    if not verified_facts:
        return 1.0, verified_facts, stats

    for fact in verified_facts:
        tv = fact.get("text_verdict")
        vv = fact.get("visual_verdict")
        text_ok, image_ok = tv == "supported", vv == "supported"

        if text_ok and image_ok:
            fact["verified_by"], fact["verdict"] = "both", "supported"
            stats["n_supported_both"] += 1
        elif text_ok:
            fact["verified_by"], fact["verdict"] = "text", "supported"
            stats["n_supported_text_only"] += 1
        elif image_ok:
            fact["verified_by"], fact["verdict"] = "image", "supported"
            stats["n_supported_image_only"] += 1
        elif tv == "not_supported" and vv == "not_supported":
            fact["verified_by"], fact["verdict"] = "neither", "not_supported"
            stats["n_supported_neither"] += 1
        else:
            fact["verified_by"], fact["verdict"] = "unk", "unk"
            stats["n_unk"] += 1

    stats["n_supported_total"] = (
        stats["n_supported_text_only"] + stats["n_supported_image_only"] + stats["n_supported_both"]
    )
    decided = stats["n_total"] - stats["n_unk"]
    if decided <= 0:
        logger.warning("All %d facts unparseable; S_fact† defaults to 1.0", stats["n_total"])
        score = 1.0
    else:
        score = stats["n_supported_total"] / decided
    return round(score, 6), verified_facts, stats


# ── pillar entry point ─────────────────────────────────────────────────────

def score(record: Dict[str, Any]) -> Dict[str, Any]:
    """Run the cascade for one record and return the ``SCORE_COLUMNS`` values."""
    article, summary = record["article"], record["summary"]
    facts = decompose(summary["text"], pair_id=record.get("pair_id", ""))
    facts = verify_text(facts, article["source_text"])

    needs_visual = [f for f in facts if f.get("text_verdict") != "supported"]
    if needs_visual:
        verify_visual(needs_visual, article.get("source_images") or [])
    for fact in facts:
        if "visual_verdict" not in fact:
            fact["visual_verdict"] = "skipped_text_supported"

    sfact_unified, verified, stats = unified_score(facts)
    return {
        "sfact_unified": float(sfact_unified),
        "n_atomic_facts": len(verified),
        "n_supported_text_only": stats["n_supported_text_only"],
        "n_supported_image_only": stats["n_supported_image_only"],
        "n_supported_both": stats["n_supported_both"],
        "n_supported_neither": stats["n_supported_neither"],
        "n_unk": stats["n_unk"],
        "n_supported_total": stats["n_supported_total"],
        "atomic_facts_json": json.dumps(verified, ensure_ascii=False),
    }

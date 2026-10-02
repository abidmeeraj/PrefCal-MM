"""Text generation helpers for the LLM judges."""
from __future__ import annotations

import math
from typing import Optional

from .config import load_config


def _format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    use_chat_template: bool = True,
) -> str:
    """Greedy single-prompt generation, honouring ``config.yaml::decoding``.

    The prompt is wrapped in the model's chat template when one exists.
    """
    import torch
    from transformers import GenerationConfig

    dec = load_config().get("decoding", {})
    if max_new_tokens is None:
        max_new_tokens = int(dec.get("max_new_tokens", 512))
    do_sample = bool(dec.get("do_sample", False))
    temperature = float(dec.get("temperature", 0))

    formatted_prompt = _format_prompt(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generation_config = GenerationConfig.from_model_config(model.config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = do_sample
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.return_dict_in_generate = True
        # Clear sampling defaults shipped with some model configs so that
        # greedy decoding is not mixed with top-p / top-k settings.
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.temperature = temperature if do_sample else None

        out = model.generate(**inputs, generation_config=generation_config)

    new_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _logsumexp(values: list) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in values))


def get_label_token_ids(tokenizer, labels: tuple = ("1", "2", "3", "4", "5")) -> dict:
    """Map each score label to the token ids that encode it as a single token.

    Space- and newline-prefixed variants are included because tokenizers may
    encode "1" and " 1" differently.
    """
    label_to_token_ids = {}
    for label in labels:
        token_ids = []
        for variant in (label, " " + label, "\n" + label, "\n " + label):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        if not token_ids:
            raise ValueError(
                f"No single-token variant found for label '{label}' with tokenizer "
                f"{tokenizer.__class__.__name__}."
            )
        label_to_token_ids[label] = sorted(set(token_ids))
    return label_to_token_ids


def score_prompt_weighted(
    tokenizer,
    model,
    prompt: str,
    labels: tuple = ("1", "2", "3", "4", "5"),
    use_chat_template: bool = True,
) -> tuple:
    """Probability-weighted score from next-token logits (G-Eval, Liu et al. 2023).

    Returns ``(weighted_score, probs, logprobs)`` where ``probs`` is the
    distribution renormalised over ``labels``.
    """
    import torch

    formatted_prompt = _format_prompt(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

    label_to_logprob = {}
    for label, token_ids in get_label_token_ids(tokenizer, labels).items():
        label_to_logprob[label] = _logsumexp([float(logprobs[t].cpu()) for t in token_ids])

    max_lp = max(label_to_logprob.values())
    unnormalised = {k: math.exp(lp - max_lp) for k, lp in label_to_logprob.items()}
    z = sum(unnormalised.values())
    probs = {k: p / z for k, p in unnormalised.items()}
    weighted_score = sum(int(k) * p for k, p in probs.items())
    return weighted_score, probs, label_to_logprob

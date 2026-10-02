"""Lazy, cached loaders for the LLM, MLLM and CLIP judges.

Model identifiers are read from ``configs/models.yaml``. Each model is loaded
once per process; roles that resolve to the same identifier share weights.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

from .config import load_models_config

logger = logging.getLogger(__name__)

_LLM_ROLES = {"factual_consistency", "qualitative"}
_MLLM_ROLES = {"image_text_relevance", "vafv"}
_CLIP_ROLES = {"visual_diversity"}

# model_id -> (processor_or_tokenizer, model); role -> same tuple.
_by_model: dict = {}
_by_role: dict = {}


def _offline() -> bool:
    """True if Hugging Face offline mode is set, so loads skip the hub probe."""
    return (
        os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    )


def resolve_model_id(role: str) -> str:
    """Model identifier for ``role``.

    Resolution order: environment variable ``PREFCAL_MODEL_<ROLE>``, then
    ``configs/models.yaml::<role>.final``.
    """
    env_key = f"PREFCAL_MODEL_{role.upper()}"
    if os.environ.get(env_key):
        override = os.environ[env_key]
        logger.info("Using %s=%s for role=%s", env_key, override, role)
        return override

    cfg = load_models_config()
    if role not in cfg or "final" not in cfg[role]:
        raise KeyError(f"Role '{role}' has no 'final' entry in configs/models.yaml")
    return cfg[role]["final"]


def load_llm(role: str) -> Tuple[object, object]:
    """Load ``(tokenizer, model)`` for a text-only role."""
    if role not in _LLM_ROLES:
        raise ValueError(f"load_llm: unsupported role '{role}'. Expected one of {sorted(_LLM_ROLES)}")
    if role in _by_role:
        return _by_role[role]

    model_id = resolve_model_id(role)
    if model_id in _by_model:
        _by_role[role] = _by_model[model_id]
        return _by_role[role]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading LLM for role=%s : %s", role, model_id)
    local_only = _offline()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_only,
    )
    model.eval()

    pair = (tokenizer, model)
    _by_model[model_id] = pair
    _by_role[role] = pair
    return pair


def load_mllm(role: str) -> Tuple[object, object]:
    """Load ``(processor, model)`` for a vision-language role."""
    if role not in _MLLM_ROLES:
        raise ValueError(f"load_mllm: unsupported role '{role}'. Expected one of {sorted(_MLLM_ROLES)}")
    if role in _by_role:
        return _by_role[role]

    model_id = resolve_model_id(role)
    if model_id in _by_model:
        _by_role[role] = _by_model[model_id]
        return _by_role[role]

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Loading MLLM for role=%s : %s", role, model_id)
    local_only = _offline()
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )
    model.eval()

    pair = (processor, model)
    _by_model[model_id] = pair
    _by_role[role] = pair
    return pair


def load_clip(role: str = "visual_diversity") -> Tuple[object, object]:
    """Load ``(processor, model)`` for the CLIP diversity encoder."""
    if role not in _CLIP_ROLES:
        raise ValueError(f"load_clip: unsupported role '{role}'. Expected one of {sorted(_CLIP_ROLES)}")
    if role in _by_role:
        return _by_role[role]

    model_id = resolve_model_id(role)
    if model_id in _by_model:
        _by_role[role] = _by_model[model_id]
        return _by_role[role]

    import torch
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP for role=%s : %s", role, model_id)
    local_only = _offline()
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_only)
    # CLIP is small; fp32 on a single device avoids fp16 layer-norm
    # instability on larger variants (e.g. ViT-L/14@336).
    model = CLIPModel.from_pretrained(
        model_id, torch_dtype=torch.float32, local_files_only=local_only
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    pair = (processor, model)
    _by_model[model_id] = pair
    _by_role[role] = pair
    return pair


def clear_cache() -> None:
    """Release all cached models."""
    import gc

    _by_role.clear()
    _by_model.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

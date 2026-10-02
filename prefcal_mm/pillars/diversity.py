"""Pillar 3: visual diversity S_div of the selected image set.

Two formulas over CLIP image features F (N images):

  dpp_logdet (default)
      K = F̂ F̂ᵀ over L2-normalised features; S_div = exp(log det(K + εI) / N),
      the geometric mean of the kernel eigenvalues, in [ε, 1].

  tce
      Truncated CLIP Entropy: von Neumann entropy of the top-k eigenvalues of
      the empirical covariance (FᵀF)/(N-1), normalised by a fixed bound.

Diversity is undefined for fewer than two images; such records get NaN and
are excluded from every diversity-dependent comparison.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ..config import load_config
from ..models import load_clip
from ..normalisation import normalise

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ("sdiversity", "raw_diversity", "n_selected_images")
FORMULAS = ("dpp_logdet", "tce")


def _cfg() -> dict:
    return load_config().get("diversity", {}) or {}


def resolve_formula(override: Optional[str] = None) -> str:
    """Explicit argument > PREFCAL_DIVERSITY_FORMULA env var > config."""
    formula = override or os.environ.get("PREFCAL_DIVERSITY_FORMULA") or _cfg().get("formula", "dpp_logdet")
    if formula not in FORMULAS:
        raise ValueError(f"Unknown diversity formula {formula!r}. Valid: {FORMULAS}")
    return formula


def clip_features(images: List[Any]):
    """(N, D) CLIP image features."""
    import torch

    processor, model = load_clip("visual_diversity")
    device = next(model.parameters()).device
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    # transformers >= 5 returns a model output whose pooler_output holds the
    # projected image embeddings; earlier versions return the tensor directly.
    return feats if torch.is_tensor(feats) else feats.pooler_output


def dpp_logdet(feats) -> Tuple[float, float]:
    """Return ``(score, log_det)``; NaN if the kernel is numerically singular."""
    import torch

    eps = float(_cfg().get("dpp_eps", 1e-3))
    feats = torch.nn.functional.normalize(feats, dim=-1)
    n = feats.shape[0]
    K = feats @ feats.T + eps * torch.eye(n, device=feats.device, dtype=feats.dtype)
    sign, logdet = torch.slogdet(K)
    if not bool(sign > 0):
        logger.warning("DPP: slogdet sign=%s (rank-deficient); returning NaN", float(sign))
        return float("nan"), float("nan")
    score = float(torch.exp(logdet / n).item())
    score = max(0.0, min(1.0, score))
    return round(score, 6), round(float(logdet.item()), 6)


def tce(feats) -> Tuple[float, float]:
    """Return ``(normalised_entropy, entropy)``."""
    import torch

    n = feats.shape[0]
    max_eig = int(_cfg().get("tce_max_eigenvalues", 20))
    cov = (feats.T @ feats) / (n - 1)
    cov = (cov + cov.T) / 2.0
    eigvals, _ = torch.sort(torch.linalg.eigvalsh(cov), descending=True)
    eigvals = eigvals[: min(n, max_eig)]
    eigvals = eigvals[eigvals > 0]
    if eigvals.numel() == 0 or eigvals.sum() <= 0:
        return 0.0, 0.0
    probs = eigvals / eigvals.sum()
    raw = max(float(-(probs * torch.log(probs.clamp(min=1e-30))).sum().item()), 0.0)
    lo, hi = (float(b) for b in load_config()["normalisation"]["tce"])
    return normalise(min(max(raw, lo), hi), "tce"), round(raw, 6)


def compute(images: List[Any], formula: Optional[str] = None) -> Tuple[float, float]:
    """Return ``(sdiversity, raw_value)``; ``(nan, nan)`` for fewer than two images."""
    if len(images) < 2:
        return float("nan"), float("nan")

    import torch

    active = resolve_formula(formula)
    feats = clip_features(images).to(torch.float32)
    return dpp_logdet(feats) if active == "dpp_logdet" else tce(feats)


def score(record: Dict[str, Any]) -> Dict[str, Any]:
    images = record["summary"].get("selected_images") or []
    sdiversity, raw = compute(images)
    return {
        "sdiversity": float(sdiversity),
        "raw_diversity": float(raw),
        "n_selected_images": len(images),
    }

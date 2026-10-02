"""Feature-augmented Bradley-Terry aggregation.

P(s_i > s_j) = sigmoid(w · (X_i - X_j)). Maximum-likelihood estimation is L2
logistic regression on feature differences with no intercept. Accuracy is
cross-validated with GroupKFold over articles, so pairs from one article
never appear in both training and validation folds. The reported weights are
the full-data fit with negative coefficients clipped to zero and normalised
to sum to one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import load_config

STAGE1_FEATURES = ["sfact_unified", "srel", "scoh", "sflu"]
STAGE1_KEYS = ["w_fact", "w_rel", "w_coh", "w_flu"]
STAGE2_FEATURES = ["stext_unified", "srelevance", "sdiversity"]
STAGE2_KEYS = ["w_text", "w_relevance", "w_diversity"]


def normalise_nonnegative(raw: np.ndarray) -> np.ndarray:
    """Clip negative weights to zero and normalise to sum to one (uniform if all <= 0)."""
    clipped = np.maximum(raw, 0.0)
    s = float(clipped.sum())
    if s <= 0.0:
        return np.full_like(clipped, 1.0 / len(clipped))
    return clipped / s


def pairwise_design(
    features: np.ndarray,
    pairs: Sequence[Dict[str, Any]],
    sample_to_idx: Dict[Tuple[str, str], int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference matrix, labels (1 = system_i preferred) and article groups.

    Pairs with a system missing from ``sample_to_idx`` are skipped.
    """
    x_rows, y_rows, groups = [], [], []
    for p in pairs:
        aid = str(p["article_id"])
        i = sample_to_idx.get((aid, str(p["system_i"])))
        j = sample_to_idx.get((aid, str(p["system_j"])))
        if i is None or j is None:
            continue
        x_rows.append(features[i] - features[j])
        y_rows.append(1 if p["preferred"] == "i" else 0)
        groups.append(aid)
    if not x_rows:
        raise RuntimeError("No usable preference pairs for Bradley-Terry training")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=int), np.asarray(groups)


def fit(
    feature_lists: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    pairs: Sequence[Dict[str, Any]],
    sample_to_idx: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    """Fit BT weights for the given feature columns; returns raw/normalised weights and CV accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    cfg = load_config().get("bradley_terry", {})
    C = float(cfg.get("C", 100.0))
    max_iter = int(cfg.get("max_iter", 2000))
    folds = int(cfg.get("cv_folds", 5))

    features = np.column_stack(feature_lists).astype(float)
    X, y, groups = pairwise_design(features, pairs, sample_to_idx)

    def _model():
        return LogisticRegression(C=C, fit_intercept=False, max_iter=max_iter)

    cv_scores: List[float] = []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        m = _model()
        m.fit(X[tr], y[tr])
        cv_scores.append(float(m.score(X[te], y[te])))

    m = _model()
    m.fit(X, y)
    raw = m.coef_[0].astype(float)
    return {
        "feature_order": list(feature_names),
        "raw_weights": [float(v) for v in raw],
        "normalised_weights": [float(v) for v in normalise_nonnegative(raw)],
        "n_pairs": int(len(X)),
        "cv_accuracy": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
    }


def _named(result: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    out = {k: w for k, w in zip(keys, result["normalised_weights"])}
    out.update({
        "raw_weights": result["raw_weights"],
        "method": "feature_augmented_bt",
        "target": "overall_quality",
        "n_pairs": result["n_pairs"],
        "cv_accuracy": result["cv_accuracy"],
        "cv_accuracy_std": result["cv_accuracy_std"],
        "feature_order": result["feature_order"],
    })
    return out


def train_stage1(df, pairs, sample_to_idx) -> Dict[str, Any]:
    """Stage 1: weights over [S_fact†, S_rel, S_coh, S_flu] -> S_text†."""
    result = fit([df[c].tolist() for c in STAGE1_FEATURES], STAGE1_FEATURES, pairs, sample_to_idx)
    return _named(result, STAGE1_KEYS)


def train_stage2(df, pairs, sample_to_idx) -> Dict[str, Any]:
    """Stage 2: weights over [S_text†, S_relevance, S_div] -> PrefCal-MM."""
    result = fit([df[c].tolist() for c in STAGE2_FEATURES], STAGE2_FEATURES, pairs, sample_to_idx)
    return _named(result, STAGE2_KEYS)


def save_weights(weights: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(weights, f, indent=2)

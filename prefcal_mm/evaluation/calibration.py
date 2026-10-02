"""Calibration gate.

Each sub-metric is correlated (Kendall tau-b) with the human dimension it
claims to measure. It is admitted to the composite if tau-b >= threshold
(0.10 by default); a negative tau-b means it measures its construct backwards.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..config import load_config


def gate_status(tau: float, threshold: float) -> str:
    if tau != tau:  # NaN
        return "NO_DATA"
    if tau < 0:
        return "FAIL_NEGATIVE"
    if tau < threshold:
        return "FAIL"
    return "PASS"


def run_gate(
    merged_df,
    dimensions: Optional[Dict[str, str]] = None,
    threshold: Optional[float] = None,
):
    """Gate report for every ``{score_column: human_column}`` pair present in ``merged_df``.

    ``merged_df`` holds automatic scores and human ratings side by side, one
    row per (article, system).
    """
    import pandas as pd
    from scipy.stats import kendalltau

    cfg = load_config().get("calibration", {})
    dimensions = dimensions or cfg.get("dimensions", {})
    threshold = float(cfg.get("threshold", 0.10) if threshold is None else threshold)

    rows = []
    for score_col, human_col in dimensions.items():
        if score_col not in merged_df.columns or human_col not in merged_df.columns:
            continue
        sub = merged_df[[score_col, human_col]].dropna()
        if len(sub) < 3:
            tau, p = float("nan"), float("nan")
        else:
            tau, p = kendalltau(sub[score_col], sub[human_col])
            tau, p = float(tau), float(p)
        rows.append({
            "sub_metric": score_col,
            "human_dimension": human_col,
            "n": int(len(sub)),
            "kendall_tau_b": tau,
            "p_value": p,
            "threshold": threshold,
            "status": gate_status(tau, threshold),
        })
    return pd.DataFrame(rows)

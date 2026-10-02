"""Segment- and system-level agreement with a human rating."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def bootstrap_ci(stat_fn, x, y, n: int, seed: int = 42) -> Tuple[float, float]:
    """Percentile 95% CI of ``stat_fn(x, y)`` over ``n`` row resamples."""
    rng = np.random.default_rng(seed)
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    samples: List[float] = []
    N = len(x_arr)
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        try:
            s = stat_fn(x_arr[idx], y_arr[idx])
            if s == s:
                samples.append(float(s))
        except Exception:
            continue
    if not samples:
        return float("nan"), float("nan")
    arr = np.asarray(samples)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def correlate(
    df,
    score_col: str,
    target: str = "overall_quality",
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Kendall tau-b (with bootstrap CI), Spearman rho and Pearson r at segment
    level, and tau-b / r over per-system means at system level.

    Rows with a missing score or target are dropped, so metrics that depend on
    diversity are evaluated on the multi-image support only.
    """
    from scipy.stats import kendalltau, pearsonr, spearmanr

    sub = df[[score_col, target, "system_name"]].dropna()
    result: Dict[str, Any] = {"metric": score_col, "n_segment": int(len(sub))}
    if len(sub) < 10:
        return result

    x = sub[score_col].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    tau, _ = kendalltau(x, y)
    rho, _ = spearmanr(x, y)
    r, _ = pearsonr(x, y)
    lo, hi = bootstrap_ci(lambda a, b: kendalltau(a, b)[0], x, y, bootstrap_n, seed)

    grp = sub.groupby("system_name").agg(score=(score_col, "mean"), human=(target, "mean"))
    if len(grp) >= 3:
        tau_sys, _ = kendalltau(grp["score"].to_numpy(), grp["human"].to_numpy())
        r_sys, _ = pearsonr(grp["score"].to_numpy(), grp["human"].to_numpy())
    else:
        tau_sys = r_sys = float("nan")

    result.update({
        "kendall_tau_b": float(tau),
        "kendall_tau_b_ci_low": lo,
        "kendall_tau_b_ci_high": hi,
        "spearman_rho": float(rho),
        "pearson_r": float(r),
        "n_systems": int(len(grp)),
        "system_kendall_tau_b": float(tau_sys),
        "system_pearson_r": float(r_sys),
    })
    return result


def correlation_table(
    df,
    score_cols,
    target: str = "overall_quality",
    bootstrap_n: int = 1000,
    seed: int = 42,
):
    import pandas as pd

    rows = [correlate(df, c, target, bootstrap_n, seed) for c in score_cols if c in df.columns]
    return pd.DataFrame(rows)

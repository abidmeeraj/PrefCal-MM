"""Fixed-bound normalisation of native component ranges to [0, 1].

Bounds are read from ``configs/config.yaml::normalisation`` and are fixed
rather than data-driven, so the same transformation applies to any dataset.
"""
from __future__ import annotations

from .config import load_config


def normalise(value: float, metric_name: str) -> float:
    """Map ``value`` into [0, 1] using the configured bounds for ``metric_name``.

    Raises ValueError on an unknown metric or an out-of-range value.
    """
    bounds = load_config().get("normalisation", {})
    if metric_name not in bounds:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Expected one of: {sorted(bounds)}"
        )
    lo, hi = (float(b) for b in bounds[metric_name])
    v = float(value)
    if v < lo or v > hi:
        raise ValueError(
            f"Value {v} for metric '{metric_name}' outside declared range [{lo}, {hi}]"
        )
    span = hi - lo
    if span == 0:
        return 0.0
    return round((v - lo) / span, 6)

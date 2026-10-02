"""Composite scores from trained Bradley-Terry weights.

    S_text†    = w_fact·S_fact† + w_rel·S_rel + w_coh·S_coh + w_flu·S_flu
    PrefCal-MM = w_text·S_text† + w_relevance·S_relevance + w_diversity·S_div
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import load_config, resolve_path

STAGE1_FILE = "stage1_weights.json"
STAGE2_FILE = "stage2_weights.json"


def load_weights(weights_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load ``(stage1, stage2)`` weights; defaults to the released weights in ``weights/``."""
    d = resolve_path(weights_dir or load_config().get("paths", {}).get("weights_dir", "weights"))
    out = []
    for name in (STAGE1_FILE, STAGE2_FILE):
        p = Path(d) / name
        if not p.exists():
            raise FileNotFoundError(f"Weights file not found: {p}")
        with p.open() as f:
            out.append(json.load(f))
    return out[0], out[1]


def stext_unified(sfact_unified: float, srel: float, scoh: float, sflu: float, s1: Dict[str, Any]) -> float:
    val = (
        float(s1["w_fact"]) * float(sfact_unified)
        + float(s1["w_rel"]) * float(srel)
        + float(s1["w_coh"]) * float(scoh)
        + float(s1["w_flu"]) * float(sflu)
    )
    return round(val, 6)


def prefcal_mm(stext: float, srelevance: float, sdiversity: float, s2: Dict[str, Any]) -> float:
    """Composite score; NaN when diversity is undefined (fewer than two images)."""
    if any(math.isnan(float(v)) for v in (stext, srelevance, sdiversity)):
        return float("nan")
    return float(
        float(s2["w_text"]) * float(stext)
        + float(s2["w_relevance"]) * float(srelevance)
        + float(s2["w_diversity"]) * float(sdiversity)
    )


def add_composites(df, s1: Dict[str, Any], s2: Dict[str, Any]):
    """Return a copy of ``df`` with ``stext_unified`` and ``prefcal_mm`` columns (re)computed."""
    out = df.copy()
    out["stext_unified"] = [
        stext_unified(r["sfact_unified"], r["srel"], r["scoh"], r["sflu"], s1)
        for r in out.to_dict("records")
    ]
    out["prefcal_mm"] = [
        prefcal_mm(r["stext_unified"], r["srelevance"], r["sdiversity"], s2)
        for r in out.to_dict("records")
    ]
    return out

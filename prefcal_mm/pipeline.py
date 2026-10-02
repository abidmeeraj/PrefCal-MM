"""Score a single (article, summary) record with all pillars in one process.

For large datasets, prefer the per-pillar command-line tools
(``prefcal_mm.cli.score``), which load one judge at a time and can resume.

Example::

    from prefcal_mm.pipeline import score_record
    record = {
        "pair_id": "a1_sys", "article_id": "a1", "system_name": "sys",
        "article": {"source_text": text, "source_images": [PIL images]},
        "summary": {"text": summary, "selected_images": [PIL images]},
    }
    scores = score_record(record)   # scores["prefcal_mm"], scores["sfact_unified"], ...
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .aggregation import composite
from .pillars import diversity, factual, qualitative, relevance


def score_record(record: Dict[str, Any], weights_dir: Optional[str] = None) -> Dict[str, Any]:
    s1, s2 = composite.load_weights(weights_dir)
    out: Dict[str, Any] = {}
    for pillar in (factual, qualitative, relevance, diversity):
        out.update(pillar.score(record))
    out["stext_unified"] = composite.stext_unified(
        out["sfact_unified"], out["srel"], out["scoh"], out["sflu"], s1
    )
    out["prefcal_mm"] = composite.prefcal_mm(
        out["stext_unified"], out["srelevance"], out["sdiversity"], s2
    )
    return out

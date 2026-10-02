"""Article-local pairwise preferences from scalar human ratings."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def derive_pairs(annotations_df, target: str = "overall_quality") -> List[Dict[str, Any]]:
    """For every article and unordered system pair, prefer the higher-rated system.

    ``annotations_df`` needs ``article_id``, ``system_name`` and ``target``.
    Ties are discarded. Returns dicts with keys ``article_id``, ``system_i``,
    ``system_j`` and ``preferred`` ('i' or 'j').
    """
    pairs: List[Dict[str, Any]] = []
    for article_id, group in annotations_df.groupby("article_id"):
        systems = group.to_dict("records")
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                yi, yj = float(systems[i][target]), float(systems[j][target])
                if yi == yj:
                    continue
                pairs.append({
                    "article_id": article_id,
                    "system_i": systems[i]["system_name"],
                    "system_j": systems[j]["system_name"],
                    "preferred": "i" if yi > yj else "j",
                })
    return pairs


def build_sample_index(scores_df) -> Dict[Tuple[str, str], int]:
    """``(article_id, system_name) -> row index`` over ``scores_df`` (after reset_index)."""
    out: Dict[Tuple[str, str], int] = {}
    for idx, row in scores_df.reset_index(drop=True).iterrows():
        out[(str(row["article_id"]), str(row["system_name"]))] = int(idx)
    return out

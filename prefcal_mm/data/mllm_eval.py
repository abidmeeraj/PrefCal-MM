"""mLLM-EVAL benchmark loader (Zhuang et al., 2024).

Expected folder layout (one sub-folder per (article, system) pair)::

    <data_dir>/<article_id>_<system_name>/
        source_doc.txt   # source article text            (T_source)
        source_img/      # source article images          (V_source, used by the VAFV)
        summary.txt      # system-generated summary text  (T_gen)
        summary_img/     # images selected by the system  (V_sel, used by Pillars 2 and 3)

Human annotations are a CSV with one row per pair, keyed by
``<article_id>_<system_name>``; column names are mapped in
``configs/config.yaml::datasets.mllm_eval.annotation_columns``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ..config import load_config, resolve_path
from .images import load_image_dir

logger = logging.getLogger(__name__)

KNOWN_SYSTEMS = (
    "bart", "distilbart", "gpt2", "gpt4", "gpt4vision",
    "msmo", "pegasus", "prophetnet", "random_sample", "ref", "t5",
)
EXPECTED_ARTICLES = 142
EXPECTED_PAIRS = 1562


def _cfg() -> dict:
    return load_config().get("datasets", {}).get("mllm_eval", {}) or {}


def split_pair_id(pair_id: str) -> tuple:
    """``<article_id>_<system_name>`` -> (article_id, system_name).

    System names may contain underscores (``random_sample``), so the split is
    matched against the known system suffixes, longest first.
    """
    for system in sorted(KNOWN_SYSTEMS, key=len, reverse=True):
        suffix = f"_{system}"
        if pair_id.endswith(suffix):
            return pair_id[: -len(suffix)], system
    raise ValueError(f"Cannot split pair_id '{pair_id}' against KNOWN_SYSTEMS")


def load_annotations(path: Optional[str] = None):
    """Load the human-annotation CSV as a DataFrame.

    Output columns: ``id``, ``article_id``, ``system_name`` and one column per
    mapped human dimension (e.g. ``overall_quality``, ``consistency``).
    """
    import pandas as pd

    cfg = _cfg()
    csv_path = resolve_path(path or cfg["annotations_csv"])
    df = pd.read_csv(csv_path)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]

    mapping: Dict[str, str] = cfg.get("annotation_columns") or {}
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation CSV {csv_path} is missing columns {missing}")
    df = df.rename(columns=mapping)

    split = [split_pair_id(pid) for pid in df["id"]]
    df["article_id"] = [a for a, _ in split]
    df["system_name"] = [s for _, s in split]

    if len(df) != EXPECTED_PAIRS:
        logger.warning(
            "Annotation CSV has %d rows; the full benchmark has %d.", len(df), EXPECTED_PAIRS
        )
    return df


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace").strip()


def load_pair(folder: Path, *, load_images: bool = True) -> Dict[str, Any]:
    article_id, system_name = split_pair_id(folder.name)
    return {
        "pair_id": folder.name,
        "article_id": article_id,
        "system_name": system_name,
        "article": {
            "article_id": article_id,
            "source_text": _read_text(folder / "source_doc.txt"),
            "source_images": load_image_dir(folder / "source_img") if load_images else [],
        },
        "summary": {
            "system_name": system_name,
            "text": _read_text(folder / "summary.txt"),
            "selected_images": load_image_dir(folder / "summary_img") if load_images else [],
        },
    }


def iter_records(
    data_dir: Optional[str] = None,
    *,
    load_images: bool = True,
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield one record per (article, system) pair, in sorted folder order."""
    base = resolve_path(data_dir or _cfg()["data_dir"])
    if not base.is_dir():
        raise FileNotFoundError(
            f"mLLM-EVAL data not found at {base}. Set datasets.mllm_eval.data_dir in "
            f"configs/config.yaml or pass --data-dir."
        )

    article_ids = set()
    n = 0
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            rec = load_pair(folder, load_images=load_images)
        except ValueError as e:
            logger.warning("Skipping folder %s: %s", folder, e)
            continue
        article_ids.add(rec["article_id"])
        yield rec
        n += 1
        if max_records is not None and n >= max_records:
            return

    if len(article_ids) != EXPECTED_ARTICLES:
        logger.warning(
            "Found %d articles under %s; the full benchmark has %d.",
            len(article_ids), base, EXPECTED_ARTICLES,
        )

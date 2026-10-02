"""Dataset dispatch: every adapter yields the same record shape.

Record::

    {
      "pair_id": "<article_id>_<system_name>", "article_id": str, "system_name": str,
      "article": {"article_id": str, "source_text": str, "source_images": [PIL.Image]},
      "summary": {"system_name": str, "text": str, "selected_images": [PIL.Image]},
    }
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..config import load_config


def dataset_config(name: str, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the config block for a CSV dataset.

    ``name`` selects a block under ``configs/config.yaml::datasets``; an
    explicit ``csv_path`` overrides that block's ``csv_path``.
    """
    datasets = load_config().get("datasets", {}) or {}
    if name not in datasets:
        raise ValueError(
            f"Unknown dataset '{name}'. Define it under datasets in configs/config.yaml "
            f"(known: {sorted(datasets)})."
        )
    cfg = dict(datasets[name])
    if csv_path:
        cfg["csv_path"] = csv_path
    return cfg


def iter_records(
    dataset: str,
    *,
    csv_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    max_records: Optional[int] = None,
    load_images: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield records from ``mllm_eval`` or from a CSV dataset defined in the config."""
    if dataset == "mllm_eval":
        from . import mllm_eval

        yield from mllm_eval.iter_records(
            data_dir, load_images=load_images, max_records=max_records
        )
        return

    from . import csv_dataset

    yield from csv_dataset.iter_records(
        dataset_config(dataset, csv_path),
        load_images=load_images,
        max_records=max_records,
    )

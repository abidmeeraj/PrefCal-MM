"""Shared helpers for the command-line tools."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import load_config, resolve_path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", default="mllm_eval",
        help="'mllm_eval' or the name of a CSV dataset block in configs/config.yaml (default: mllm_eval)",
    )
    parser.add_argument("--data-dir", default=None, help="mLLM-EVAL folder (overrides the config)")
    parser.add_argument(
        "--csv", default=None,
        help="CSV of summaries; uses the column mapping of --dataset (default block: custom_csv)",
    )


def resolve_dataset(args) -> str:
    """``--csv`` without an explicit CSV dataset falls back to the custom_csv template."""
    if args.csv and args.dataset == "mllm_eval":
        return "custom_csv"
    return args.dataset


def dataset_dir(dataset: str) -> Path:
    root = resolve_path(load_config().get("paths", {}).get("outputs_dir", "outputs"))
    return root / dataset


def load_human(dataset: str, path: Optional[str]):
    """Human ratings with ``pair_id``, ``article_id``, ``system_name`` and one column per dimension.

    mLLM-EVAL uses the configured column mapping. Any other CSV must already
    provide ``article_id`` and ``system_name`` (or ``pair_id``) plus the
    rating columns, e.g. ``overall_quality``.
    """
    import pandas as pd

    if dataset == "mllm_eval":
        from ..data import mllm_eval

        df = mllm_eval.load_annotations(path)
        return df.rename(columns={"id": "pair_id"})

    if not path:
        raise ValueError("--annotations is required for datasets other than mllm_eval")
    df = pd.read_csv(resolve_path(path))
    if "pair_id" not in df.columns:
        if not {"article_id", "system_name"} <= set(df.columns):
            raise ValueError("Annotation CSV needs 'pair_id' or both 'article_id' and 'system_name'")
        df["pair_id"] = df["article_id"].astype(str) + "_" + df["system_name"].astype(str)
    return df

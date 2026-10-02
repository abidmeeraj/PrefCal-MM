"""Join the four pillar CSVs and compute S_text† and the PrefCal-MM score.

Records are inner-joined on ``pair_id``; records missing from any pillar are
dropped with a warning. ``prefcal_mm`` is NaN for records with fewer than two
selected images, where diversity is undefined.

Example::

    python -m prefcal_mm.cli.consolidate --dataset mllm_eval
    python -m prefcal_mm.cli.consolidate --weights-dir outputs/mllm_eval/weights
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ..aggregation import composite
from ..config import resolve_path
from ..pillars import diversity, factual, qualitative, relevance
from ._common import dataset_dir, setup_logging

logger = logging.getLogger(__name__)

PILLAR_COLUMNS = {
    "factual": [c for c in factual.SCORE_COLUMNS if c != "atomic_facts_json"],
    "qualitative": list(qualitative.SCORE_COLUMNS),
    "relevance": ["srelevance", "raw_srelevance", "n_selected_images"],
    "diversity": ["sdiversity", "raw_diversity"],
}

OUTPUT_COLUMNS = [
    "pair_id", "article_id", "system_name",
    "prefcal_mm", "stext_unified",
    "sfact_unified", "srel", "scoh", "sflu",
    "srelevance", "sdiversity",
    "raw_srelevance", "raw_diversity", "n_selected_images",
    "n_atomic_facts", "n_supported_text_only", "n_supported_image_only",
    "n_supported_both", "n_supported_neither", "n_unk", "n_supported_total",
]


def consolidate(pillar_csvs, weights_dir: Optional[str] = None):
    import pandas as pd

    merged = None
    for pillar, cols in PILLAR_COLUMNS.items():
        df = pd.read_csv(pillar_csvs[pillar])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{pillar_csvs[pillar]} is missing columns {missing}")
        keep = (["pair_id", "article_id", "system_name"] if merged is None else ["pair_id"]) + cols
        df = df[keep]
        logger.info("[%s] %d rows from %s", pillar, len(df), pillar_csvs[pillar])
        merged = df if merged is None else merged.merge(df, on="pair_id", how="inner")
    logger.info("Joined %d records across all pillars", len(merged))

    s1, s2 = composite.load_weights(weights_dir)
    return composite.add_composites(merged, s1, s2)[OUTPUT_COLUMNS]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prefcal_mm.cli.consolidate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mllm_eval")
    for p in PILLAR_COLUMNS:
        parser.add_argument(f"--{p}-csv", default=None,
                            help=f"{p} pillar CSV (default: outputs/<dataset>/pillars/{p}.csv)")
    parser.add_argument("--weights-dir", default=None, help="BT weights (default: released weights/)")
    parser.add_argument("--output", default=None, help="default: outputs/<dataset>/scores.csv")
    args = parser.parse_args(argv)
    setup_logging()

    pillar_dir = dataset_dir(args.dataset) / "pillars"
    csvs = {}
    for p in PILLAR_COLUMNS:
        given = getattr(args, f"{p}_csv")
        path = resolve_path(given) if given else pillar_dir / f"{p}.csv"
        if not path.exists():
            parser.error(f"{p} pillar CSV not found: {path}")
        csvs[p] = path

    out = consolidate(csvs, args.weights_dir)
    out_csv = resolve_path(args.output) if args.output else dataset_dir(args.dataset) / "scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows (%d with a PrefCal-MM score) -> %s",
                len(out), int(out["prefcal_mm"].notna().sum()), out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

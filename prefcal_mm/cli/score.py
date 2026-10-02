"""Score one pillar over a dataset and write a resumable CSV.

Rows already present in the output CSV are skipped, so an interrupted run
can simply be restarted. A record that raises is written as a NaN row with
the error message and the run continues.

Examples::

    python -m prefcal_mm.cli.score --pillar factual
    python -m prefcal_mm.cli.score --pillar diversity --formula tce
    python -m prefcal_mm.cli.score --pillar relevance --csv data/custom/summaries.csv
    python -m prefcal_mm.cli.score --pillar qualitative --max 5          # smoke test
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import traceback
from pathlib import Path
from typing import List, Optional

from ..config import resolve_path, set_deterministic_mode
from ..data.registry import iter_records
from ..pillars import diversity, factual, qualitative, relevance
from ._common import add_dataset_args, dataset_dir, resolve_dataset, setup_logging

logger = logging.getLogger(__name__)

# name -> (module, primary model role, needs images)
PILLARS = {
    "factual": (factual, "factual_consistency", True),
    "qualitative": (qualitative, "qualitative", False),
    "relevance": (relevance, "image_text_relevance", True),
    "diversity": (diversity, "visual_diversity", True),
}


def _short(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.split("/")[-1]).strip("-")


def default_output(dataset: str, pillar: str, model: Optional[str], formula: Optional[str]) -> Path:
    name = pillar
    if model:
        name += f"__{_short(model)}"
    if formula:
        name += f"__{formula}"
    return dataset_dir(dataset) / "pillars" / f"{name}.csv"


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", newline="") as f:
        return {row["pair_id"] for row in csv.DictReader(f) if row.get("pair_id")}


def run(args) -> Path:
    module, role, needs_images = PILLARS[args.pillar]
    set_deterministic_mode()

    # Model overrides must be in place before the first model is loaded.
    if args.model:
        os.environ[f"PREFCAL_MODEL_{role.upper()}"] = args.model
    if args.vafv_model:
        os.environ["PREFCAL_MODEL_VAFV"] = args.vafv_model
    if args.formula:
        os.environ["PREFCAL_DIVERSITY_FORMULA"] = args.formula

    dataset = resolve_dataset(args)
    out_csv = (
        resolve_path(args.output) if args.output
        else default_output(dataset, args.pillar, args.model, args.formula)
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pair_id", "article_id", "system_name", *module.SCORE_COLUMNS, "error"]

    done = _done_ids(out_csv)
    if done:
        logger.info("Resuming: %d records already scored in %s", len(done), out_csv)

    n_new = n_err = 0
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        records = iter_records(
            dataset, csv_path=args.csv, data_dir=args.data_dir,
            max_records=args.max, load_images=needs_images,
        )
        for rec in records:
            if rec["pair_id"] in done:
                continue
            row = {"pair_id": rec["pair_id"], "article_id": rec["article_id"],
                   "system_name": rec["system_name"], "error": ""}
            try:
                result = module.score(rec)
                row.update({c: result.get(c, float("nan")) for c in module.SCORE_COLUMNS})
            except Exception as e:  # noqa: BLE001 - one bad record must not stop the run
                logger.error("[%s] %s failed: %s\n%s", args.pillar, rec["pair_id"], e, traceback.format_exc())
                row.update({c: float("nan") for c in module.SCORE_COLUMNS})
                row["error"] = str(e)
                n_err += 1
            writer.writerow(row)
            f.flush()
            n_new += 1
            if n_new % 25 == 0:
                logger.info("[%s] %d records scored (%d errors)", args.pillar, n_new, n_err)

    logger.info("[%s] done: %d new rows (%d errors) -> %s", args.pillar, n_new, n_err, out_csv)
    return out_csv


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prefcal_mm.cli.score", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pillar", required=True, choices=sorted(PILLARS))
    add_dataset_args(parser)
    parser.add_argument("--model", default=None,
                        help="override the pillar's model (factual: decomposition + Stage A)")
    parser.add_argument("--vafv-model", default=None, help="override the Stage-B visual verifier")
    parser.add_argument("--formula", default=None, choices=diversity.FORMULAS,
                        help="diversity formula (default from configs/config.yaml)")
    parser.add_argument("--max", type=int, default=None, help="score at most N records")
    parser.add_argument("--output", default=None,
                        help="output CSV (default: outputs/<dataset>/pillars/<pillar>.csv)")
    args = parser.parse_args(argv)
    setup_logging()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

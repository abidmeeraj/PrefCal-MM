"""Agreement of PrefCal-MM and its components with human ratings.

Recomputes S_text† and PrefCal-MM with the given weights, then reports
segment-level Kendall tau-b (bootstrap 95% CI), Spearman rho and Pearson r,
system-level tau-b and r over per-system means, and the calibration gate.

Example::

    python -m prefcal_mm.cli.evaluate --dataset mllm_eval
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from ..aggregation import composite
from ..config import load_config, resolve_path
from ..evaluation import calibration, correlation
from ._common import dataset_dir, load_human, setup_logging

logger = logging.getLogger(__name__)

METRICS = ["prefcal_mm", "stext_unified", "sfact_unified", "srel", "scoh", "sflu", "srelevance", "sdiversity"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prefcal_mm.cli.evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mllm_eval")
    parser.add_argument("--scores", default=None, help="consolidated scores (default: outputs/<dataset>/scores.csv)")
    parser.add_argument("--annotations", default=None, help="human ratings CSV (mllm_eval: from the config)")
    parser.add_argument("--weights-dir", default=None, help="BT weights (default: released weights/)")
    parser.add_argument("--target", default=None, help="human rating (default: overall_quality)")
    parser.add_argument("--out-dir", default=None, help="default: outputs/<dataset>/evaluation")
    args = parser.parse_args(argv)
    setup_logging()

    import pandas as pd

    cfg = load_config()
    eval_cfg = cfg.get("evaluation", {})
    target = args.target or eval_cfg.get("target", "overall_quality")
    boot_n = int(eval_cfg.get("bootstrap_n", 1000))
    seed = int(cfg.get("runtime", {}).get("seed", 42))

    scores_csv = resolve_path(args.scores) if args.scores else dataset_dir(args.dataset) / "scores.csv"
    out_dir = resolve_path(args.out_dir) if args.out_dir else dataset_dir(args.dataset) / "evaluation"

    s1, s2 = composite.load_weights(args.weights_dir)
    scores = composite.add_composites(pd.read_csv(scores_csv), s1, s2)
    human = load_human(args.dataset, args.annotations)
    human_cols = [c for c in human.columns if c not in ("article_id", "system_name")]
    merged = scores.merge(human[human_cols], on="pair_id", how="inner")

    table = correlation.correlation_table(merged, METRICS, target, boot_n, seed)
    gate = calibration.run_gate(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "correlation.csv", index=False)
    gate.to_csv(out_dir / "calibration_gate.csv", index=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\nAgreement with human {target}\n" + table.to_string(index=False))
    print("\nCalibration gate\n" + gate.to_string(index=False))
    print(f"\nResults written to {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit the two Bradley-Terry stages on human Overall Quality preferences.

Steps:
  1. Calibration gate: each sub-metric against its matched human dimension.
  2. Preference pairs per article from the target rating (ties discarded);
     only records with all features defined (incl. diversity) are used.
  3. Stage 1 over [S_fact†, S_rel, S_coh, S_flu].
  4. S_text† recomputed with the new Stage-1 weights.
  5. Stage 2 over [S_text†, S_relevance, S_div].

Example::

    python -m prefcal_mm.cli.train --dataset mllm_eval --scores outputs/mllm_eval/scores.csv
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from ..aggregation import bradley_terry as bt
from ..aggregation import composite, preferences
from ..config import load_config, resolve_path
from ..evaluation import calibration
from ._common import dataset_dir, load_human, setup_logging

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prefcal_mm.cli.train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mllm_eval")
    parser.add_argument("--scores", default=None, help="consolidated scores (default: outputs/<dataset>/scores.csv)")
    parser.add_argument("--annotations", default=None, help="human ratings CSV (mllm_eval: from the config)")
    parser.add_argument("--target", default=None, help="human rating to learn from (default: overall_quality)")
    parser.add_argument("--out-dir", default=None, help="default: outputs/<dataset>/weights")
    args = parser.parse_args(argv)
    setup_logging()

    import pandas as pd

    target = args.target or load_config().get("evaluation", {}).get("target", "overall_quality")
    scores_csv = resolve_path(args.scores) if args.scores else dataset_dir(args.dataset) / "scores.csv"
    out_dir = resolve_path(args.out_dir) if args.out_dir else dataset_dir(args.dataset) / "weights"

    scores = pd.read_csv(scores_csv)
    human = load_human(args.dataset, args.annotations)
    human_cols = [c for c in human.columns if c not in ("article_id", "system_name")]
    merged = scores.merge(human[human_cols], on="pair_id", how="inner")
    logger.info("Merged %d scored records with human ratings", len(merged))

    gate = calibration.run_gate(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate.to_csv(out_dir / "calibration_gate.csv", index=False)
    print("\nCalibration gate\n" + gate.to_string(index=False))
    if (gate["status"] != "PASS").any():
        logger.warning("Some sub-metrics did not pass the calibration gate; see calibration_gate.csv")

    features = ["sfact_unified", "srel", "scoh", "sflu", "srelevance", "sdiversity", target]
    frame = merged.dropna(subset=features).reset_index(drop=True)
    pairs = preferences.derive_pairs(human[["article_id", "system_name", target]], target)
    index = preferences.build_sample_index(frame[["article_id", "system_name"]])

    s1 = bt.train_stage1(frame, pairs, index)
    frame["stext_unified"] = [
        composite.stext_unified(r["sfact_unified"], r["srel"], r["scoh"], r["sflu"], s1)
        for r in frame.to_dict("records")
    ]
    s2 = bt.train_stage2(frame, pairs, index)
    for w in (s1, s2):
        w["target"] = target

    bt.save_weights(s1, out_dir / composite.STAGE1_FILE)
    bt.save_weights(s2, out_dir / composite.STAGE2_FILE)

    print(f"\nTraining records: {len(frame)}   preference pairs used: {s1['n_pairs']}")
    print("Stage 1  " + "  ".join(f"{k}={s1[k]:.4f}" for k in bt.STAGE1_KEYS)
          + f"   CV acc {s1['cv_accuracy']:.4f} ± {s1['cv_accuracy_std']:.4f}")
    print("Stage 2  " + "  ".join(f"{k}={s2[k]:.4f}" for k in bt.STAGE2_KEYS)
          + f"   CV acc {s2['cv_accuracy']:.4f} ± {s2['cv_accuracy_std']:.4f}")
    print(f"Weights written to {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

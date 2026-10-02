"""Download the judge models into the local Hugging Face cache.

Examples::

    python -m prefcal_mm.cli.download_models                    # all `final` models
    python -m prefcal_mm.cli.download_models --role vafv,qualitative
    python -m prefcal_mm.cli.download_models --candidates       # also every candidate

Gated models (e.g. Llama, Mistral) require ``huggingface-cli login`` first.
The cache location follows the usual ``HF_HOME`` environment variable.
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from ..config import load_models_config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prefcal_mm.cli.download_models", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", default=None, help="comma-separated roles (default: all)")
    parser.add_argument("--candidates", action="store_true", help="also download candidate models")
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    cfg = load_models_config()
    roles = args.role.split(",") if args.role else list(cfg)
    model_ids = []
    for role in roles:
        if role not in cfg:
            parser.error(f"unknown role '{role}'; known: {sorted(cfg)}")
        ids = [cfg[role]["final"]] + (cfg[role].get("candidates", []) if args.candidates else [])
        model_ids += [m for m in ids if m not in model_ids]

    failed = []
    for model_id in model_ids:
        print(f"Downloading {model_id} ...", flush=True)
        try:
            snapshot_download(repo_id=model_id)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
            failed.append(model_id)
    if failed:
        print(f"\n{len(failed)} model(s) failed: {failed}")
        return 1
    print("\nAll models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

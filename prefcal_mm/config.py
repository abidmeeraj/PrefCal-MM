"""Configuration loading, path resolution and determinism.

All tunable values live in ``configs/`` at the repository root:
``config.yaml`` (hyperparameters, paths, datasets), ``models.yaml`` (judge
model identifiers) and ``prompts.yaml`` (prompt templates).
"""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configs_dir() -> Path:
    return repo_root() / "configs"


@lru_cache(maxsize=8)
def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict:
    return _load_yaml(str(configs_dir() / "config.yaml"))


def load_models_config() -> dict:
    return _load_yaml(str(configs_dir() / "models.yaml"))


def load_prompts() -> dict:
    return _load_yaml(str(configs_dir() / "prompts.yaml"))


def resolve_path(path: Union[str, Path]) -> Path:
    """Return ``path`` as absolute, interpreting relative paths against the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root() / p


def set_deterministic_mode(seed: Optional[int] = None) -> None:
    """Best-effort determinism for Python, NumPy and PyTorch.

    Bit-exact reproducibility is not guaranteed across GPU types or library
    versions because of floating-point non-associativity in CUDA kernels.
    """
    if seed is None:
        seed = int(load_config().get("runtime", {}).get("seed", 42))

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass

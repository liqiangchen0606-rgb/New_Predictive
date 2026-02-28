"""Utility helpers for reproducible churn modelling workflows."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RANDOM_STATE = 42


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Set deterministic seed across libraries used in this project."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(name: str = "churn", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a consistent formatting style."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def save_json(payload: dict[str, Any], output_path: Path | str) -> None:
    """Persist a dictionary to JSON with indentation for readability."""
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

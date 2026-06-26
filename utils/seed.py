"""Global seeding for reproducible experiments.

:func:`set_global_seed` seeds Python's ``random`` module, NumPy and — when it is
installed — PyTorch. PyTorch is optional in Phase 1.5, so its absence is handled
gracefully. This utility is intended to be called once at the start of every
future RL experiment.
"""

from __future__ import annotations

import random

import numpy as np

from utils.logger import get_logger

_logger = get_logger(__name__)


def set_global_seed(seed: int) -> int:
    """Seed all relevant random number generators.

    Seeds Python ``random`` and NumPy unconditionally, and PyTorch (including
    CUDA, if available) only when PyTorch is importable.

    Args:
        seed: The seed value to apply.

    Returns:
        The seed that was applied, for convenient logging or chaining.
    """
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed)
    _logger.info("Global seed set to %d", seed)
    return seed


def _seed_torch(seed: int) -> None:
    """Seed PyTorch if it is installed; otherwise do nothing."""
    try:
        import torch
    except ImportError:
        _logger.debug("PyTorch not installed; skipping torch seeding.")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

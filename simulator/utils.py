"""Backwards-compatibility shim for the original simulator utilities.

The utility helpers moved into the top-level :mod:`utils` package during
Phase 1.5. This module re-exports them unchanged so existing imports such as
``from simulator.utils import generate_timestamp`` keep working. New code should
import from :mod:`utils` directly.
"""

from __future__ import annotations

from utils.io import write_csv
from utils.paths import (
    TIMESTAMP_FORMAT,
    ensure_directory,
    generate_timestamp,
    make_episode_name,
)

__all__ = [
    "TIMESTAMP_FORMAT",
    "generate_timestamp",
    "ensure_directory",
    "make_episode_name",
    "write_csv",
]

"""Filesystem and naming helpers for RaceMind AI.

Canonical home for directory creation, timestamp generation and artifact
naming. These helpers depend only on the standard library so they can be reused
freely across the project without import cycles.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

# Format used for every human-readable timestamp embedded in a filename. It
# sorts lexicographically in chronological order.
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"


def generate_timestamp() -> str:
    """Return the current local time formatted as ``YYYYmmdd_HHMMSS``."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def ensure_directory(path: Path) -> Path:
    """Create ``path`` (and any missing parents) if it does not yet exist.

    Args:
        path: Directory to create.

    Returns:
        The same ``path``, so the call can be used inline.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directories(paths: Iterable[Path]) -> list[Path]:
    """Create every directory in ``paths``.

    Args:
        paths: Directories to create.

    Returns:
        The created paths as a list, in iteration order.
    """
    return [ensure_directory(path) for path in paths]


def make_episode_name(
    prefix: str,
    episode_index: int,
    timestamp: str | None = None,
) -> str:
    """Build a stable, sortable name stem for a single episode artifact.

    Args:
        prefix: Logical prefix, e.g. ``"episode"`` or ``"telemetry"``.
        episode_index: Episode counter.
        timestamp: Optional pre-generated timestamp; one is created if omitted.

    Returns:
        A filename stem such as ``"episode_003_20260625_141503"`` (no suffix).
    """
    stamp = timestamp if timestamp is not None else generate_timestamp()
    return f"{prefix}_{episode_index:03d}_{stamp}"

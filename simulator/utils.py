"""Reusable utility helpers for the RaceMind AI simulator.

This module collects small, side-effect-light helpers that are shared by the
telemetry, recorder and replay components: timestamp generation, directory
creation, CSV writing and episode naming. Keeping them here avoids duplicated
code and keeps the higher-level modules focused on their own responsibilities.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Format used for every human-readable timestamp the framework produces.
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"


def generate_timestamp() -> str:
    """Return the current local time formatted as ``YYYYmmdd_HHMMSS``.

    The format sorts lexicographically in chronological order, which makes the
    generated filenames easy to browse and sort on disk.
    """
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


def make_episode_name(
    prefix: str,
    episode_index: int,
    timestamp: str | None = None,
) -> str:
    """Build a stable, sortable name for a single episode artifact.

    Args:
        prefix: Logical prefix, e.g. ``"episode"`` or ``"telemetry"``.
        episode_index: Zero-or-one based episode counter.
        timestamp: Optional pre-generated timestamp; one is created if omitted.

    Returns:
        A filename stem such as ``"episode_003_20260625_141503"`` (no suffix).
    """
    stamp = timestamp if timestamp is not None else generate_timestamp()
    return f"{prefix}_{episode_index:03d}_{stamp}"


def _row_to_mapping(row: Any) -> Mapping[str, Any]:
    """Coerce a single CSV row into a mapping of column name to value."""
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    if isinstance(row, Mapping):
        return row
    raise TypeError(
        f"CSV rows must be dataclass instances or mappings, got {type(row)!r}"
    )


def write_csv(
    path: Path,
    rows: Iterable[Any],
    fieldnames: Sequence[str],
) -> Path:
    """Write ``rows`` to ``path`` as a CSV file with a header.

    Each row may be either a mapping or a dataclass instance; dataclasses are
    converted automatically. Parent directories are created as needed.

    Args:
        path: Destination CSV path.
        rows: Iterable of dataclass instances or mappings.
        fieldnames: Ordered column names written as the header.

    Returns:
        The path that was written.
    """
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_mapping(row))
    return path

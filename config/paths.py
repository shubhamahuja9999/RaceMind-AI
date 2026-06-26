"""Canonical filesystem locations for the RaceMind AI project.

This module is intentionally dependency-free (standard library only) so it can
be imported from anywhere — including the ``utils`` and ``config`` packages —
without risking circular imports. It defines *where* things live; the helpers
that actually create directories live in :mod:`utils.paths`.
"""

from __future__ import annotations

from pathlib import Path

# Resolve the project root relative to this file so paths are portable and
# never depend on the current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Top-level data directory and its canonical sub-directories. Keeping the names
# here as a single source of truth means the data layout is defined in exactly
# one place and consumed by the configuration objects.
DATA_DIRNAME: str = "data"

DATA_SUBDIRS: tuple[str, ...] = (
    "telemetry",
    "recordings",
    "videos",
    "plots",
    "evaluation",
    "models",
    "checkpoints",
)

# Directory where log files are written.
LOGS_DIRNAME: str = "logs"
DEFAULT_LOG_FILENAME: str = "project.log"


def data_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the root data directory for a given project root."""
    return project_root / DATA_DIRNAME


def logs_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the logs directory for a given project root."""
    return project_root / LOGS_DIRNAME

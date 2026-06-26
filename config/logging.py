"""Logging configuration for RaceMind AI.

Defines a small, immutable :class:`LoggingConfig` describing where logs go and
how they are formatted. The actual logger setup lives in :mod:`utils.logger`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.paths import DEFAULT_LOG_FILENAME, PROJECT_ROOT, logs_dir


@dataclass(frozen=True)
class LoggingConfig:
    """Immutable description of the project's logging behaviour."""

    level: str = "INFO"
    log_format: str = "%(asctime)s %(levelname)s %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_filename: str = DEFAULT_LOG_FILENAME
    console: bool = True
    project_root: Path = PROJECT_ROOT

    @property
    def log_dir(self) -> Path:
        """Directory where log files are written."""
        return logs_dir(self.project_root)

    @property
    def log_file(self) -> Path:
        """Full path to the log file."""
        return self.log_dir / self.log_filename

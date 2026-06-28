"""Lightweight git metadata helper for experiment provenance.

Used to stamp model cards and reports with the current commit. Returns ``None``
when git is unavailable or the project is not a git repository, so callers never
need to handle exceptions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from config.paths import PROJECT_ROOT


def get_git_commit(short: bool = True) -> Optional[str]:
    """Return the current git commit hash, or ``None`` if unavailable.

    Args:
        short: When ``True``, return the abbreviated hash.

    Returns:
        The commit hash string, or ``None`` if git is not available.
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    return _run_git(args, PROJECT_ROOT)


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    """Run a git command, returning stripped stdout or ``None`` on failure."""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

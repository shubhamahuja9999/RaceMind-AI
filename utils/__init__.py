"""Top-level utility package for RaceMind AI.

Canonical home for reusable, dependency-light helpers. New code should import
from here (e.g. ``from utils.paths import ensure_directory``). The lightweight
path/io helpers are re-exported for convenience.

Note: :func:`utils.logger.get_logger` and :func:`utils.seed.set_global_seed`
are intentionally *not* re-exported here. Import them from their submodules to
keep this package's import side-effect free.
"""

from utils.io import read_json, safe_write_text, write_csv, write_json
from utils.paths import (
    TIMESTAMP_FORMAT,
    ensure_directories,
    ensure_directory,
    generate_timestamp,
    make_episode_name,
)

__all__ = [
    "TIMESTAMP_FORMAT",
    "ensure_directory",
    "ensure_directories",
    "generate_timestamp",
    "make_episode_name",
    "write_csv",
    "safe_write_text",
    "write_json",
    "read_json",
]

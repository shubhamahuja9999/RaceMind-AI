"""Safe file I/O helpers for RaceMind AI.

Centralises CSV writing, atomic text writing and small JSON helpers (useful for
loading and saving experiment configuration). All writers create parent
directories as needed and write atomically where practical.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils.paths import ensure_directory


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


def safe_write_text(path: Path, text: str) -> Path:
    """Atomically write ``text`` to ``path``.

    The content is written to a temporary file in the same directory and then
    replaced into place, so a partially written file is never observed.

    Args:
        path: Destination file path.
        text: Text content to write (UTF-8).

    Returns:
        The path that was written.
    """
    ensure_directory(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def write_json(path: Path, data: Mapping[str, Any]) -> Path:
    """Serialise ``data`` to ``path`` as indented JSON (atomically).

    Non-JSON-native values (e.g. :class:`~pathlib.Path`) are stringified via
    ``default=str`` so config and metadata objects serialise without manual
    conversion.
    """
    return safe_write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str))


def read_json(path: Path) -> dict[str, Any]:
    """Load and return the JSON object stored at ``path``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    """Load and return the mapping stored in a YAML file at ``path``.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping (an empty dict for an empty file).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    # Imported locally so the rest of utils.io works without PyYAML installed.
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

"""Tiny Markdown table helper, shared by the report generators.

Kept separate so benchmark reports and experiment reports format tables
identically without duplicating the logic.
"""

from __future__ import annotations

from typing import Sequence


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table.

    Args:
        headers: Column headers.
        rows: Row values; each row is stringified cell-by-cell.

    Returns:
        The table as a Markdown string (no trailing newline).
    """
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])

"""Rendering and persistence for multi-agent benchmark comparisons.

Turns a :class:`~evaluation.benchmark_runner.BenchmarkComparison` into a
Markdown comparison table and saves both Markdown and JSON.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.benchmark_runner import BenchmarkComparison
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory
from utils.tables import markdown_table

_logger = get_logger(__name__)

_HEADERS = [
    "Agent",
    "Avg",
    "95% CI",
    "Median",
    "Std",
    "Max",
    "Min",
    "Avg Length",
    "Success %",
    "Eval (s)",
]


def comparison_to_markdown(comparison: BenchmarkComparison) -> str:
    """Render ``comparison`` as a Markdown document with a comparison table."""
    rows = [
        [
            agent.name,
            f"{agent.average_reward:.2f}",
            f"[{agent.ci95_low:.1f}, {agent.ci95_high:.1f}]",
            f"{agent.median_reward:.2f}",
            f"{agent.std_reward:.2f}",
            f"{agent.max_reward:.2f}",
            f"{agent.min_reward:.2f}",
            f"{agent.average_length:.1f}",
            f"{agent.success_rate * 100:.1f}",
            f"{agent.evaluation_seconds:.2f}",
        ]
        for agent in comparison.ranked()
    ]
    return "\n".join(
        [
            f"# Benchmark — {comparison.environment}",
            "",
            f"Success threshold: {comparison.success_threshold:.1f} | "
            f"Best agent: **{comparison.best_agent}**",
            "",
            markdown_table(_HEADERS, rows),
            "",
        ]
    )


def save_comparison(comparison: BenchmarkComparison, output_dir: Path) -> tuple[Path, Path]:
    """Save the comparison as ``benchmark.md`` and ``benchmark.json``.

    Args:
        comparison: The comparison to persist.
        output_dir: Directory to write into (created if missing).

    Returns:
        A ``(markdown_path, json_path)`` tuple.
    """
    ensure_directory(output_dir)
    md_path = output_dir / "benchmark.md"
    json_path = output_dir / "benchmark.json"
    md_path.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    write_json(json_path, comparison.to_dict())
    _logger.info("Benchmark report saved: %s", md_path)
    return md_path, json_path

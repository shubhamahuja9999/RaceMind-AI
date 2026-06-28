"""Experiment report generation for RaceMind AI.

Produces a per-experiment report in both JSON (machine-readable) and Markdown
(human-readable), saved under ``data/evaluation/<experiment>/``. The report
bundles configuration, training summary, evaluation summary, multi-seed results,
generated plot paths and a reference to the model card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ExperimentReport:
    """All data describing a completed experiment."""

    experiment_name: str
    configuration: dict[str, Any]
    training_summary: Optional[dict[str, Any]]
    evaluation_summary: dict[str, Any]
    multi_seed: Optional[dict[str, Any]] = None
    plots: Sequence[str] = field(default_factory=tuple)
    model_card_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "experiment_name": self.experiment_name,
            "configuration": self.configuration,
            "training_summary": self.training_summary,
            "evaluation_summary": self.evaluation_summary,
            "multi_seed": self.multi_seed,
            "plots": list(self.plots),
            "model_card_path": self.model_card_path,
        }


def _json_block(title: str, data: Any) -> list[str]:
    """Render a titled fenced-JSON section, or a note when data is absent."""
    if data is None:
        return [f"## {title}", "", "_Not available._", ""]
    rendered = json.dumps(data, indent=2, sort_keys=True, default=str)
    return [f"## {title}", "", "```json", rendered, "```", ""]


def report_to_markdown(report: ExperimentReport) -> str:
    """Render an :class:`ExperimentReport` as Markdown."""
    lines: list[str] = [f"# Experiment Report — {report.experiment_name}", ""]
    lines += _json_block("Configuration", report.configuration)
    lines += _json_block("Training Summary", report.training_summary)
    lines += _json_block("Evaluation Summary", report.evaluation_summary)
    lines += _json_block("Multi-Seed Evaluation", report.multi_seed)

    lines += ["## Generated Plots", ""]
    if report.plots:
        lines += [f"- `{plot}`" for plot in report.plots]
    else:
        lines.append("_No plots generated._")
    lines.append("")

    lines += ["## Model Card", ""]
    lines.append(f"`{report.model_card_path}`" if report.model_card_path else "_No model card._")
    lines.append("")
    return "\n".join(lines)


def save_report(report: ExperimentReport, output_dir: Path) -> tuple[Path, Path]:
    """Save the report as ``report.json`` and ``report.md`` in ``output_dir``.

    Args:
        report: The report to persist.
        output_dir: Directory to write into (created if missing).

    Returns:
        A ``(json_path, markdown_path)`` tuple.
    """
    ensure_directory(output_dir)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    write_json(json_path, report.to_dict())
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    _logger.info("Experiment report saved: %s", md_path)
    return json_path, md_path

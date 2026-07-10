"""Experiment specifications for controlled RaceMind AI studies.

An :class:`ExperimentSpec` captures the scientific design of a single
experiment: its hypothesis, the one independent variable being changed, the
controlled variables held identical to the baseline, and the expected outcome.
It is saved automatically so every experiment is self-documenting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ExperimentSpec:
    """The scientific design of one controlled experiment."""

    experiment_id: str
    name: str
    hypothesis: str
    independent_variable: str
    controlled_variables: dict[str, Any]
    expected_outcome: str
    baseline_name: str = "baseline_ppo_1m"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the specification as a Markdown block."""
        controlled = "\n".join(
            f"| `{key}` | {value} |" for key, value in self.controlled_variables.items()
        )
        return "\n".join(
            [
                f"## Experiment {self.experiment_id}: {self.name}",
                "",
                f"- **Hypothesis:** {self.hypothesis}",
                f"- **Independent variable:** {self.independent_variable}",
                f"- **Expected outcome:** {self.expected_outcome}",
                f"- **Baseline:** `{self.baseline_name}`",
                "",
                "**Controlled variables (held identical to baseline):**",
                "",
                "| Variable | Value |",
                "| --- | --- |",
                controlled,
                "",
            ]
        )


def save_spec(spec: ExperimentSpec, output_dir: Path) -> tuple[Path, Path]:
    """Save the specification as ``spec.json`` and ``spec.md`` in ``output_dir``.

    Args:
        spec: The experiment specification.
        output_dir: Directory to write into (created if missing).

    Returns:
        A ``(json_path, markdown_path)`` tuple.
    """
    ensure_directory(output_dir)
    json_path = output_dir / "spec.json"
    md_path = output_dir / "spec.md"
    write_json(json_path, spec.to_dict())
    md_path.write_text(spec.to_markdown(), encoding="utf-8")
    _logger.info("Experiment spec saved: %s", json_path)
    return json_path, md_path

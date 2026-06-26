"""Experiment configuration for RaceMind AI.

:class:`ExperimentConfig` captures everything needed to reproduce a single
experiment run. It composes :class:`~config.simulator.SimulatorConfig` rather
than replacing it, and carries the seed, output location, recording switches
and the (future) algorithm name. It is the object PPO/SAC/A2C runners will read
in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.paths import PROJECT_ROOT
from config.simulator import SimulatorConfig
from utils.paths import ensure_directory


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable, reproducible description of a single experiment run."""

    experiment_name: str = "default"
    seed: int = 42
    environment_name: str = "CarRacing-v3"
    future_algorithm: Optional[str] = None

    # Recording switches consumed by callbacks / loggers in later phases.
    record_video: bool = False
    record_telemetry: bool = True
    record_checkpoints: bool = False

    # Composed simulator configuration and output location.
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    project_root: Path = PROJECT_ROOT
    output_subdir: str = "experiments"

    @property
    def output_directory(self) -> Path:
        """Per-experiment output directory under ``data/experiments``."""
        return (
            self.simulator.data_dir / self.output_subdir / self.experiment_name
        )

    def ensure_output_directory(self) -> Path:
        """Create the experiment output directory if missing.

        Returns:
            The ensured output directory.
        """
        return ensure_directory(self.output_directory)

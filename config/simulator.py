"""Simulator configuration for the RaceMind AI framework.

Defines :class:`SimulatorConfig`, the immutable settings object shared across
the simulator (environment, rendering and data-storage settings). All tunable
values live here — no magic numbers, window sizes, FPS values or hardcoded
paths elsewhere in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.paths import DATA_DIRNAME, PROJECT_ROOT
from utils.paths import ensure_directories


@dataclass(frozen=True)
class SimulatorConfig:
    """Immutable configuration object shared across the framework.

    Attributes are grouped into environment, rendering and storage settings.
    Sub-directory paths are exposed as computed properties so changing
    ``project_root`` or ``data_subdir`` updates everything consistently.
    """

    # --- Environment -----------------------------------------------------
    env_name: str = "CarRacing-v3"
    continuous: bool = True
    max_episode_steps: int = 1000
    seed: Optional[int] = None

    # --- Rendering -------------------------------------------------------
    render_mode: Optional[str] = "human"
    fps: int = 50
    window_width: int = 1000
    window_height: int = 800

    # --- Storage ---------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_subdir: str = DATA_DIRNAME
    telemetry_subdir: str = "telemetry"
    recordings_subdir: str = "recordings"
    videos_subdir: str = "videos"
    plots_subdir: str = "plots"
    evaluation_subdir: str = "evaluation"
    models_subdir: str = "models"
    checkpoints_subdir: str = "checkpoints"

    # --- Naming ----------------------------------------------------------
    episode_prefix: str = "episode"
    telemetry_prefix: str = "telemetry"

    # ---------------------------------------------------------------------
    # Computed path properties
    # ---------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        """Root directory for all generated data."""
        return self.project_root / self.data_subdir

    @property
    def telemetry_dir(self) -> Path:
        """Directory where telemetry CSV files are written."""
        return self.data_dir / self.telemetry_subdir

    @property
    def recordings_dir(self) -> Path:
        """Directory where compressed episode recordings are written."""
        return self.data_dir / self.recordings_subdir

    @property
    def videos_dir(self) -> Path:
        """Directory where rendered episode videos are written."""
        return self.data_dir / self.videos_subdir

    @property
    def plots_dir(self) -> Path:
        """Directory where generated plots are written."""
        return self.data_dir / self.plots_subdir

    @property
    def evaluation_dir(self) -> Path:
        """Directory where evaluation summaries are written."""
        return self.data_dir / self.evaluation_subdir

    @property
    def models_dir(self) -> Path:
        """Directory where trained models are saved (future phases)."""
        return self.data_dir / self.models_subdir

    @property
    def checkpoints_dir(self) -> Path:
        """Directory where training checkpoints are saved (future phases)."""
        return self.data_dir / self.checkpoints_subdir

    @property
    def window_size(self) -> tuple[int, int]:
        """Convenience accessor for the replay window size."""
        return (self.window_width, self.window_height)

    @property
    def data_directories(self) -> tuple[Path, ...]:
        """All data sub-directories managed by this configuration."""
        return (
            self.telemetry_dir,
            self.recordings_dir,
            self.videos_dir,
            self.plots_dir,
            self.evaluation_dir,
            self.models_dir,
            self.checkpoints_dir,
        )

    def ensure_directories(self) -> tuple[Path, ...]:
        """Create every managed data sub-directory if missing.

        Returns:
            The directories that were ensured.
        """
        ensure_directories(self.data_directories)
        return self.data_directories


def default_config() -> SimulatorConfig:
    """Return the default simulator configuration."""
    return SimulatorConfig()

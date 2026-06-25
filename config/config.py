"""Central configuration for the RaceMind AI simulator framework.

All tunable values live here. No magic numbers, window sizes, FPS values or
filesystem paths should be hardcoded anywhere else in the codebase. Modules
receive a :class:`SimulatorConfig` instance and read everything from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Resolve the project root relative to this file so the configuration is
# portable across machines and never depends on the current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SimulatorConfig:
    """Immutable configuration object shared across the framework.

    Attributes are grouped into environment settings, rendering settings and
    storage settings. Sub-directory paths are exposed as computed properties so
    that changing ``project_root`` or ``data_subdir`` updates everything
    consistently.
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
    data_subdir: str = "data"
    telemetry_subdir: str = "telemetry"
    recordings_subdir: str = "recordings"

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
    def window_size(self) -> tuple[int, int]:
        """Convenience accessor for the replay window size."""
        return (self.window_width, self.window_height)


def default_config() -> SimulatorConfig:
    """Return the default simulator configuration.

    Provided as a factory so callers have a single, obvious entry point rather
    than constructing :class:`SimulatorConfig` directly throughout the code.
    """
    return SimulatorConfig()

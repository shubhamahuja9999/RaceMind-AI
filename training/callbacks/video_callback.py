"""Video callback interface for RaceMind AI training (Phase 2+).

Defines *when* and *where* episode videos should be recorded. The actual frame
capture and encoding will be wired in alongside training; this class only
manages the cadence and output directory. No RL logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.simulator import SimulatorConfig, default_config
from training.callbacks.base_callback import BaseCallback, Context
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


class VideoCallback(BaseCallback):
    """Marks episodes for video capture every ``frequency`` episodes."""

    def __init__(
        self,
        frequency: int = 10,
        config: Optional[SimulatorConfig] = None,
    ) -> None:
        """Initialise the callback.

        Args:
            frequency: Record video every ``frequency`` episodes.
            config: Simulator configuration providing the videos directory.
        """
        if frequency < 1:
            raise ValueError("frequency must be a positive integer.")
        self._frequency = frequency
        self._config = config or default_config()

    @property
    def video_dir(self) -> Path:
        """Directory where videos are written."""
        return self._config.videos_dir

    def should_record(self, episode: int) -> bool:
        """Return whether ``episode`` should be recorded."""
        return episode % self._frequency == 0

    def on_episode_start(self, context: Context) -> None:
        """Flag the episode for recording when it matches the cadence."""
        episode = context.get("episode")
        if isinstance(episode, int) and self.should_record(episode):
            ensure_directory(self.video_dir)
            context["record_video"] = True
            _logger.debug("Episode %d flagged for video capture.", episode)

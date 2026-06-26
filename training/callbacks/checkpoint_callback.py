"""Checkpoint callback interface for RaceMind AI training (Phase 2+).

Defines *when* and *where* checkpoints should be written. It deliberately does
not know how to serialise a model — that responsibility belongs to the future
trainer, which passes a ``save_fn`` in the context. No RL logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.simulator import SimulatorConfig, default_config
from training.callbacks.base_callback import BaseCallback, Context
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


class CheckpointCallback(BaseCallback):
    """Triggers checkpoint saves every ``frequency`` episodes."""

    def __init__(
        self,
        frequency: int = 1,
        config: Optional[SimulatorConfig] = None,
    ) -> None:
        """Initialise the callback.

        Args:
            frequency: Save a checkpoint every ``frequency`` episodes.
            config: Simulator configuration providing the checkpoints directory.
        """
        if frequency < 1:
            raise ValueError("frequency must be a positive integer.")
        self._frequency = frequency
        self._config = config or default_config()

    @property
    def checkpoint_dir(self) -> Path:
        """Directory where checkpoints are written."""
        return self._config.checkpoints_dir

    def on_episode_end(self, context: Context) -> None:
        """Invoke ``context['save_fn']`` on the configured cadence, if present."""
        episode = context.get("episode")
        if not isinstance(episode, int) or episode % self._frequency != 0:
            return

        ensure_directory(self.checkpoint_dir)
        path = self.checkpoint_dir / f"checkpoint_{episode:04d}"
        save_fn = context.get("save_fn")
        if callable(save_fn):
            save_fn(path)
            _logger.info("Checkpoint saved: %s", path)
        else:
            _logger.debug("No save_fn in context; checkpoint skipped at episode %d.", episode)

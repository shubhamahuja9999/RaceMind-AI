"""Logging callback for RaceMind AI training (Phase 2+).

Logs training lifecycle events via the project logger. Contains no RL logic.
"""

from __future__ import annotations

from training.callbacks.base_callback import BaseCallback, Context
from utils.logger import get_logger

_logger = get_logger(__name__)


class LoggingCallback(BaseCallback):
    """Emits concise log lines at episode and training boundaries."""

    def on_training_start(self, context: Context) -> None:
        _logger.info("Training started.")

    def on_episode_start(self, context: Context) -> None:
        _logger.info("Episode started: %s", context.get("episode"))

    def on_episode_end(self, context: Context) -> None:
        _logger.info(
            "Episode ended: %s (reward=%s)",
            context.get("episode"),
            context.get("episode_reward"),
        )

    def on_training_end(self, context: Context) -> None:
        _logger.info("Training finished.")

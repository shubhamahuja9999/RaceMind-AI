"""Checkpoint management for RaceMind AI training.

:class:`CheckpointManager` saves the latest checkpoint every time it is called
and tracks the best checkpoint by a monitored metric (higher is better). It is
agent-agnostic: it persists via ``agent.save`` / ``agent.load`` and stores the
concrete paths in a small JSON state file, so it never needs to know an agent's
file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from utils.io import read_json, write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

LATEST_STEM: str = "latest"
BEST_STEM: str = "best"
STATE_FILENAME: str = "checkpoint_state.json"


class CheckpointManager:
    """Saves latest/best checkpoints and restores them on request."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialise the manager.

        Args:
            checkpoint_dir: Directory in which checkpoints are stored.
        """
        self._dir = ensure_directory(checkpoint_dir)
        self._state_path = self._dir / STATE_FILENAME
        self._state = self._load_state()

    @property
    def best_metric(self) -> Optional[float]:
        """The best monitored metric seen so far, or ``None``."""
        return self._state.get("best_metric")

    def _load_state(self) -> dict:
        """Load the on-disk checkpoint state, or return an empty state."""
        if self._state_path.exists():
            return read_json(self._state_path)
        return {}

    def save(self, agent: BaseAgent, metric: float, step: int) -> None:
        """Save the latest checkpoint and update the best one if improved.

        Args:
            agent: The agent to checkpoint.
            metric: The monitored metric for this checkpoint (higher is better).
            step: The training timestep this checkpoint corresponds to.
        """
        latest_path = agent.save(self._dir / LATEST_STEM)
        self._state.update(
            {"latest_path": str(latest_path), "latest_step": step, "latest_metric": metric}
        )
        _logger.info("Checkpoint saved (latest) at step %d: metric=%.2f", step, metric)

        if self.best_metric is None or metric > self.best_metric:
            best_path = agent.save(self._dir / BEST_STEM)
            self._state.update(
                {"best_path": str(best_path), "best_step": step, "best_metric": metric}
            )
            _logger.info("New best checkpoint at step %d: metric=%.2f", step, metric)

        write_json(self._state_path, self._state)

    def has_latest(self) -> bool:
        """Return whether a latest checkpoint is available to load."""
        path = self._state.get("latest_path")
        return bool(path) and Path(path).exists()

    def has_best(self) -> bool:
        """Return whether a best checkpoint is available to load."""
        path = self._state.get("best_path")
        return bool(path) and Path(path).exists()

    def load_latest(self, agent: BaseAgent) -> BaseAgent:
        """Restore the latest checkpoint into ``agent``."""
        return self._load(agent, "latest_path", "latest")

    def load_best(self, agent: BaseAgent) -> BaseAgent:
        """Restore the best checkpoint into ``agent``."""
        return self._load(agent, "best_path", "best")

    def _load(self, agent: BaseAgent, key: str, label: str) -> BaseAgent:
        """Restore a checkpoint referenced by ``key`` in the state file."""
        path = self._state.get(key)
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"No {label} checkpoint found in {self._dir}.")
        _logger.info("Loading %s checkpoint: %s", label, path)
        return agent.load(Path(path))

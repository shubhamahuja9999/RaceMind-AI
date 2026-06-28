"""Random baseline agent for RaceMind AI.

:class:`RandomAgent` samples uniformly from the action space. It is useful as a
sanity check, as a lower bound for evaluation, and to exercise the training and
evaluation framework without any learning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


class RandomAgent(BaseAgent):
    """An agent that selects uniformly random valid actions."""

    def __init__(self, action_space: gym.spaces.Space) -> None:
        """Initialise the agent.

        Args:
            action_space: The environment action space to sample from.
        """
        self._action_space = action_space

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[Any]]:
        """Return a random action; ``deterministic`` is ignored by design."""
        return np.asarray(self._action_space.sample()), None

    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
    ) -> "RandomAgent":
        """No-op: the random agent does not learn."""
        _logger.debug("RandomAgent.learn called (no-op) for %d steps.", total_timesteps)
        return self

    def save(self, path: Path) -> Path:
        """Write a small JSON marker describing the agent."""
        target = path.with_suffix(".json")
        ensure_directory(target.parent)
        target.write_text(json.dumps({"agent": self.name}), encoding="utf-8")
        return target

    def load(self, path: Path) -> "RandomAgent":
        """No state to restore; returns ``self`` unchanged."""
        return self

"""Common agent interface for RaceMind AI (Phase 2).

:class:`BaseAgent` defines the contract every agent obeys, so the training and
evaluation framework can depend on this interface alone — never on a concrete
algorithm. PPO, the random baseline and any future algorithm plug in here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class BaseAgent(ABC):
    """Abstract base class defining the agent interface.

    Concrete agents must implement :meth:`predict`, :meth:`learn`, :meth:`save`
    and :meth:`load`. :meth:`act` is provided in terms of :meth:`predict`.
    """

    @property
    def name(self) -> str:
        """The agent's class name, used in logs and summaries."""
        return type(self).__name__

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[Any]]:
        """Return an action for ``observation``.

        Args:
            observation: A single environment observation.
            deterministic: Whether to act deterministically (greedy).

        Returns:
            A ``(action, state)`` tuple, mirroring the Stable-Baselines3 API.
            ``state`` is ``None`` for non-recurrent agents.
        """

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return just the action for ``observation`` (convenience wrapper).

        Args:
            observation: A single environment observation.
            deterministic: Whether to act deterministically (greedy).

        Returns:
            The selected action.
        """
        action, _ = self.predict(observation, deterministic=deterministic)
        return action

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
    ) -> "BaseAgent":
        """Train the agent for ``total_timesteps`` environment steps.

        Agents without a learning procedure (e.g. the random baseline) implement
        this as a no-op.

        Args:
            total_timesteps: Number of environment steps to train for.
            reset_num_timesteps: Whether to reset the internal step counter;
                set to ``False`` to continue an existing schedule across chunks.

        Returns:
            ``self``, to allow chaining.
        """

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Persist the agent to ``path``.

        Args:
            path: Destination path stem (extension is implementation-defined).

        Returns:
            The concrete path that was written.
        """

    @abstractmethod
    def load(self, path: Path) -> "BaseAgent":
        """Restore the agent's state from ``path``.

        Args:
            path: Path previously returned by :meth:`save`.

        Returns:
            ``self``, with restored state.
        """

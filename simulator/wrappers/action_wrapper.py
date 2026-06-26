"""Action wrapper for RaceMind AI.

Identity transformation for now. It is the designated place to later add action
clipping or action smoothing, following the :class:`gymnasium.ActionWrapper`
convention (override :meth:`action`).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ActionWrapper(gym.ActionWrapper):
    """Pass-through action wrapper (identity)."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Return the action unchanged.

        Args:
            action: The action requested by the agent or controller.

        Returns:
            The same action. Override to clip or smooth actions.
        """
        return action

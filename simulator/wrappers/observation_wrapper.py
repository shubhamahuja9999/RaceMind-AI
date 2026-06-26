"""Observation wrapper for RaceMind AI.

Identity transformation for now. It is the designated place to later add
observation normalization, frame resizing, grayscale conversion or frame
stacking, following the :class:`gymnasium.ObservationWrapper` convention
(override :meth:`observation`).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
    """Pass-through observation wrapper (identity)."""

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Return the observation unchanged.

        Args:
            observation: The raw observation from the wrapped environment.

        Returns:
            The same observation. Override to normalize, resize or stack.
        """
        return observation

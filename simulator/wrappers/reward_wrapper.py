"""Reward wrapper for RaceMind AI.

Identity transformation for now. It is the designated place to later add reward
shaping (e.g. speed bonuses, off-track penalties), following the
:class:`gymnasium.RewardWrapper` convention (override :meth:`reward`).
"""

from __future__ import annotations

import gymnasium as gym


class RewardWrapper(gym.RewardWrapper):
    """Pass-through reward wrapper (identity)."""

    def reward(self, reward: float) -> float:
        """Return the reward unchanged.

        Args:
            reward: The raw reward from the wrapped environment.

        Returns:
            The same reward. Override to apply reward shaping.
        """
        return reward

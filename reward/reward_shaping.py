"""Environment wrapper that replaces the reward with a composed reward.

:class:`RewardShapingWrapper` sits between the environment and the training
stack. On each step it builds a :class:`RewardContext`, asks the
:class:`~reward.reward_manager.RewardManager` for the composed reward, returns
that as the step reward, and exposes the per-component breakdown in ``info`` so
it can be logged. The agent (and the simulator) receive only the final reward.

This wrapper is applied **only** for reward experiments; the default training
path does not use it, so baseline behaviour is unchanged.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from reward.base_reward import RewardContext
from reward.reward_manager import RewardManager


class RewardShapingWrapper(gym.Wrapper):
    """Replaces the environment reward with a :class:`RewardManager` composition."""

    def __init__(self, env: gym.Env, manager: RewardManager) -> None:
        """Wrap ``env`` and compose its reward with ``manager``.

        Args:
            env: The environment to wrap.
            manager: The reward manager providing the composed reward.
        """
        super().__init__(env)
        self._manager = manager
        self._prev_action: Optional[np.ndarray] = None
        self._step = 0

    def reset(self, **kwargs):
        """Reset the environment, the step counter and the manager."""
        observation, info = self.env.reset(**kwargs)
        self._prev_action = None
        self._step = 0
        self._manager.reset()
        return observation, info

    def step(self, action):
        """Step the env and return the composed reward (+ breakdown in info)."""
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        action_array = np.asarray(action, dtype=np.float32)
        context = RewardContext(
            step=self._step,
            observation=observation,
            action=action_array,
            base_reward=float(base_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            prev_action=self._prev_action,
        )
        result = self._manager.compute(context)

        enriched_info = {
            **info,
            "base_reward": float(base_reward),
            "shaped_reward": result.total,
            "reward_breakdown": result.weighted,
        }
        self._prev_action = action_array
        self._step += 1
        return observation, result.total, terminated, truncated, enriched_info

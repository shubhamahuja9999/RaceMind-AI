"""Temporal frame-stacking wrapper for RaceMind AI.

Stacks the last ``num_stack`` observations along the **channel** axis, producing
a single image observation of shape ``(H, W, C * num_stack)``. This is the
Stable-Baselines3-standard layout for image frame stacking: it keeps the
observation a valid 3-D image (so ``CnnPolicy`` accepts it) and preserves the
single-environment Gymnasium API (so the evaluator is unaffected).

Gymnasium's built-in ``FrameStackObservation`` instead adds a leading stack axis
(``(N, H, W, C)``), which is 4-D and is rejected by SB3's CNN — hence this
channel-wise variant.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class TemporalFrameStack(gym.Wrapper):
    """Channel-wise temporal frame stacking (``(H, W, C)`` -> ``(H, W, C*N)``)."""

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        """Wrap ``env`` to return the last ``num_stack`` frames stacked on channels.

        Args:
            env: The environment to wrap; its observation must be a 3-D image.
            num_stack: Number of consecutive frames to stack (e.g. 4).
        """
        super().__init__(env)
        if num_stack < 1:
            raise ValueError("num_stack must be a positive integer.")
        space = env.observation_space
        if not isinstance(space, Box) or len(space.shape) != 3:
            raise ValueError("TemporalFrameStack requires a 3-D image observation space.")

        self._num_stack = num_stack
        self._frames: deque[np.ndarray] = deque(maxlen=num_stack)

        height, width, channels = space.shape
        self.observation_space = Box(
            low=np.min(space.low),
            high=np.max(space.high),
            shape=(height, width, channels * num_stack),
            dtype=space.dtype,
        )

    def _stacked(self) -> np.ndarray:
        """Concatenate the buffered frames along the channel axis."""
        return np.concatenate(list(self._frames), axis=-1)

    def reset(self, **kwargs):
        """Reset the env and prime the buffer with the initial frame."""
        observation, info = self.env.reset(**kwargs)
        for _ in range(self._num_stack):
            self._frames.append(observation)
        return self._stacked(), info

    def step(self, action):
        """Step the env and return the newest stack of frames."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(observation)
        return self._stacked(), reward, terminated, truncated, info

"""Base wrapper for RaceMind AI environment wrappers.

:class:`BaseWrapper` is a thin, transparent :class:`gymnasium.Wrapper` that all
project wrappers (or full-control wrappers needing both observation and reward
access) can extend. By default it forwards everything unchanged, so subclassing
it and overriding nothing is a valid identity wrapper.
"""

from __future__ import annotations

import gymnasium as gym


class BaseWrapper(gym.Wrapper):
    """Transparent base wrapper following Gymnasium conventions."""

    def __init__(self, env: gym.Env) -> None:
        """Wrap ``env`` without modifying its behaviour.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

    @property
    def name(self) -> str:
        """The wrapper's class name, useful for logging the wrapper stack."""
        return type(self).__name__

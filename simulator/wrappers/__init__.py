"""Environment wrappers for RaceMind AI.

Exposes the project's Gymnasium-convention wrappers and :func:`wrap_environment`,
which applies the standard wrapper stack. All wrappers are identity transforms
in Phase 1.5; the stack is the single place to enable real transformations
later without touching call sites.
"""

from __future__ import annotations

import gymnasium as gym

from simulator.wrappers.action_wrapper import ActionWrapper
from simulator.wrappers.base_wrapper import BaseWrapper
from simulator.wrappers.observation_wrapper import ObservationWrapper
from simulator.wrappers.reward_wrapper import RewardWrapper

__all__ = [
    "BaseWrapper",
    "ObservationWrapper",
    "RewardWrapper",
    "ActionWrapper",
    "wrap_environment",
]


def wrap_environment(env: gym.Env) -> gym.Env:
    """Apply the standard RaceMind wrapper stack to ``env``.

    The wrappers are identity transforms today, so the returned environment
    behaves identically to the input. Enabling real transformations later is a
    localized change here.

    Args:
        env: The raw Gymnasium environment to wrap.

    Returns:
        The wrapped environment.
    """
    env = ObservationWrapper(env)
    env = RewardWrapper(env)
    env = ActionWrapper(env)
    return env

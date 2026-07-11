"""Environment wrappers for RaceMind AI.

Exposes the project's Gymnasium-convention wrappers and :func:`wrap_environment`,
which applies the standard wrapper stack required by Stable-Baselines3.

The Phase 3 wrapper stack is::

    raw env
    → Monitor (logs episode returns/lengths — replaces RecordEpisodeStatistics)
    → (optional) GrayscaleObservation
    → (optional) FrameStack
    → ready for PPO

All custom wrappers (observation, reward, action) are applied *before* the
SB3-standard layers so that domain-specific transforms happen first.

Note: SB3's ``Monitor`` already records episode statistics (return, length),
so Gymnasium's ``RecordEpisodeStatistics`` is intentionally omitted to avoid
a duplicate-key assertion error in the ``info`` dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.monitor import Monitor

from simulator.wrappers.action_wrapper import ActionWrapper
from simulator.wrappers.base_wrapper import BaseWrapper
from simulator.wrappers.frame_stack import TemporalFrameStack
from simulator.wrappers.observation_wrapper import ObservationWrapper
from simulator.wrappers.reward_wrapper import RewardWrapper

__all__ = [
    "BaseWrapper",
    "ObservationWrapper",
    "RewardWrapper",
    "ActionWrapper",
    "TemporalFrameStack",
    "wrap_environment",
]


def wrap_environment(
    env: gym.Env,
    monitor_dir: Optional[Path] = None,
    grayscale: bool = False,
    frame_stack: int = 0,
    reward_manager: Optional["RewardManager"] = None,
) -> gym.Env:
    """Apply the standard RaceMind wrapper stack to ``env``.

    The wrapper ordering follows the Phase 3 spec for SB3 compatibility::

        raw env
        → ObservationWrapper (custom observation transforms)
        → RewardWrapper      (custom reward shaping)
        → ActionWrapper       (custom action transforms)
        → RewardShapingWrapper (optional, composed reward from a RewardManager)
        → Monitor             (SB3 episode logging + statistics)
        → GrayscaleObservation (optional, reduces channels 3→1)
        → TemporalFrameStack   (optional, stacks N frames on the channel axis)

    Args:
        env: The raw Gymnasium environment to wrap.
        monitor_dir: Optional directory for Monitor CSV logs. When ``None``,
            Monitor still wraps the env but does not write files.
        grayscale: When ``True``, convert RGB observations to grayscale.
        frame_stack: When > 0, stack this many consecutive frames.
        reward_manager: Optional reward manager; when provided, the environment
            reward is replaced by the composed reward (before Monitor, so logged
            returns reflect the shaped reward). ``None`` keeps the native reward,
            i.e. baseline behaviour.

    Returns:
        The fully wrapped environment, ready for PPO training.
    """
    # --- Custom domain transforms (identity by default) ---
    env = ObservationWrapper(env)
    env = RewardWrapper(env)
    env = ActionWrapper(env)

    # --- Optional composed reward (before Monitor so returns reflect it) ---
    if reward_manager is not None:
        from reward.reward_shaping import RewardShapingWrapper

        env = RewardShapingWrapper(env, reward_manager)

    # --- SB3-standard wrappers (Phase 3) ---
    env = Monitor(env, filename=str(monitor_dir) if monitor_dir else None)

    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)

    if frame_stack > 0:
        # Channel-wise stacking (H, W, C*N) keeps the observation a valid image
        # for SB3's CnnPolicy and preserves the single-env Gymnasium API.
        env = TemporalFrameStack(env, num_stack=frame_stack)

    return env

"""Episode metric computation for RaceMind AI.

Pure, dependency-light functions that turn raw per-step rewards and actions into
an :class:`EpisodeMetrics` dataclass. No environment or training code is
involved, so these utilities can score both manually driven episodes (Phase 1.5)
and future agent rollouts (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    """Summary statistics for a single episode."""

    total_reward: float
    average_reward: float
    max_reward: float
    episode_length: int
    reward_per_second: Optional[float]
    average_steering: Optional[float]
    average_throttle: Optional[float]
    average_brake: Optional[float]


def _column_mean(actions: np.ndarray, index: int) -> Optional[float]:
    """Return the mean of action column ``index``, or ``None`` if empty."""
    if actions.size == 0:
        return None
    return float(actions[:, index].mean())


def compute_episode_metrics(
    rewards: np.ndarray,
    actions: np.ndarray,
    duration_seconds: Optional[float] = None,
    fps: Optional[float] = None,
) -> EpisodeMetrics:
    """Compute :class:`EpisodeMetrics` from per-step rewards and actions.

    Args:
        rewards: 1-D array of per-step rewards.
        actions: 2-D array of shape ``(steps, 3)`` holding ``[steering, gas,
            brake]`` per step. May be empty.
        duration_seconds: Optional wall-clock episode duration. When omitted but
            ``fps`` is given, duration is derived as ``length / fps``.
        fps: Optional frames-per-second used to derive duration if needed.

    Returns:
        The computed metrics. Reward-per-second and action averages are ``None``
        when the inputs needed to compute them are unavailable.
    """
    rewards = np.asarray(rewards, dtype=np.float64).ravel()
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(0, 3) if actions.size == 0 else actions.reshape(1, -1)

    length = int(rewards.size)
    total_reward = float(rewards.sum()) if length else 0.0
    average_reward = float(rewards.mean()) if length else 0.0
    max_reward = float(rewards.max()) if length else 0.0

    duration = _resolve_duration(duration_seconds, fps, length)
    reward_per_second = total_reward / duration if duration else None

    return EpisodeMetrics(
        total_reward=total_reward,
        average_reward=average_reward,
        max_reward=max_reward,
        episode_length=length,
        reward_per_second=reward_per_second,
        average_steering=_column_mean(actions, 0),
        average_throttle=_column_mean(actions, 1),
        average_brake=_column_mean(actions, 2),
    )


def _resolve_duration(
    duration_seconds: Optional[float],
    fps: Optional[float],
    length: int,
) -> Optional[float]:
    """Resolve the episode duration in seconds from the available inputs."""
    if duration_seconds is not None and duration_seconds > 0:
        return duration_seconds
    if fps and fps > 0 and length:
        return length / fps
    return None

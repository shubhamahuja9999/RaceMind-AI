"""Multi-episode benchmark aggregation for RaceMind AI.

Turns a set of per-episode outcomes into a structured :class:`BenchmarkResult`.
This complements :mod:`evaluation.metrics` (which summarises a *single* episode)
by aggregating *across* episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from training.training_loop import EpisodeOutcome


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate statistics over a batch of evaluation episodes."""

    n_episodes: int
    average_reward: float
    max_reward: float
    min_reward: float
    reward_variance: float
    average_length: float
    success_rate: float
    success_threshold: float
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        data = asdict(self)
        data["episode_rewards"] = list(self.episode_rewards)
        data["episode_lengths"] = list(self.episode_lengths)
        return data


def aggregate_outcomes(
    outcomes: Sequence[EpisodeOutcome],
    success_threshold: float,
) -> BenchmarkResult:
    """Aggregate episode outcomes into a :class:`BenchmarkResult`.

    Args:
        outcomes: The per-episode results to aggregate (must be non-empty).
        success_threshold: Total-reward threshold at or above which an episode
            counts as a success.

    Returns:
        The aggregated benchmark result.

    Raises:
        ValueError: If ``outcomes`` is empty.
    """
    if not outcomes:
        raise ValueError("Cannot aggregate an empty set of episode outcomes.")

    rewards = np.array([o.total_reward for o in outcomes], dtype=np.float64)
    lengths = np.array([o.length for o in outcomes], dtype=np.int64)
    successes = rewards >= success_threshold

    return BenchmarkResult(
        n_episodes=len(outcomes),
        average_reward=float(rewards.mean()),
        max_reward=float(rewards.max()),
        min_reward=float(rewards.min()),
        reward_variance=float(rewards.var()),
        average_length=float(lengths.mean()),
        success_rate=float(successes.mean()),
        success_threshold=float(success_threshold),
        episode_rewards=tuple(float(r) for r in rewards),
        episode_lengths=tuple(int(length) for length in lengths),
    )

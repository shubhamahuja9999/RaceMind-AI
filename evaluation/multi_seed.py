"""Multi-seed evaluation for RaceMind AI.

Evaluates an agent across several random seeds and aggregates the per-seed mean
rewards into mean / standard-deviation / best-run / worst-run statistics. This
quantifies how reliable an agent's performance is, independent of a single
lucky or unlucky seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent
from evaluation.evaluator import Evaluator
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


@dataclass(frozen=True)
class MultiSeedSummary:
    """Aggregated statistics for an agent evaluated across multiple seeds."""

    seeds: tuple[int, ...]
    per_seed_reward: dict[int, float]
    mean: float
    std: float
    best_seed: int
    best_reward: float
    worst_seed: int
    worst_reward: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def evaluate_multi_seed(
    agent: BaseAgent,
    env: gym.Env,
    seeds: Sequence[int],
    n_episodes: int = 5,
    deterministic: bool = True,
    success_threshold: float = 900.0,
    max_steps: Optional[int] = None,
) -> MultiSeedSummary:
    """Evaluate ``agent`` once per seed and aggregate the mean rewards.

    Args:
        agent: The agent to evaluate.
        env: The evaluation environment.
        seeds: The seeds to evaluate over (must be non-empty).
        n_episodes: Episodes per seed.
        deterministic: Whether the agent acts deterministically.
        success_threshold: Reward threshold counted as a success.
        max_steps: Optional per-episode step cap.

    Returns:
        The aggregated :class:`MultiSeedSummary`.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("At least one seed is required for multi-seed evaluation.")

    per_seed: dict[int, float] = {}
    for seed in seeds:
        evaluator = Evaluator(
            env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            success_threshold=success_threshold,
            max_steps=max_steps,
            seed=seed,
        )
        per_seed[seed] = evaluator.evaluate(agent).average_reward

    rewards = np.array(list(per_seed.values()), dtype=np.float64)
    seed_list = list(per_seed.keys())
    best_index = int(rewards.argmax())
    worst_index = int(rewards.argmin())

    summary = MultiSeedSummary(
        seeds=tuple(seed_list),
        per_seed_reward=per_seed,
        mean=float(rewards.mean()),
        std=float(rewards.std()),
        best_seed=seed_list[best_index],
        best_reward=float(rewards[best_index]),
        worst_seed=seed_list[worst_index],
        worst_reward=float(rewards[worst_index]),
    )
    _logger.info(
        "Multi-seed evaluation: mean=%.2f std=%.2f best=%.2f@%d worst=%.2f@%d",
        summary.mean,
        summary.std,
        summary.best_reward,
        summary.best_seed,
        summary.worst_reward,
        summary.worst_seed,
    )
    return summary


def save_multi_seed(summary: MultiSeedSummary, output_dir: Path) -> Path:
    """Write the multi-seed summary to ``multi_seed.json`` in ``output_dir``."""
    ensure_directory(output_dir)
    path = output_dir / "multi_seed.json"
    write_json(path, summary.to_dict())
    return path

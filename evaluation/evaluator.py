"""Evaluation orchestration for RaceMind AI.

:class:`Evaluator` rolls an agent out over several episodes and aggregates the
results into a :class:`~evaluation.benchmark.BenchmarkResult`. It depends only on
the :class:`~agents.base_agent.BaseAgent` interface, so it evaluates the random
baseline and PPO identically.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym

from agents.base_agent import BaseAgent
from evaluation.benchmark import BenchmarkResult, aggregate_outcomes
from training.training_loop import run_episode
from utils.logger import get_logger

_logger = get_logger(__name__)


class Evaluator:
    """Runs multi-episode evaluations and aggregates the outcomes."""

    def __init__(
        self,
        env: gym.Env,
        n_episodes: int = 5,
        deterministic: bool = True,
        success_threshold: float = 900.0,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the evaluator.

        Args:
            env: The evaluation environment (may be ``RecordVideo``-wrapped).
            n_episodes: Number of episodes to run per evaluation.
            deterministic: Whether the agent acts deterministically.
            success_threshold: Reward threshold counted as a success.
            max_steps: Optional per-episode step cap.
            seed: Optional base seed; episode ``i`` uses ``seed + i``.
        """
        if n_episodes < 1:
            raise ValueError("n_episodes must be a positive integer.")
        self._env = env
        self._n_episodes = n_episodes
        self._deterministic = deterministic
        self._success_threshold = success_threshold
        self._max_steps = max_steps
        self._seed = seed

    def evaluate(self, agent: BaseAgent) -> BenchmarkResult:
        """Evaluate ``agent`` and return aggregated metrics.

        Args:
            agent: The agent to evaluate.

        Returns:
            The aggregated benchmark result.
        """
        outcomes = []
        for episode in range(self._n_episodes):
            episode_seed = None if self._seed is None else self._seed + episode
            outcome = run_episode(
                self._env,
                agent,
                deterministic=self._deterministic,
                max_steps=self._max_steps,
                seed=episode_seed,
            )
            outcomes.append(outcome)

        result = aggregate_outcomes(outcomes, self._success_threshold)
        _logger.info(
            "Evaluation over %d episodes: avg=%.2f max=%.2f min=%.2f success=%.0f%%",
            result.n_episodes,
            result.average_reward,
            result.max_reward,
            result.min_reward,
            result.success_rate * 100.0,
        )
        return result

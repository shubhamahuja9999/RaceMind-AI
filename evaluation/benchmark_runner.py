"""Multi-agent benchmark runner for RaceMind AI.

Evaluates several agents under identical conditions and produces a structured,
comparable result per agent. Composes over the existing
:class:`~evaluation.evaluator.Evaluator` (it does not re-implement rollouts) and
enriches the per-episode rewards with median / standard-deviation / timing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import gymnasium as gym

from agents.base_agent import BaseAgent
from evaluation.evaluator import Evaluator
from evaluation.statistics import compute_reward_statistics
from utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentBenchmark:
    """Benchmark statistics for a single agent."""

    name: str
    n_episodes: int
    average_reward: float
    max_reward: float
    min_reward: float
    median_reward: float
    std_reward: float
    ci95_low: float
    ci95_high: float
    average_length: float
    success_rate: float
    evaluation_seconds: float
    episode_rewards: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data = asdict(self)
        data["episode_rewards"] = list(self.episode_rewards)
        return data


@dataclass(frozen=True)
class BenchmarkComparison:
    """A comparison of several agents benchmarked under identical conditions."""

    environment: str
    success_threshold: float
    agents: tuple[AgentBenchmark, ...]

    @property
    def best_agent(self) -> str:
        """Name of the agent with the highest average reward."""
        return max(self.agents, key=lambda a: a.average_reward).name

    def ranked(self) -> tuple[AgentBenchmark, ...]:
        """Agents sorted by average reward, descending."""
        return tuple(sorted(self.agents, key=lambda a: a.average_reward, reverse=True))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "environment": self.environment,
            "success_threshold": self.success_threshold,
            "best_agent": self.best_agent,
            "agents": [agent.to_dict() for agent in self.ranked()],
        }


def _environment_id(env: gym.Env) -> str:
    """Best-effort environment identifier for reporting."""
    spec = getattr(env, "spec", None)
    return getattr(spec, "id", None) or "unknown"


class BenchmarkRunner:
    """Benchmarks multiple agents on one evaluation environment."""

    def __init__(
        self,
        env: gym.Env,
        n_episodes: int = 5,
        deterministic: bool = True,
        success_threshold: float = 900.0,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialise the runner.

        Args:
            env: The shared evaluation environment.
            n_episodes: Episodes per agent.
            deterministic: Whether agents act deterministically.
            success_threshold: Reward threshold counted as a success.
            max_steps: Optional per-episode step cap.
            seed: Optional base seed (shared across agents for fairness).
        """
        self._env = env
        self._success_threshold = success_threshold
        self._evaluator = Evaluator(
            env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            success_threshold=success_threshold,
            max_steps=max_steps,
            seed=seed,
        )

    def run(self, agents: Mapping[str, BaseAgent]) -> BenchmarkComparison:
        """Benchmark each named agent and return the comparison.

        Args:
            agents: Mapping of display name to agent.

        Returns:
            A :class:`BenchmarkComparison` over all agents.
        """
        benchmarks: list[AgentBenchmark] = []
        for name, agent in agents.items():
            _logger.info("Benchmarking agent: %s", name)
            start = time.perf_counter()
            result = self._evaluator.evaluate(agent)
            elapsed = time.perf_counter() - start
            benchmarks.append(self._to_benchmark(name, result, elapsed))

        return BenchmarkComparison(
            environment=_environment_id(self._env),
            success_threshold=self._success_threshold,
            agents=tuple(benchmarks),
        )

    @staticmethod
    def _to_benchmark(name: str, result: Any, elapsed: float) -> AgentBenchmark:
        """Build an :class:`AgentBenchmark` from a benchmark result and timing."""
        stats = compute_reward_statistics(result.episode_rewards)
        return AgentBenchmark(
            name=name,
            n_episodes=result.n_episodes,
            average_reward=result.average_reward,
            max_reward=result.max_reward,
            min_reward=result.min_reward,
            median_reward=stats.median,
            std_reward=stats.std,
            ci95_low=stats.ci95_low,
            ci95_high=stats.ci95_high,
            average_length=result.average_length,
            success_rate=result.success_rate,
            evaluation_seconds=elapsed,
            episode_rewards=result.episode_rewards,
        )

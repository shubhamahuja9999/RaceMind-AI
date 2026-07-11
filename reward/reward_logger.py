"""Per-component reward logging, stored alongside telemetry.

Records, for every step, the base (environment) reward, the composed total and
each component's weighted contribution, and writes them to a CSV. The columns
are one-per-component so downstream analysis and plotting are trivial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym

from agents.base_agent import BaseAgent
from reward.base_reward import RewardResult
from utils.io import write_csv
from utils.logger import get_logger

_logger = get_logger(__name__)


class RewardLogger:
    """Accumulates per-step reward breakdowns and writes them to CSV."""

    def __init__(self, component_names: Sequence[str]) -> None:
        """Initialise with the component names to log (defines the columns)."""
        self._component_names = list(component_names)
        self._rows: list[dict[str, float]] = []

    def log(self, step: int, base_reward: float, result: RewardResult) -> None:
        """Append one step's base reward, total and per-component contributions."""
        row: dict[str, float] = {
            "step": step,
            "base_reward": float(base_reward),
            "total_reward": float(result.total),
        }
        for name in self._component_names:
            row[name] = float(result.weighted.get(name, 0.0))
        self._rows.append(row)

    def log_info(self, step: int, base_reward: float, total: float, breakdown: dict) -> None:
        """Append a step from an already-computed ``info`` breakdown."""
        row: dict[str, float] = {
            "step": step,
            "base_reward": float(base_reward),
            "total_reward": float(total),
        }
        for name in self._component_names:
            row[name] = float(breakdown.get(name, 0.0))
        self._rows.append(row)

    @property
    def fieldnames(self) -> list[str]:
        """CSV column order."""
        return ["step", "base_reward", "total_reward", *self._component_names]

    def save(self, path: Path) -> Path:
        """Write the accumulated rows to ``path`` as CSV."""
        write_csv(path, self._rows, self.fieldnames)
        _logger.info("Reward log saved: %s (%d steps)", path, len(self._rows))
        return path


def collect_reward_log(
    shaped_env: gym.Env,
    agent: BaseAgent,
    component_names: Sequence[str],
    max_steps: int = 1000,
    seed: Optional[int] = None,
) -> RewardLogger:
    """Run one episode on a reward-shaped env, logging the per-component breakdown.

    Args:
        shaped_env: An environment wrapped with :class:`RewardShapingWrapper`
            (so ``info`` carries ``reward_breakdown``/``base_reward``).
        agent: The acting agent.
        component_names: Names of the enabled components (defines the columns).
        max_steps: Maximum steps to record.
        seed: Optional reset seed.

    Returns:
        A populated :class:`RewardLogger`.
    """
    logger = RewardLogger(component_names)
    observation, _ = shaped_env.reset(seed=seed)
    for step in range(max_steps):
        action = agent.act(observation, deterministic=True)
        observation, total, terminated, truncated, info = shaped_env.step(action)
        logger.log_info(step, info.get("base_reward", 0.0), total, info.get("reward_breakdown", {}))
        if terminated or truncated:
            break
    return logger

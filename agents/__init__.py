"""Agents package for RaceMind AI.

Exposes the common agent interface, the concrete agents and a small factory so
experiments can select an algorithm by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym

from agents.base_agent import BaseAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from config.rl import PPOConfig

__all__ = ["BaseAgent", "PPOAgent", "RandomAgent", "build_agent"]


def build_agent(
    algorithm: str,
    env: gym.Env,
    ppo_config: Optional[PPOConfig] = None,
    seed: Optional[int] = None,
    tensorboard_log: Optional[Path] = None,
) -> BaseAgent:
    """Create an agent for ``algorithm`` (case-insensitive).

    Args:
        algorithm: Algorithm name — ``"ppo"`` or ``"random"``.
        env: The training environment.
        ppo_config: Hyperparameters used when ``algorithm == "ppo"``.
        seed: Optional RNG seed for the algorithm.
        tensorboard_log: Optional TensorBoard log directory (PPO only).

    Returns:
        The constructed agent.

    Raises:
        ValueError: If ``algorithm`` is not supported.
    """
    algo = algorithm.lower()
    if algo == "ppo":
        return PPOAgent(
            env,
            config=ppo_config,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )
    if algo == "random":
        return RandomAgent(env.action_space)
    raise ValueError(f"Unsupported algorithm: {algorithm!r} (expected 'ppo' or 'random').")

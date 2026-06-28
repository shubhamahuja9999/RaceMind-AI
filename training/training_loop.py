"""Reusable episode rollout for RaceMind AI.

A single, agent-agnostic stepping loop used by the evaluation pipeline (and
available to any manual loop). It depends only on the
:class:`~agents.base_agent.BaseAgent` interface and the Gymnasium environment
API, so it works identically for the random baseline and PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym

from agents.base_agent import BaseAgent


@dataclass(frozen=True)
class EpisodeOutcome:
    """The result of a single rolled-out episode."""

    total_reward: float
    length: int


def run_episode(
    env: gym.Env,
    agent: BaseAgent,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> EpisodeOutcome:
    """Run one episode of ``agent`` in ``env`` and return its outcome.

    Args:
        env: A Gymnasium environment.
        agent: The acting agent.
        deterministic: Whether the agent should act deterministically.
        max_steps: Optional hard cap on steps (in addition to the env's own
            termination/truncation).
        seed: Optional reset seed for reproducibility.

    Returns:
        The episode's total reward and length.
    """
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = agent.act(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        if max_steps is not None and steps >= max_steps:
            break

    return EpisodeOutcome(total_reward=total_reward, length=steps)

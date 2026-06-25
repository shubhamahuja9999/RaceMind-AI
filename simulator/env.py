"""Thin, well-behaved wrapper around Gymnasium's ``CarRacing-v3`` environment.

The :class:`RaceEnv` class hides the details of constructing the Gymnasium
environment and exposes a small, stable surface (``reset``/``step``/``close``/
``sample_action``) that the rest of the framework — and, later, the RL training
code — can depend on. No training happens here; this is purely the simulator
boundary.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from config.config import SimulatorConfig, default_config


class RaceEnv:
    """Object-oriented wrapper over a single ``CarRacing-v3`` instance.

    The wrapper owns the underlying Gymnasium environment and forwards the
    standard five-tuple step API. It can be used as a context manager so the
    environment is always closed cleanly::

        with RaceEnv(config) as env:
            env.reset()
            env.step(env.sample_action())
    """

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        """Create the wrapped environment from a configuration.

        Args:
            config: Simulator configuration; the framework default is used when
                omitted.
        """
        self._config: SimulatorConfig = config or default_config()
        self._env: gym.Env = gym.make(
            self._config.env_name,
            render_mode=self._config.render_mode,
            continuous=self._config.continuous,
            max_episode_steps=self._config.max_episode_steps,
        )
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------
    @property
    def config(self) -> SimulatorConfig:
        """The configuration this environment was built with."""
        return self._config

    @property
    def env(self) -> gym.Env:
        """Direct access to the underlying Gymnasium environment."""
        return self._env

    @property
    def action_space(self) -> gym.spaces.Space:
        """The action space of the wrapped environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        """The observation space of the wrapped environment."""
        return self._env.observation_space

    @property
    def step_count(self) -> int:
        """Number of steps taken since the last :meth:`reset`."""
        return self._step_count

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and the internal step counter.

        Args:
            seed: Optional RNG seed; falls back to the configured seed.

        Returns:
            The initial observation and the Gymnasium ``info`` dict.
        """
        self._step_count = 0
        effective_seed = seed if seed is not None else self._config.seed
        return self._env.reset(seed=effective_seed)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one action.

        Args:
            action: An action compatible with :attr:`action_space`.

        Returns:
            The standard ``(observation, reward, terminated, truncated, info)``
            tuple from Gymnasium.
        """
        self._step_count += 1
        observation, reward, terminated, truncated, info = self._env.step(action)
        return observation, float(reward), bool(terminated), bool(truncated), info

    def sample_action(self) -> np.ndarray:
        """Return a uniformly random action from the action space."""
        return self.action_space.sample()

    def render(self) -> Any:
        """Render the environment using its configured render mode."""
        return self._env.render()

    def close(self) -> None:
        """Release all resources held by the underlying environment."""
        self._env.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "RaceEnv":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()


def describe_env(env: RaceEnv) -> None:
    """Print useful debugging information about an environment.

    Args:
        env: The wrapped environment to describe.
    """
    print("=" * 60)
    print("RaceMind AI :: Environment description")
    print("=" * 60)
    print(f"Environment name : {env.config.env_name}")
    print(f"Continuous       : {env.config.continuous}")
    print(f"Render mode      : {env.config.render_mode}")
    print(f"Max episode steps: {env.config.max_episode_steps}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")
    print("=" * 60)


def run_random_episode(
    config: Optional[SimulatorConfig] = None,
    max_steps: Optional[int] = None,
) -> float:
    """Run a single episode driven by random actions, printing debug output.

    This is a smoke test for the environment wiring — it trains nothing.

    Args:
        config: Optional configuration; the default is used when omitted.
        max_steps: Optional hard cap on steps; falls back to the configured
            ``max_episode_steps``.

    Returns:
        The cumulative reward earned during the episode.
    """
    config = config or default_config()
    step_limit = max_steps if max_steps is not None else config.max_episode_steps

    with RaceEnv(config) as env:
        describe_env(env)
        observation, info = env.reset()
        print(f"Initial observation shape: {np.asarray(observation).shape}")

        cumulative_reward = 0.0
        for _ in range(step_limit):
            action = env.sample_action()
            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward

            if env.step_count % 50 == 0:
                print(
                    f"step={env.step_count:4d} "
                    f"reward={reward:+.3f} "
                    f"cumulative={cumulative_reward:+.3f}"
                )

            if terminated or truncated:
                break

        print("-" * 60)
        print(
            f"Episode finished after {env.step_count} steps | "
            f"cumulative reward = {cumulative_reward:+.3f}"
        )
        return cumulative_reward


if __name__ == "__main__":
    run_random_episode()

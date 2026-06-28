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

from config.simulator import SimulatorConfig, default_config
from utils.logger import get_logger

_logger = get_logger(__name__)


# Sentinel distinguishing "use the configured render mode" from an explicit None.
_USE_CONFIG_RENDER_MODE = object()


def build_gym_env(
    config: SimulatorConfig,
    render_mode: Any = _USE_CONFIG_RENDER_MODE,
) -> gym.Env:
    """Construct the raw Gymnasium environment described by ``config``.

    This is the single place ``gym.make`` is called, so both :class:`RaceEnv`
    and the environment factory share identical construction logic.

    Args:
        config: Simulator configuration.
        render_mode: Optional render-mode override. When left at its sentinel
            default, ``config.render_mode`` is used; pass an explicit value
            (including ``None``) to override it.

    Returns:
        A freshly created (unwrapped) Gymnasium environment.
    """
    effective_render_mode = (
        config.render_mode
        if render_mode is _USE_CONFIG_RENDER_MODE
        else render_mode
    )
    return gym.make(
        config.env_name,
        render_mode=effective_render_mode,
        continuous=config.continuous,
        max_episode_steps=config.max_episode_steps,
    )


class RaceEnv:
    """Object-oriented wrapper over a single ``CarRacing-v3`` instance.

    The wrapper owns the underlying Gymnasium environment and forwards the
    standard five-tuple step API. It can be used as a context manager so the
    environment is always closed cleanly::

        with RaceEnv(config) as env:
            env.reset()
            env.step(env.sample_action())

    Prefer :func:`simulator.environment_factory.make_env` over constructing
    this class directly, so the standard wrapper stack is applied consistently.
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        env: Optional[gym.Env] = None,
    ) -> None:
        """Create or adopt the wrapped environment.

        Args:
            config: Simulator configuration; the framework default is used when
                omitted.
            env: An already-constructed (and possibly wrapped) Gymnasium
                environment to adopt. When omitted, one is built from ``config``.
        """
        self._config: SimulatorConfig = config or default_config()
        self._env: gym.Env = env if env is not None else build_gym_env(self._config)
        self._step_count: int = 0
        _logger.info("Environment initialized: %s", self._config.env_name)

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
    """Log useful debugging information about an environment.

    Args:
        env: The wrapped environment to describe.
    """
    _logger.info("Environment name: %s", env.config.env_name)
    _logger.info("Continuous: %s", env.config.continuous)
    _logger.info("Render mode: %s", env.config.render_mode)
    _logger.info("Max episode steps: %d", env.config.max_episode_steps)
    _logger.info("Observation space: %s", env.observation_space)
    _logger.info("Action space: %s", env.action_space)


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
    # Local import avoids a circular dependency (the factory imports this module).
    from simulator.environment_factory import make_env

    config = config or default_config()
    step_limit = max_steps if max_steps is not None else config.max_episode_steps

    with make_env(config) as env:
        describe_env(env)
        observation, info = env.reset()
        _logger.info("Initial observation shape: %s", np.asarray(observation).shape)

        cumulative_reward = 0.0
        for _ in range(step_limit):
            action = env.sample_action()
            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward

            if env.step_count % 50 == 0:
                _logger.info(
                    "step=%4d reward=%+.3f cumulative=%+.3f",
                    env.step_count,
                    reward,
                    cumulative_reward,
                )

            if terminated or truncated:
                break

        _logger.info(
            "Episode finished after %d steps | cumulative reward = %+.3f",
            env.step_count,
            cumulative_reward,
        )
        return cumulative_reward


if __name__ == "__main__":
    run_random_episode()

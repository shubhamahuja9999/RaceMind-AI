"""PPO agent for RaceMind AI.

:class:`PPOAgent` wraps Stable-Baselines3 PPO behind the project's
:class:`~agents.base_agent.BaseAgent` interface, so the training and evaluation
framework never touches SB3 directly. TensorBoard logging is enabled by passing
a ``tensorboard_log`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agents.base_agent import BaseAgent
from config.rl import PPOConfig
from utils.logger import get_logger

_logger = get_logger(__name__)


class PPOAgent(BaseAgent):
    """Stable-Baselines3 PPO behind the common agent interface."""

    def __init__(
        self,
        env: gym.Env,
        config: Optional[PPOConfig] = None,
        seed: Optional[int] = None,
        tensorboard_log: Optional[Path] = None,
        csv_log_dir: Optional[Path] = None,
        verbose: int = 1,
        device: str = "auto",
    ) -> None:
        """Construct the underlying PPO model.

        Args:
            env: The (single) training environment. SB3 vectorises it as needed.
            config: PPO hyperparameters; defaults are used when omitted.
            seed: Optional RNG seed for the algorithm.
            tensorboard_log: Optional directory for TensorBoard logs.
            csv_log_dir: Optional directory for a CSV training log
                (``progress.csv``), used to plot learning curves.
            verbose: SB3 verbosity level.
            device: Torch device, e.g. ``"auto"``, ``"cpu"`` or ``"cuda"``.
        """
        self._env = env
        self._config = config or PPOConfig()
        self._tb_log_name = "PPO"
        self._model = PPO(
            self._config.policy,
            env,
            learning_rate=self._config.learning_rate,
            gamma=self._config.gamma,
            n_steps=self._config.n_steps,
            batch_size=self._config.batch_size,
            n_epochs=self._config.n_epochs,
            gae_lambda=self._config.gae_lambda,
            clip_range=self._config.clip_range,
            ent_coef=self._config.ent_coef,
            vf_coef=self._config.vf_coef,
            max_grad_norm=self._config.max_grad_norm,
            seed=seed,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            verbose=verbose,
            device=device,
        )
        if csv_log_dir is not None:
            self._configure_csv_logger(csv_log_dir, bool(tensorboard_log))
        _logger.info(
            "PPOAgent created (policy=%s, device=%s)",
            self._config.policy,
            self._model.device,
        )

    def _configure_csv_logger(self, csv_log_dir: Path, with_tensorboard: bool) -> None:
        """Attach an SB3 logger that also writes ``progress.csv``."""
        from stable_baselines3.common.logger import configure

        formats = ["csv", "tensorboard"] if with_tensorboard else ["csv"]
        self._model.set_logger(configure(str(csv_log_dir), formats))

    @property
    def model(self) -> PPO:
        """The underlying Stable-Baselines3 PPO model."""
        return self._model

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[Any]]:
        """Delegate action selection to the PPO policy."""
        action, state = self._model.predict(observation, deterministic=deterministic)
        return np.asarray(action), state

    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
    ) -> "PPOAgent":
        """Train PPO for ``total_timesteps`` steps."""
        self._model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=self._tb_log_name,
            progress_bar=False,
        )
        return self

    def save(self, path: Path) -> Path:
        """Save the PPO model; returns the written ``.zip`` path."""
        stem = path.with_suffix("")
        self._model.save(str(stem))
        return stem.with_suffix(".zip")

    def load(self, path: Path) -> "PPOAgent":
        """Load PPO weights from ``path`` into this agent's model."""
        candidate = path if path.suffix == ".zip" else path.with_suffix(".zip")
        self._model = PPO.load(str(candidate), env=self._env, device=self._model.device)
        _logger.info("PPO model loaded from %s", candidate)
        return self

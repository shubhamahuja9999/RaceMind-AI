"""PPO trainer assembly for RaceMind AI.

A thin convenience layer that builds a :class:`~agents.ppo_agent.PPOAgent` and
wires it into the generic :class:`~training.trainer.Trainer` via
:func:`~training.trainer.build_trainer`. The trainer itself remains
algorithm-agnostic; this module only knows how to construct the PPO agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym

from agents.ppo_agent import PPOAgent
from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from training.trainer import Trainer, build_trainer


def build_ppo_trainer(
    train_env: gym.Env,
    eval_env: gym.Env,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    checkpoint_dir: Path,
    tensorboard_log: Optional[Path] = None,
    csv_log_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    experiment_name: str = "ppo",
) -> Trainer:
    """Build a PPO agent and a trainer ready to run.

    Args:
        train_env: Environment used for training.
        eval_env: Environment used for evaluation.
        ppo_config: PPO hyperparameters.
        training_config: Training schedule.
        evaluation_config: Evaluation settings.
        checkpoint_dir: Directory for latest/best checkpoints.
        tensorboard_log: Optional TensorBoard log directory.
        csv_log_dir: Optional directory for a CSV training log (learning curves).
        seed: Optional RNG seed.
        experiment_name: Name used in logs and the summary.

    Returns:
        A :class:`Trainer` whose ``agent`` is the PPO agent.
    """
    agent = PPOAgent(
        train_env,
        config=ppo_config,
        seed=seed,
        tensorboard_log=tensorboard_log,
        csv_log_dir=csv_log_dir,
    )
    return build_trainer(
        agent=agent,
        eval_env=eval_env,
        training_config=training_config,
        evaluation_config=evaluation_config,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        experiment_name=experiment_name,
    )

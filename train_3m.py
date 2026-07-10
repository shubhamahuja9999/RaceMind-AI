"""Phase 9: Train longer — continue PPO from 1M with evaluation checkpoints.

Rather than stopping at 1M, this resumes from the 1M run and keeps training
(1M -> 2M -> 3M), evaluating and checkpointing every 100K steps. It stops early
once the evaluation reward plateaus for several consecutive evaluations, so
compute is not wasted once learning has saturated.

Resume behaviour (restart-safe):
    * If a ``ppo_3m`` checkpoint already exists, it continues from there.
    * Otherwise it continues from the final ``ppo_1m`` checkpoint.

This is a long CPU run (~hours). Monitor it with ``tensorboard --logdir runs``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from agents.ppo_agent import PPOAgent
from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from config.simulator import default_config
from simulator.environment_factory import make_eval_env, make_training_env
from training.trainer import build_trainer
from utils.logger import get_logger

_logger = get_logger(__name__)

EXPERIMENT_NAME = "ppo_3m"
SOURCE_RUN = "ppo_1m"


def _resolve_resume_source(checkpoints_dir: Path) -> Optional[Path]:
    """Return the checkpoint to resume from (extension first, then the 1M run)."""
    extended = checkpoints_dir / EXPERIMENT_NAME / "latest.zip"
    if extended.exists():
        _logger.info("Resuming the extended run from %s", extended)
        return extended
    source = checkpoints_dir / SOURCE_RUN / "latest.zip"
    if source.exists():
        _logger.info("Continuing from the 1M run: %s", source)
        return source
    return None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train longer with plateau early-stopping.")
    parser.add_argument(
        "--additional-timesteps", type=int, default=2_000_000,
        help="Extra steps to train beyond the 1M checkpoint (default 2M -> 3M total).",
    )
    parser.add_argument("--eval-frequency", type=int, default=100_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stop after this many evaluations without improvement (plateau).",
    )
    parser.add_argument(
        "--min-delta", type=float, default=10.0,
        help="Minimum eval-reward gain that counts as an improvement.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Resume from 1M and train longer with evaluation checkpoints."""
    args = _parse_args()
    sim_config = default_config()
    sim_config.ensure_directories()

    checkpoint_dir = sim_config.checkpoints_dir / EXPERIMENT_NAME
    tb_log_dir = Path("runs") / EXPERIMENT_NAME

    ppo_config = PPOConfig()
    training_config = TrainingConfig(
        total_timesteps=args.additional_timesteps,
        evaluation_frequency=args.eval_frequency,
        checkpoint_frequency=args.checkpoint_frequency,
    )
    eval_config = EvaluationConfig(n_eval_episodes=5, deterministic=True, record_video=False)

    resume_source = _resolve_resume_source(sim_config.checkpoints_dir)
    if resume_source is None:
        _logger.error(
            "No checkpoint found to resume from. Run train_1m.py first "
            "(expected %s).", sim_config.checkpoints_dir / SOURCE_RUN / "latest.zip",
        )
        return

    _logger.info("=" * 60)
    _logger.info("Phase 9 — Train Longer (target: +%d steps -> ~3M total)", args.additional_timesteps)
    _logger.info("=" * 60)
    _logger.info("PPO config:      %s", ppo_config)
    _logger.info("Training config: %s", training_config)
    _logger.info("Early stop:      patience=%d, min_delta=%.1f", args.patience, args.min_delta)

    train_env = make_training_env(sim_config)
    eval_env = make_eval_env(sim_config, record_video=False)

    agent = PPOAgent(train_env, config=ppo_config, seed=args.seed, tensorboard_log=tb_log_dir)
    agent.load(resume_source, tensorboard_log=tb_log_dir)

    trainer = build_trainer(
        agent=agent,
        eval_env=eval_env,
        training_config=training_config,
        evaluation_config=eval_config,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        experiment_name=EXPERIMENT_NAME,
        reset_num_timesteps=False,  # continue the loaded run's step counter
        early_stop_patience=args.patience,
        early_stop_min_delta=args.min_delta,
    )

    summary = trainer.train()

    _logger.info("=" * 60)
    _logger.info("Train-Longer Results")
    _logger.info("=" * 60)
    _logger.info("Additional steps:  %d", summary.total_timesteps)
    _logger.info("Elapsed:           %.1f min (%.1f hrs)",
                 summary.elapsed_seconds / 60, summary.elapsed_seconds / 3600)
    _logger.info("Steps/sec:         %.1f", summary.timesteps_per_second)
    _logger.info("Best reward:       %.2f", summary.best_reward or 0.0)
    _logger.info("Final reward:      %.2f", summary.final_reward or 0.0)
    _logger.info("Stopped early:     %s", summary.stopped_early)
    _logger.info("Evaluations:")
    for step, reward in summary.evaluations:
        _logger.info("  step=%8d  reward=%.2f", step, reward)
    _logger.info("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

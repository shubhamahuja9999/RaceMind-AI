"""Experiment runner for RaceMind AI — the main entry point for RL experiments.

Responsibilities:
    1. Load configuration from YAML (with optional CLI overrides).
    2. Create training and evaluation environments.
    3. Create the agent and trainer.
    4. Train (unless ``--eval-only``), evaluating and checkpointing along the way.
    5. Run a final benchmark and write a JSON summary.

Examples:
    python -m experiments.run_experiment --experiment-name ppo_carracing
    python -m experiments.run_experiment --algorithm random --total-timesteps 2000
    python -m experiments.run_experiment --eval-only --resume
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym

from agents import build_agent
from config.paths import PROJECT_ROOT
from config.rl import (
    EvaluationConfig,
    PPOConfig,
    TrainingConfig,
    load_evaluation_config,
    load_ppo_config,
    load_simulator_config,
    load_training_config,
)
from config.simulator import SimulatorConfig
from evaluation.benchmark import BenchmarkResult
from simulator.environment_factory import make_eval_env, make_training_env
from training.ppo_trainer import build_ppo_trainer
from training.trainer import Trainer, TrainingSummary, build_trainer
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory
from utils.seed import set_global_seed

_logger = get_logger(__name__)

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
RUNS_DIR: Path = PROJECT_ROOT / "runs"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run a RaceMind AI RL experiment.")
    parser.add_argument("--experiment-name", default="ppo_carracing")
    parser.add_argument("--algorithm", default=None, help="Override algorithm (ppo|random).")
    parser.add_argument("--ppo-config", type=Path, default=CONFIGS_DIR / "ppo.yaml")
    parser.add_argument("--simulator-config", type=Path, default=CONFIGS_DIR / "simulator.yaml")
    parser.add_argument("--eval-config", type=Path, default=CONFIGS_DIR / "evaluation.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate only.")
    parser.add_argument("--no-video", action="store_true", help="Disable evaluation video recording.")
    return parser.parse_args()


def _load_configs(
    args: argparse.Namespace,
) -> tuple[SimulatorConfig, PPOConfig, TrainingConfig, EvaluationConfig]:
    """Load all configs from YAML and apply CLI overrides."""
    simulator_config = load_simulator_config(args.simulator_config)
    ppo_config = load_ppo_config(args.ppo_config)
    training_config = load_training_config(args.ppo_config)
    evaluation_config = load_evaluation_config(args.eval_config)

    overrides: dict[str, Any] = {}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.algorithm is not None:
        overrides["algorithm"] = args.algorithm
    if overrides:
        training_config = TrainingConfig(**{**asdict(training_config), **overrides})
    if args.no_video:
        evaluation_config = EvaluationConfig(**{**asdict(evaluation_config), "record_video": False})

    return simulator_config, ppo_config, training_config, evaluation_config


def _build_trainer(
    args: argparse.Namespace,
    simulator_config: SimulatorConfig,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    train_env: gym.Env,
    eval_env: gym.Env,
    checkpoint_dir: Path,
) -> Trainer:
    """Construct the trainer for the selected algorithm."""
    algorithm = training_config.algorithm.lower()
    if algorithm == "ppo":
        return build_ppo_trainer(
            train_env=train_env,
            eval_env=eval_env,
            ppo_config=ppo_config,
            training_config=training_config,
            evaluation_config=evaluation_config,
            checkpoint_dir=checkpoint_dir,
            tensorboard_log=RUNS_DIR / args.experiment_name,
            seed=training_config.seed,
            experiment_name=args.experiment_name,
        )

    agent = build_agent(algorithm, train_env, seed=training_config.seed)
    return build_trainer(
        agent=agent,
        eval_env=eval_env,
        training_config=training_config,
        evaluation_config=evaluation_config,
        checkpoint_dir=checkpoint_dir,
        seed=training_config.seed,
        experiment_name=args.experiment_name,
    )


def _write_summary(
    path: Path,
    experiment_name: str,
    training_config: TrainingConfig,
    benchmark: BenchmarkResult,
    training_summary: Optional[TrainingSummary],
) -> Path:
    """Write a combined JSON experiment summary to ``path``."""
    payload: dict[str, Any] = {
        "experiment_name": experiment_name,
        "algorithm": training_config.algorithm,
        "seed": training_config.seed,
        "benchmark": benchmark.to_dict(),
        "training": training_summary.to_dict() if training_summary else None,
    }
    return write_json(path, payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the experiment described by ``args`` and return the summary."""
    simulator_config, ppo_config, training_config, evaluation_config = _load_configs(args)
    set_global_seed(training_config.seed)

    checkpoint_dir = simulator_config.checkpoints_dir / args.experiment_name
    video_dir = simulator_config.videos_dir / args.experiment_name
    ensure_directory(simulator_config.evaluation_dir)

    train_env = make_training_env(simulator_config)
    eval_env = make_eval_env(
        simulator_config,
        record_video=evaluation_config.record_video,
        video_dir=video_dir,
        name_prefix=args.experiment_name,
    )

    try:
        trainer = _build_trainer(
            args,
            simulator_config,
            ppo_config,
            training_config,
            evaluation_config,
            train_env,
            eval_env,
            checkpoint_dir,
        )

        if args.resume and trainer.checkpoints.has_latest():
            trainer.checkpoints.load_latest(trainer.agent)

        training_summary: Optional[TrainingSummary] = None
        if not args.eval_only:
            training_summary = trainer.train()

        _logger.info("Running final benchmark for %s", args.experiment_name)
        benchmark = trainer.evaluator.evaluate(trainer.agent)

        summary_path = simulator_config.evaluation_dir / f"{args.experiment_name}.json"
        _write_summary(summary_path, args.experiment_name, training_config, benchmark, training_summary)
        _logger.info("Experiment summary written: %s", summary_path)

        return {
            "benchmark": benchmark.to_dict(),
            "training": training_summary.to_dict() if training_summary else None,
            "summary_path": str(summary_path),
        }
    finally:
        train_env.close()
        eval_env.close()


def main() -> None:
    """Command-line entry point."""
    run(_parse_args())


if __name__ == "__main__":
    main()

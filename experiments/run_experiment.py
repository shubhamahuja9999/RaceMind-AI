"""Experiment runner CLI for RaceMind AI — the main entry point for experiments.

Loads YAML configuration (with optional CLI overrides) and delegates to
:func:`experiments.runner.execute_experiment`, which trains, evaluates and
generates all research artifacts (learning curves, model card, report, registry
entry, rankings).

Examples:
    python -m experiments.run_experiment --experiment-name ppo_carracing
    python -m experiments.run_experiment --algorithm random --total-timesteps 2000
    python -m experiments.run_experiment --eval-only --resume
    python -m experiments.run_experiment --multi-seed
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from experiments.runner import ExperimentResult, execute_experiment
from utils.logger import get_logger

_logger = get_logger(__name__)

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


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
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate only.")
    parser.add_argument("--no-video", action="store_true", help="Disable evaluation video recording.")
    parser.add_argument("--multi-seed", action="store_true", help="Also run multi-seed evaluation.")
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


def run(args: argparse.Namespace) -> ExperimentResult:
    """Execute the experiment described by ``args``."""
    simulator_config, ppo_config, training_config, evaluation_config = _load_configs(args)
    result = execute_experiment(
        experiment_name=args.experiment_name,
        simulator_config=simulator_config,
        ppo_config=ppo_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        resume=args.resume,
        eval_only=args.eval_only,
        multi_seed=args.multi_seed,
    )
    _logger.info("Experiment complete: report at %s", result.report_path)
    return result


def main() -> None:
    """Command-line entry point."""
    run(_parse_args())


if __name__ == "__main__":
    main()

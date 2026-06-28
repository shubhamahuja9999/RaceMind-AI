"""Core experiment orchestration for RaceMind AI.

:func:`execute_experiment` is the single, programmatic entry point that ties the
whole framework together: build environments and agent, train, evaluate, then
generate the research artifacts — learning curves, model card, experiment report,
registry entry and rankings. Both the CLI (``experiments.run_experiment``) and
the hyperparameter sweep call this function, so there is one source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from agents import build_agent
from config.paths import PROJECT_ROOT
from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from config.simulator import SimulatorConfig
from evaluation.benchmark import BenchmarkResult
from evaluation.learning_curves import generate_learning_curves
from evaluation.model_card import build_model_card, save_model_card
from evaluation.multi_seed import evaluate_multi_seed, save_multi_seed
from evaluation.report import ExperimentReport, save_report
from experiments.ranking import export_rankings_csv
from experiments.registry import ExperimentRecord, ExperimentRegistry
from simulator.environment_factory import make_eval_env, make_training_env
from training.ppo_trainer import build_ppo_trainer
from training.trainer import Trainer, TrainingSummary, build_trainer
from utils.gitinfo import get_git_commit
from utils.logger import get_logger
from utils.paths import ensure_directory
from utils.seed import set_global_seed

_logger = get_logger(__name__)

RUNS_DIR: Path = PROJECT_ROOT / "runs"
RANKINGS_FILENAME: str = "rankings.csv"


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved output locations for one experiment."""

    checkpoint_dir: Path
    video_dir: Path
    report_dir: Path
    plots_dir: Path
    logs_dir: Path
    tensorboard_dir: Path

    @property
    def monitor_csv(self) -> Path:
        """Path to the SB3 Monitor CSV for the training env."""
        return self.logs_dir / "monitor.csv"

    @property
    def progress_csv(self) -> Path:
        """Path to the SB3 CSV-logger progress file."""
        return self.logs_dir / "progress.csv"


@dataclass(frozen=True)
class ExperimentResult:
    """The artifacts and metrics produced by an experiment."""

    experiment_name: str
    algorithm: str
    benchmark: dict[str, Any]
    training_summary: Optional[dict[str, Any]]
    multi_seed: Optional[dict[str, Any]]
    plots: list[str]
    model_card_path: Optional[str]
    report_path: Optional[str]
    checkpoint_path: Optional[str]


def _resolve_paths(name: str, config: SimulatorConfig) -> ExperimentPaths:
    """Resolve and create all per-experiment output directories."""
    paths = ExperimentPaths(
        checkpoint_dir=config.checkpoints_dir / name,
        video_dir=config.videos_dir / name,
        report_dir=config.evaluation_dir / name,
        plots_dir=config.plots_dir / name,
        logs_dir=config.checkpoints_dir / name / "logs",
        tensorboard_dir=RUNS_DIR / name,
    )
    for directory in (paths.checkpoint_dir, paths.report_dir, paths.plots_dir, paths.logs_dir):
        ensure_directory(directory)
    return paths


def _make_trainer(
    name: str,
    algorithm: str,
    train_env: gym.Env,
    eval_env: gym.Env,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    paths: ExperimentPaths,
) -> Trainer:
    """Build the trainer for the chosen algorithm."""
    if algorithm == "ppo":
        return build_ppo_trainer(
            train_env=train_env,
            eval_env=eval_env,
            ppo_config=ppo_config,
            training_config=training_config,
            evaluation_config=evaluation_config,
            checkpoint_dir=paths.checkpoint_dir,
            tensorboard_log=paths.tensorboard_dir,
            csv_log_dir=paths.logs_dir,
            seed=training_config.seed,
            experiment_name=name,
        )
    agent = build_agent(algorithm, train_env, seed=training_config.seed)
    return build_trainer(
        agent=agent,
        eval_env=eval_env,
        training_config=training_config,
        evaluation_config=evaluation_config,
        checkpoint_dir=paths.checkpoint_dir,
        seed=training_config.seed,
        experiment_name=name,
    )


def _generate_artifacts(
    name: str,
    algorithm: str,
    simulator_config: SimulatorConfig,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    trainer: Trainer,
    paths: ExperimentPaths,
    benchmark: BenchmarkResult,
    training_summary: Optional[TrainingSummary],
    multi_seed: Optional[dict[str, Any]],
) -> ExperimentResult:
    """Generate plots, model card, report, registry entry and rankings."""
    evaluations = list(training_summary.evaluations) if training_summary else None
    plots = generate_learning_curves(
        paths.plots_dir,
        monitor_csv=paths.monitor_csv,
        progress_csv=paths.progress_csv,
        evaluations=evaluations,
    )

    hyperparameters = asdict(ppo_config) if algorithm == "ppo" else {}
    card = build_model_card(
        algorithm=training_config.algorithm,
        environment=simulator_config.env_name,
        training_timesteps=training_summary.total_timesteps if training_summary else 0,
        seed=training_config.seed,
        hyperparameters=hyperparameters,
        average_reward=benchmark.average_reward,
        best_reward=trainer.checkpoints.best_metric,
        checkpoint_path=trainer.checkpoints.best_path,
        git_commit=get_git_commit(),
        evaluation_results=benchmark.to_dict(),
    )
    card_path = save_model_card(card, paths.checkpoint_dir)

    report = ExperimentReport(
        experiment_name=name,
        configuration={
            "simulator": asdict(simulator_config),
            "ppo": asdict(ppo_config),
            "training": asdict(training_config),
            "evaluation": asdict(evaluation_config),
        },
        training_summary=training_summary.to_dict() if training_summary else None,
        evaluation_summary=benchmark.to_dict(),
        multi_seed=multi_seed,
        plots=[str(p) for p in plots],
        model_card_path=str(card_path),
    )
    _, report_md = save_report(report, paths.report_dir)

    _register(name, training_config, simulator_config, trainer, benchmark, training_summary, card_path, report_md)

    return ExperimentResult(
        experiment_name=name,
        algorithm=training_config.algorithm,
        benchmark=benchmark.to_dict(),
        training_summary=training_summary.to_dict() if training_summary else None,
        multi_seed=multi_seed,
        plots=[str(p) for p in plots],
        model_card_path=str(card_path),
        report_path=str(report_md),
        checkpoint_path=str(trainer.checkpoints.best_path) if trainer.checkpoints.best_path else None,
    )


def _register(
    name: str,
    training_config: TrainingConfig,
    simulator_config: SimulatorConfig,
    trainer: Trainer,
    benchmark: BenchmarkResult,
    training_summary: Optional[TrainingSummary],
    card_path: Path,
    report_md: Path,
) -> None:
    """Add the experiment to the registry and refresh the rankings CSV."""
    registry = ExperimentRegistry()
    registry.register(
        ExperimentRecord(
            experiment_name=name,
            algorithm=training_config.algorithm,
            environment=simulator_config.env_name,
            seed=training_config.seed,
            date=datetime.now().isoformat(timespec="seconds"),
            total_timesteps=training_summary.total_timesteps if training_summary else 0,
            average_reward=benchmark.average_reward,
            best_reward=trainer.checkpoints.best_metric,
            training_seconds=training_summary.elapsed_seconds if training_summary else None,
            checkpoint_path=str(trainer.checkpoints.best_path) if trainer.checkpoints.best_path else None,
            report_path=str(report_md),
            model_card_path=str(card_path),
        )
    )
    export_rankings_csv(registry.all(), simulator_config.evaluation_dir / RANKINGS_FILENAME)


def execute_experiment(
    experiment_name: str,
    simulator_config: SimulatorConfig,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    resume: bool = False,
    eval_only: bool = False,
    multi_seed: bool = False,
) -> ExperimentResult:
    """Run one experiment end-to-end and generate all research artifacts.

    Args:
        experiment_name: Unique experiment name (drives all output paths).
        simulator_config: Environment configuration.
        ppo_config: PPO hyperparameters (ignored for the random algorithm).
        training_config: Training schedule (incl. algorithm and seed).
        evaluation_config: Evaluation settings.
        resume: Resume from the latest checkpoint before training.
        eval_only: Skip training and only evaluate.
        multi_seed: Also run a multi-seed evaluation.

    Returns:
        The :class:`ExperimentResult` with metrics and artifact paths.
    """
    algorithm = training_config.algorithm.lower()
    set_global_seed(training_config.seed)
    paths = _resolve_paths(experiment_name, simulator_config)

    train_env = make_training_env(simulator_config)
    if algorithm == "ppo" and not eval_only:
        train_env = Monitor(train_env, filename=str(paths.monitor_csv))
    eval_env = make_eval_env(
        simulator_config,
        record_video=evaluation_config.record_video,
        video_dir=paths.video_dir,
        name_prefix=experiment_name,
    )

    try:
        trainer = _make_trainer(
            experiment_name, algorithm, train_env, eval_env,
            ppo_config, training_config, evaluation_config, paths,
        )

        if resume and trainer.checkpoints.has_latest():
            trainer.checkpoints.load_latest(trainer.agent)

        training_summary: Optional[TrainingSummary] = None
        if not eval_only:
            training_summary = trainer.train()

        _logger.info("Running final benchmark for %s", experiment_name)
        benchmark = trainer.evaluator.evaluate(trainer.agent)

        multi_seed_dict = _run_multi_seed(
            multi_seed, trainer, simulator_config, evaluation_config, paths,
        )

        return _generate_artifacts(
            experiment_name, algorithm, simulator_config, ppo_config,
            training_config, evaluation_config, trainer, paths, benchmark,
            training_summary, multi_seed_dict,
        )
    finally:
        train_env.close()
        eval_env.close()


def _run_multi_seed(
    enabled: bool,
    trainer: Trainer,
    simulator_config: SimulatorConfig,
    evaluation_config: EvaluationConfig,
    paths: ExperimentPaths,
) -> Optional[dict[str, Any]]:
    """Run a multi-seed evaluation on a non-recording env, if enabled."""
    if not enabled:
        return None
    seed_env = make_eval_env(simulator_config, record_video=False)
    try:
        summary = evaluate_multi_seed(
            trainer.agent,
            seed_env,
            seeds=evaluation_config.eval_seeds,
            n_episodes=evaluation_config.n_eval_episodes,
            deterministic=evaluation_config.deterministic,
            success_threshold=evaluation_config.success_threshold,
            max_steps=evaluation_config.max_steps,
        )
    finally:
        seed_env.close()
    save_multi_seed(summary, paths.report_dir)
    return summary.to_dict()

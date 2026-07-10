"""Experiment 001 — Learning-Rate Schedule (controlled study).

The single independent variable is the learning-rate *schedule*: the baseline
uses a constant ``3e-4``; this experiment uses a linear decay ``3e-4 -> 0`` over
the additional training. Every other PPO hyperparameter, the environment and the
evaluation protocol are held identical to the frozen baseline.

Workflow (all artifacts land in ``research/experiments/001/``):
    1. Save the experiment specification (hypothesis, variables, expectation).
    2. Resume from ``ppo_1m/best.zip`` and train 500k more steps with the
       linear LR schedule, evaluating + checkpointing every 50k.
    3. Evaluate the best checkpoint with the frozen 30-episode protocol.
    4. Compare against the baseline (paired t-test) and render a verdict.
    5. Generate learning curves, a model card, videos and the research paper.

The baseline checkpoint is only ever READ; it is never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agents.ppo_agent import PPOAgent
from config.paths import PROJECT_ROOT
from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from config.simulator import SimulatorConfig, default_config
from evaluation.evaluator import Evaluator
from evaluation.learning_curves import generate_learning_curves
from evaluation.model_card import build_model_card, save_model_card
from evaluation.statistics import RewardStatistics, compute_reward_statistics
from research.comparison import compare_to_baseline
from research.experiment import ExperimentSpec, save_spec
from research.paper import build_experiment_paper
from simulator.environment_factory import make_eval_env, make_training_env
from training.trainer import build_trainer
from utils.gitinfo import get_git_commit
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

EXPERIMENT_ID = "001"
EXPERIMENT_NAME = "lr_schedule_linear_decay"
BASELINE_CHECKPOINT = "ppo_1m/best.zip"
EVAL_SEED_BASE = 1000  # frozen protocol: episode i uses seed 1000 + i
ARCHIVE_DIR = PROJECT_ROOT / "research" / "experiments" / EXPERIMENT_ID


def _build_spec(ppo_config: PPOConfig, sim_config: SimulatorConfig, additional: int) -> ExperimentSpec:
    """Construct the experiment specification (one changed variable)."""
    controlled = {
        "policy": ppo_config.policy,
        "gamma": ppo_config.gamma,
        "n_steps": ppo_config.n_steps,
        "batch_size": ppo_config.batch_size,
        "n_epochs": ppo_config.n_epochs,
        "gae_lambda": ppo_config.gae_lambda,
        "clip_range": ppo_config.clip_range,
        "ent_coef": ppo_config.ent_coef,
        "vf_coef": ppo_config.vf_coef,
        "max_grad_norm": ppo_config.max_grad_norm,
        "environment": sim_config.env_name,
        "resume_checkpoint": BASELINE_CHECKPOINT,
        "additional_timesteps": additional,
        "evaluation": "30 deterministic episodes, seeds 1000-1029",
    }
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        name=EXPERIMENT_NAME,
        hypothesis=(
            "Linearly decaying the learning rate from 3e-4 to 0 over the "
            "additional 500k steps will stabilise late-stage PPO updates and "
            "yield equal-or-higher mean reward than the constant-LR baseline, "
            "avoiding the late-training regression seen with a constant rate."
        ),
        independent_variable=(
            "learning_rate schedule: constant 3e-4 -> linear decay 3e-4 to 0 "
            "(applied as a stepped schedule, updated each 50k-step evaluation chunk)"
        ),
        controlled_variables=controlled,
        expected_outcome="Improved or No significant change (more stable); not Regressed.",
    )


def _evaluate_protocol(
    agent: PPOAgent,
    eval_env,
    episodes: int,
    seed_base: int,
) -> tuple[RewardStatistics, list[float]]:
    """Run the frozen evaluation protocol and return statistics + rewards."""
    result = Evaluator(
        eval_env,
        n_episodes=episodes,
        deterministic=True,
        success_threshold=900.0,
        seed=seed_base,
    ).evaluate(agent)
    rewards = list(result.episode_rewards)
    return compute_reward_statistics(rewards), rewards


def _record_videos(sim_config: SimulatorConfig, best_path: Path, episodes: int, video_dir: Path) -> None:
    """Record a few episodes of the best model for the archive."""
    if episodes < 1:
        return
    video_env = make_eval_env(sim_config, record_video=True, video_dir=video_dir, name_prefix=EXPERIMENT_NAME)
    try:
        agent = PPOAgent(video_env, config=PPOConfig())
        agent.load(best_path)
        Evaluator(video_env, n_episodes=episodes, deterministic=True, seed=EVAL_SEED_BASE).evaluate(agent)
    finally:
        video_env.close()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Experiment 001 (LR schedule).")
    parser.add_argument("--additional-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--initial-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-episodes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run the controlled learning-rate-schedule experiment end to end."""
    args = _parse_args()
    sim_config = default_config()
    sim_config.ensure_directories()

    checkpoints_dir = ensure_directory(ARCHIVE_DIR / "checkpoints")
    plots_dir = ensure_directory(ARCHIVE_DIR / "plots")
    logs_dir = ensure_directory(ARCHIVE_DIR / "logs")
    videos_dir = ARCHIVE_DIR / "videos"

    baseline_ckpt = sim_config.checkpoints_dir / BASELINE_CHECKPOINT
    if not baseline_ckpt.exists():
        _logger.error("Baseline checkpoint not found: %s", baseline_ckpt)
        return

    ppo_config = PPOConfig()  # identical to baseline — nothing changed here
    spec = _build_spec(ppo_config, sim_config, args.additional_timesteps)
    save_spec(spec, ARCHIVE_DIR)

    _logger.info("=" * 60)
    _logger.info("Experiment %s — %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    _logger.info("=" * 60)
    _logger.info("Independent variable: %s", spec.independent_variable)
    _logger.info("Resuming from: %s", baseline_ckpt)

    train_env = make_training_env(sim_config)
    eval_env = make_eval_env(sim_config, record_video=False)

    # --- Resume; the ONLY change is the LR schedule, applied per chunk below ---
    agent = PPOAgent(train_env, config=ppo_config, seed=args.seed)
    agent.load(baseline_ckpt)
    agent.configure_logger(logs_dir, tensorboard=True)

    def linear_lr_schedule(steps_done: int, total: int) -> None:
        """Stepped linear decay 3e-4 -> 0, updated before each 50k chunk."""
        fraction = steps_done / total if total else 0.0
        agent.set_learning_rate(args.initial_lr * (1.0 - fraction))

    training_config = TrainingConfig(
        total_timesteps=args.additional_timesteps,
        evaluation_frequency=args.eval_frequency,
        checkpoint_frequency=args.eval_frequency,
    )
    eval_config = EvaluationConfig(
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        record_video=False,
        success_threshold=900.0,
    )
    trainer = build_trainer(
        agent=agent,
        eval_env=eval_env,
        training_config=training_config,
        evaluation_config=eval_config,
        checkpoint_dir=checkpoints_dir,
        seed=EVAL_SEED_BASE,
        experiment_name=f"exp_{EXPERIMENT_ID}_{EXPERIMENT_NAME}",
        reset_num_timesteps=False,  # continue the loaded run
        on_chunk_start=linear_lr_schedule,  # the single independent variable
    )
    summary = trainer.train()

    # --- Final evaluation of the BEST checkpoint under the frozen protocol ---
    best_path = trainer.checkpoints.best_path
    best_agent = PPOAgent(eval_env, config=ppo_config)
    best_agent.load(best_path)
    exp_stats, exp_rewards = _evaluate_protocol(best_agent, eval_env, args.eval_episodes, EVAL_SEED_BASE)

    train_env.close()
    eval_env.close()

    # --- Automatic comparison against the frozen baseline ---
    comparison = compare_to_baseline(f"experiment_{EXPERIMENT_ID}_{EXPERIMENT_NAME}", exp_rewards)
    write_json(ARCHIVE_DIR / "comparison.json", comparison.to_dict())
    (ARCHIVE_DIR / "comparison.md").write_text(comparison.to_markdown(), encoding="utf-8")
    write_json(ARCHIVE_DIR / "evaluation.json", exp_stats.to_dict())

    # --- Learning curves (training reward, eval reward, LR, entropy, FPS, loss) ---
    plots = generate_learning_curves(
        plots_dir,
        monitor_csv=None,
        progress_csv=logs_dir / "progress.csv",
        evaluations=summary.evaluations,
    )

    # --- Model card (archived beside the checkpoint) ---
    hyperparameters = {**asdict(ppo_config), "learning_rate_schedule": "linear_decay_to_0"}
    card = build_model_card(
        algorithm="PPO",
        environment=sim_config.env_name,
        training_timesteps=1_000_000 + summary.total_timesteps,
        seed=args.seed,
        hyperparameters=hyperparameters,
        average_reward=exp_stats.mean,
        best_reward=trainer.checkpoints.best_metric,
        checkpoint_path=best_path,
        git_commit=get_git_commit(),
        evaluation_results=exp_stats.to_dict(),
    )
    save_model_card(card, checkpoints_dir)

    # --- Videos of the best model ---
    _record_videos(sim_config, best_path, args.video_episodes, videos_dir)

    # --- Research paper (archive + top-level research/) ---
    paper = build_experiment_paper(
        spec=spec,
        comparison=comparison,
        experiment_stats=exp_stats,
        training_timesteps=summary.total_timesteps,
        elapsed_seconds=summary.elapsed_seconds,
        evaluations=summary.evaluations,
        plots=[p.name for p in plots],
        resumed_from=BASELINE_CHECKPOINT,
    )
    (ARCHIVE_DIR / "experiment_001.md").write_text(paper, encoding="utf-8")
    (PROJECT_ROOT / "research" / "experiment_001.md").write_text(paper, encoding="utf-8")

    _logger.info("=" * 60)
    _logger.info("Experiment %s complete", EXPERIMENT_ID)
    _logger.info("  Baseline mean:   %.2f", comparison.baseline_mean)
    _logger.info("  Experiment mean: %.2f", comparison.experiment_mean)
    _logger.info("  Absolute change: %+.2f (%+.2f%%)", comparison.absolute_improvement, comparison.percent_improvement)
    _logger.info("  Verdict:         %s", comparison.verdict)
    _logger.info("  Archive:         %s", ARCHIVE_DIR)
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()

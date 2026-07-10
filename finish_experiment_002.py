"""Resume and complete Experiment 002 after an interrupted run.

The original 1M-from-scratch FrameStack run was stopped at 700k steps when the
session ended. This script continues from the saved ``latest.zip`` (700k) to the
full 1M, then generates the final artifacts (evaluation, paired comparison,
plots, model card, videos, paper). It reuses the Experiment 002 building blocks
so the methodology is identical — only the training was split across two sittings.

The frozen baseline is never read or modified.
"""

from __future__ import annotations

from dataclasses import asdict

from agents.ppo_agent import PPOAgent
from config.paths import PROJECT_ROOT
from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from config.simulator import default_config
from evaluation.learning_curves import generate_learning_curves
from evaluation.model_card import build_model_card, save_model_card
from research.comparison import compare_to_baseline
from research.experiment import save_spec
from research.paper import build_experiment_paper
from run_experiment_002 import (
    ABSTRACT,
    ARCHIVE_DIR,
    EVAL_SEED_BASE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    FRAME_STACK,
    _build_spec,
    _evaluate_protocol,
    _prose,
    _record_stage_video,
)
from simulator.environment_factory import make_eval_env, make_training_env
from training.trainer import build_trainer
from utils.gitinfo import get_git_commit
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

TARGET_TIMESTEPS = 1_000_000
# Evaluations recorded during the original (interrupted) 700k run, from the log.
PRIOR_EVALS: list[tuple[int, float]] = [
    (100_000, 169.62), (200_000, 185.05), (300_000, 169.07), (400_000, 10.41),
    (500_000, 74.90), (600_000, 278.16), (700_000, 282.11),
]
# Wall-clock of the original run to reach 700k (from the log: elapsed=36746.9s).
ORIGINAL_ELAPSED_SECONDS = 36_746.9


def main() -> None:
    """Continue training to 1M and emit the full Experiment 002 archive."""
    sim_config = default_config()
    sim_config.ensure_directories()

    checkpoints_dir = ARCHIVE_DIR / "checkpoints"
    plots_dir = ensure_directory(ARCHIVE_DIR / "plots")
    logs_dir = ARCHIVE_DIR / "logs"                       # original training log (0-700k)
    resume_logs = ensure_directory(ARCHIVE_DIR / "logs_resume")
    videos_dir = ARCHIVE_DIR / "videos"

    latest = checkpoints_dir / "latest.zip"
    if not latest.exists():
        _logger.error("No checkpoint to resume from: %s", latest)
        return

    ppo_config = PPOConfig()
    spec = _build_spec(ppo_config, sim_config)
    save_spec(spec, ARCHIVE_DIR)

    train_env = make_training_env(sim_config, frame_stack=FRAME_STACK)
    eval_env = make_eval_env(sim_config, record_video=False, frame_stack=FRAME_STACK)

    agent = PPOAgent(train_env, config=ppo_config, seed=42)
    agent.load(latest)
    agent.configure_logger(resume_logs, tensorboard=True)

    done = agent.model.num_timesteps
    remaining = max(0, TARGET_TIMESTEPS - done)
    _logger.info("Resuming Experiment 002 from %d steps; remaining = %d", done, remaining)

    new_evals: list[tuple[int, float]] = []
    elapsed = 0.0
    total_steps = done
    best_path = checkpoints_dir / "best.zip"

    if remaining > 0:
        training_config = TrainingConfig(
            total_timesteps=remaining,
            evaluation_frequency=100_000,
            checkpoint_frequency=100_000,
        )
        eval_config = EvaluationConfig(
            n_eval_episodes=30, deterministic=True, record_video=False, success_threshold=900.0,
        )
        trainer = build_trainer(
            agent=agent,
            eval_env=eval_env,
            training_config=training_config,
            evaluation_config=eval_config,
            checkpoint_dir=checkpoints_dir,
            seed=EVAL_SEED_BASE,
            experiment_name=f"exp_{EXPERIMENT_ID}_{EXPERIMENT_NAME}",
            reset_num_timesteps=False,  # continue the same from-scratch run
        )
        summary = trainer.train()
        new_evals = [(done + step, reward) for step, reward in summary.evaluations]
        best_path = trainer.checkpoints.best_path
        best_metric = trainer.checkpoints.best_metric
        elapsed = summary.elapsed_seconds
        total_steps = done + summary.total_timesteps
    else:
        best_metric = 282.11

    # --- Final protocol evaluation of the best checkpoint over the full run ---
    best_agent = PPOAgent(eval_env, config=ppo_config)
    best_agent.load(best_path)
    exp_stats, exp_rewards = _evaluate_protocol(best_agent, eval_env, 30, EVAL_SEED_BASE)

    train_env.close()
    eval_env.close()

    full_evals = PRIOR_EVALS + new_evals
    total_elapsed = ORIGINAL_ELAPSED_SECONDS + elapsed

    # --- Comparison, plots, model card, videos, paper ---
    comparison = compare_to_baseline(f"experiment_{EXPERIMENT_ID}_{EXPERIMENT_NAME}", exp_rewards)
    write_json(ARCHIVE_DIR / "comparison.json", comparison.to_dict())
    (ARCHIVE_DIR / "comparison.md").write_text(comparison.to_markdown(), encoding="utf-8")
    write_json(ARCHIVE_DIR / "evaluation.json", exp_stats.to_dict())

    plots = generate_learning_curves(
        plots_dir, monitor_csv=None, progress_csv=logs_dir / "progress.csv", evaluations=full_evals,
    )

    hyperparameters = {**asdict(ppo_config), "frame_stack": FRAME_STACK, "learning_rate_schedule": "constant"}
    card = build_model_card(
        algorithm="PPO", environment=sim_config.env_name, training_timesteps=total_steps,
        seed=42, hyperparameters=hyperparameters, average_reward=exp_stats.mean,
        best_reward=best_metric, checkpoint_path=best_path, git_commit=get_git_commit(),
        evaluation_results=exp_stats.to_dict(),
    )
    save_model_card(card, checkpoints_dir)

    _record_stage_video(sim_config, None, "random", videos_dir, 1)
    _record_stage_video(sim_config, checkpoints_dir / "checkpoint_100000.zip", "early", videos_dir, 1)
    _record_stage_video(sim_config, checkpoints_dir / "checkpoint_500000.zip", "mid", videos_dir, 1)
    _record_stage_video(sim_config, best_path, "best", videos_dir, 1)

    interpretation, lessons, future_work = _prose(comparison.verdict)
    paper = build_experiment_paper(
        spec=spec, comparison=comparison, experiment_stats=exp_stats,
        training_timesteps=total_steps, elapsed_seconds=total_elapsed,
        evaluations=full_evals, plots=[p.name for p in plots], resumed_from=None,
        abstract=ABSTRACT, interpretation=interpretation, lessons=lessons, future_work=future_work,
    )
    (ARCHIVE_DIR / "experiment_002.md").write_text(paper, encoding="utf-8")
    (PROJECT_ROOT / "research" / "experiment_002.md").write_text(paper, encoding="utf-8")

    _logger.info("=" * 60)
    _logger.info("Experiment %s complete (total steps: %d)", EXPERIMENT_ID, total_steps)
    _logger.info("  Baseline mean:   %.2f", comparison.baseline_mean)
    _logger.info("  Experiment mean: %.2f", comparison.experiment_mean)
    _logger.info("  Absolute change: %+.2f (%+.2f%%)", comparison.absolute_improvement, comparison.percent_improvement)
    _logger.info("  Effect size (d): %.3f (%s)", comparison.effect_size, comparison.effect_size_label)
    _logger.info("  Verdict:         %s", comparison.verdict)
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()

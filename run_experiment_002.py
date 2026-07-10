"""Experiment 002 — Temporal Observation via FrameStack(4) (controlled study).

The single independent variable is the observation: the baseline sees one frame;
this experiment stacks the last 4 frames on the channel axis (temporal
observation). Every PPO hyperparameter, the environment and the evaluation
protocol are identical to the frozen baseline, and the policy is trained **from
scratch** (no checkpoint is loaded).

All artifacts land in ``research/experiments/002/``. The frozen baseline
(``ppo_1m/best.zip``) is never read or modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from config.paths import PROJECT_ROOT
from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from config.simulator import SimulatorConfig, default_config
from evaluation.evaluator import Evaluator
from evaluation.learning_curves import generate_learning_curves
from evaluation.model_card import build_model_card, save_model_card
from evaluation.statistics import RewardStatistics, compute_reward_statistics
from research.comparison import IMPROVED, NO_CHANGE, REGRESSED, compare_to_baseline
from research.experiment import ExperimentSpec, save_spec
from research.paper import build_experiment_paper
from simulator.environment_factory import make_eval_env, make_training_env
from training.trainer import build_trainer
from utils.gitinfo import get_git_commit
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

EXPERIMENT_ID = "002"
EXPERIMENT_NAME = "framestack_4"
FRAME_STACK = 4
EVAL_SEED_BASE = 1000  # frozen protocol: episode i uses seed 1000 + i
ARCHIVE_DIR = PROJECT_ROOT / "research" / "experiments" / EXPERIMENT_ID

ABSTRACT = (
    "CarRacing-v3 is partially observable from a single frame: velocity and "
    "angular momentum are not directly observable. This experiment tests whether "
    "providing four stacked consecutive frames (temporal observation) improves a "
    "PPO agent relative to the frozen single-frame baseline, changing only the "
    "observation representation and holding all other factors constant. The policy "
    "is trained from scratch for 1M steps and evaluated with a 30-episode paired "
    "protocol against the baseline."
)


def _build_spec(ppo_config: PPOConfig, sim_config: SimulatorConfig) -> ExperimentSpec:
    """Construct the experiment specification (one changed variable)."""
    controlled = {
        "algorithm": "PPO (CnnPolicy)",
        "environment": sim_config.env_name,
        "learning_rate": f"{ppo_config.learning_rate} (constant)",
        "ent_coef": ppo_config.ent_coef,
        "gamma": ppo_config.gamma,
        "n_steps": ppo_config.n_steps,
        "batch_size": ppo_config.batch_size,
        "n_epochs": ppo_config.n_epochs,
        "clip_range": ppo_config.clip_range,
        "gae_lambda": ppo_config.gae_lambda,
        "vf_coef": ppo_config.vf_coef,
        "max_grad_norm": ppo_config.max_grad_norm,
        "training_timesteps": 1_000_000,
        "initialisation": "from scratch (no checkpoint loaded)",
        "evaluation": "30 deterministic episodes, seeds 1000-1029",
    }
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        name=EXPERIMENT_NAME,
        hypothesis=(
            "Providing the policy with four consecutive frames will improve driving "
            "performance because CarRacing is partially observable from a single frame."
        ),
        independent_variable="observation: single frame -> FrameStack(4) (temporal, channel-stacked)",
        controlled_variables=controlled,
        expected_outcome="Improved (reject the null hypothesis) if partial observability matters.",
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


def _record_stage_video(
    sim_config: SimulatorConfig,
    checkpoint_path: Optional[Path],
    name_prefix: str,
    video_dir: Path,
    episodes: int,
) -> None:
    """Record ``episodes`` of a stage (random if checkpoint_path is None)."""
    if episodes < 1:
        return
    if checkpoint_path is not None and not checkpoint_path.exists():
        _logger.warning("Skipping %s video; checkpoint missing: %s", name_prefix, checkpoint_path)
        return
    env = make_eval_env(
        sim_config, record_video=True, video_dir=video_dir,
        name_prefix=name_prefix, frame_stack=FRAME_STACK,
    )
    try:
        if checkpoint_path is None:
            agent = RandomAgent(env.action_space)
        else:
            agent = PPOAgent(env, config=PPOConfig())
            agent.load(checkpoint_path)
        Evaluator(env, n_episodes=episodes, deterministic=True, seed=EVAL_SEED_BASE).evaluate(agent)
        _logger.info("Recorded %s video(s): %s", name_prefix, video_dir)
    finally:
        env.close()


def _prose(verdict: str) -> tuple[str, str, str]:
    """Return (interpretation, lessons, future_work) tailored to the verdict."""
    if verdict == IMPROVED:
        return (
            "FrameStack(4) produced a statistically significant improvement over the "
            "single-frame baseline, supporting the hypothesis that CarRacing's partial "
            "observability (unobservable velocity/heading rate from one frame) limits "
            "the baseline. Temporal observation lets the CNN infer motion.",
            "Temporal stacking is a worthwhile, low-risk representation change for this "
            "task and should be adopted in the default configuration.",
            "Combine FrameStack(4) with the other levers (e.g. grayscale to offset the "
            "added channels, or a larger training budget) as Experiment 003.",
        )
    if verdict == REGRESSED:
        return (
            "FrameStack(4) produced a statistically significant regression. The 4x "
            "larger observation (12 channels) increases the input dimensionality and, "
            "within the fixed 1M-step budget and unchanged hyperparameters, the policy "
            "learned more slowly rather than benefiting from temporal information. The "
            "hypothesis is rejected under these conditions.",
            "Adding input channels is not free: it can slow convergence at a fixed "
            "budget. Frame stacking should be paired with grayscale (to keep channel "
            "count comparable) and/or a larger training budget before judging it.",
            "Re-run FrameStack with grayscale (4 stacked grayscale frames = 4 channels, "
            "comparable to the baseline's 3) to isolate temporal information from the "
            "channel-count increase, as Experiment 003.",
        )
    return (
        "FrameStack(4) produced no statistically significant difference from the "
        "single-frame baseline; the null hypothesis cannot be rejected. Either a "
        "single frame is already near-sufficient for the CNN on this track, or 1M "
        "steps is too few for the larger input to pay off given the high evaluation "
        "variance.",
        "A single 1M-step run at high per-episode variance is underpowered to detect a "
        "modest representation effect. More evaluation episodes or a larger training "
        "budget would be needed to resolve it.",
        "Increase statistical power (more eval episodes / seeds) or the training "
        "budget, then re-test FrameStack, as Experiment 003.",
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Experiment 002 (FrameStack).")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-frequency", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Run the controlled FrameStack experiment end to end."""
    args = _parse_args()
    sim_config = default_config()
    sim_config.ensure_directories()

    checkpoints_dir = ensure_directory(ARCHIVE_DIR / "checkpoints")
    plots_dir = ensure_directory(ARCHIVE_DIR / "plots")
    logs_dir = ensure_directory(ARCHIVE_DIR / "logs")
    videos_dir = ARCHIVE_DIR / "videos"

    ppo_config = PPOConfig()  # identical to baseline — nothing changed here
    spec = _build_spec(ppo_config, sim_config)
    save_spec(spec, ARCHIVE_DIR)

    _logger.info("=" * 60)
    _logger.info("Experiment %s — %s (train from scratch)", EXPERIMENT_ID, EXPERIMENT_NAME)
    _logger.info("=" * 60)
    _logger.info("Independent variable: %s", spec.independent_variable)

    # --- Envs with the ONLY change: FrameStack(4) ---
    train_env = make_training_env(sim_config, frame_stack=FRAME_STACK)
    eval_env = make_eval_env(sim_config, record_video=False, frame_stack=FRAME_STACK)

    agent = PPOAgent(train_env, config=ppo_config, seed=args.seed)
    agent.configure_logger(logs_dir, tensorboard=True)

    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
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
        reset_num_timesteps=True,  # from scratch
    )
    summary = trainer.train()

    # --- Final evaluation of the BEST checkpoint under the frozen protocol ---
    best_path = trainer.checkpoints.best_path
    best_agent = PPOAgent(eval_env, config=ppo_config)
    best_agent.load(best_path)
    exp_stats, exp_rewards = _evaluate_protocol(best_agent, eval_env, args.eval_episodes, EVAL_SEED_BASE)

    train_env.close()
    eval_env.close()

    # --- Automatic comparison against the frozen baseline (paired + effect size) ---
    comparison = compare_to_baseline(f"experiment_{EXPERIMENT_ID}_{EXPERIMENT_NAME}", exp_rewards)
    write_json(ARCHIVE_DIR / "comparison.json", comparison.to_dict())
    (ARCHIVE_DIR / "comparison.md").write_text(comparison.to_markdown(), encoding="utf-8")
    write_json(ARCHIVE_DIR / "evaluation.json", exp_stats.to_dict())

    # --- Learning curves ---
    plots = generate_learning_curves(
        plots_dir,
        monitor_csv=None,
        progress_csv=logs_dir / "progress.csv",
        evaluations=summary.evaluations,
    )

    # --- Model card (FrameStack enabled) ---
    hyperparameters = {**asdict(ppo_config), "frame_stack": FRAME_STACK, "learning_rate_schedule": "constant"}
    card = build_model_card(
        algorithm="PPO",
        environment=sim_config.env_name,
        training_timesteps=summary.total_timesteps,
        seed=args.seed,
        hyperparameters=hyperparameters,
        average_reward=exp_stats.mean,
        best_reward=trainer.checkpoints.best_metric,
        checkpoint_path=best_path,
        git_commit=get_git_commit(),
        evaluation_results=exp_stats.to_dict(),
    )
    save_model_card(card, checkpoints_dir)

    # --- Videos: random, early, mid, best ---
    early_step = args.eval_frequency
    mid_step = max(args.eval_frequency, (args.timesteps // 2 // args.eval_frequency) * args.eval_frequency)
    _record_stage_video(sim_config, None, "random", videos_dir, args.video_episodes)
    _record_stage_video(sim_config, checkpoints_dir / f"checkpoint_{early_step}.zip", "early", videos_dir, args.video_episodes)
    _record_stage_video(sim_config, checkpoints_dir / f"checkpoint_{mid_step}.zip", "mid", videos_dir, args.video_episodes)
    _record_stage_video(sim_config, best_path, "best", videos_dir, args.video_episodes)

    # --- Research paper ---
    interpretation, lessons, future_work = _prose(comparison.verdict)
    paper = build_experiment_paper(
        spec=spec,
        comparison=comparison,
        experiment_stats=exp_stats,
        training_timesteps=summary.total_timesteps,
        elapsed_seconds=summary.elapsed_seconds,
        evaluations=summary.evaluations,
        plots=[p.name for p in plots],
        resumed_from=None,  # trained from scratch
        abstract=ABSTRACT,
        interpretation=interpretation,
        lessons=lessons,
        future_work=future_work,
    )
    (ARCHIVE_DIR / "experiment_002.md").write_text(paper, encoding="utf-8")
    (PROJECT_ROOT / "research" / "experiment_002.md").write_text(paper, encoding="utf-8")

    _logger.info("=" * 60)
    _logger.info("Experiment %s complete", EXPERIMENT_ID)
    _logger.info("  Baseline mean:   %.2f", comparison.baseline_mean)
    _logger.info("  Experiment mean: %.2f", comparison.experiment_mean)
    _logger.info("  Absolute change: %+.2f (%+.2f%%)", comparison.absolute_improvement, comparison.percent_improvement)
    _logger.info("  Effect size (d): %.3f (%s)", comparison.effect_size, comparison.effect_size_label)
    _logger.info("  Verdict:         %s", comparison.verdict)
    _logger.info("  Archive:         %s", ARCHIVE_DIR)
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()

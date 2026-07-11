"""Reward ablation experiment runner for RaceMind AI.

Trains PPO with a chosen reward configuration (the ONLY thing that changes),
then evaluates on the native task reward with the frozen 30-episode protocol and
compares against the baseline. Everything except the reward is held identical to
the baseline (single-frame observation, unchanged PPO hyperparameters).

The default training budget is a small smoke-test value — this runner never
launches a long run automatically. Point ``--config`` at a reward YAML to select
the ablation:

    python -m research.run_reward_experiment --config configs/reward_centerline.yaml

Evaluation always uses the native environment reward (reward shaping is applied
to training only), so results are directly comparable to the baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
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
from reward.reward_config import RewardConfig
from reward.reward_logger import collect_reward_log
from reward.reward_manager import RewardManager
from reward.reward_plots import generate_reward_plots
from simulator.environment_factory import make_eval_env, make_training_env
from training.trainer import build_trainer
from utils.gitinfo import get_git_commit
from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

EVAL_SEED_BASE = 1000  # frozen protocol: episode i uses seed 1000 + i
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "research" / "experiments"
REWARD_ANALYSIS_DIR = PROJECT_ROOT / "research" / "reward_analysis"


def _build_spec(name: str, reward_config: RewardConfig, ppo_config: PPOConfig) -> ExperimentSpec:
    """Construct the experiment specification (reward config is the variable)."""
    enabled = {
        component: {"weight": cfg.weight}
        for component, cfg in reward_config.enabled_components().items()
    }
    controlled = {
        "algorithm": "PPO (CnnPolicy)",
        "environment": "CarRacing-v3",
        "observation": "single frame (no FrameStack)",
        "learning_rate": f"{ppo_config.learning_rate} (constant)",
        "ent_coef": ppo_config.ent_coef,
        "gamma": ppo_config.gamma,
        "n_steps": ppo_config.n_steps,
        "batch_size": ppo_config.batch_size,
        "evaluation": "30 deterministic episodes, seeds 1000-1029 (native reward)",
    }
    return ExperimentSpec(
        experiment_id=name,
        name=name,
        hypothesis=(
            "The enabled reward components change the driving behaviour the policy "
            "learns; this experiment measures the effect on native-task performance "
            "relative to the progress-only baseline reward."
        ),
        independent_variable=f"reward composition: enabled components = {enabled}",
        controlled_variables=controlled,
        expected_outcome="To be determined by the paired comparison (this is a controlled study, not an optimisation).",
    )


def _evaluate_protocol(agent, eval_env, episodes: int) -> tuple[RewardStatistics, list[float]]:
    """Run the frozen 30-episode protocol (native reward) and return stats."""
    result = Evaluator(
        eval_env, n_episodes=episodes, deterministic=True,
        success_threshold=900.0, seed=EVAL_SEED_BASE,
    ).evaluate(agent)
    rewards = list(result.episode_rewards)
    return compute_reward_statistics(rewards), rewards


def _reward_analysis(
    sim_config: SimulatorConfig,
    manager: RewardManager,
    best_path: Path,
    analysis_dir: Path,
) -> list[Path]:
    """Record a per-component reward log and generate reward plots."""
    ensure_directory(analysis_dir)
    shaped_env = make_training_env(sim_config, reward_manager=manager)
    try:
        agent = PPOAgent(shaped_env, config=PPOConfig())
        agent.load(best_path)
        logger = collect_reward_log(shaped_env, agent, manager.component_names, max_steps=1000, seed=EVAL_SEED_BASE)
    finally:
        shaped_env.close()
    csv_path = logger.save(analysis_dir / "reward_log.csv")
    return generate_reward_plots(csv_path, analysis_dir, manager.component_names)


def _record_videos(sim_config: SimulatorConfig, best_path: Path, video_dir: Path, episodes: int) -> None:
    """Record a random-policy and best-policy video on the native-reward env."""
    if episodes < 1:
        return
    stages = [("random", None), ("best", best_path)]
    for name_prefix, checkpoint in stages:
        env = make_eval_env(sim_config, record_video=True, video_dir=video_dir, name_prefix=name_prefix)
        try:
            if checkpoint is None:
                agent = RandomAgent(env.action_space)
            else:
                agent = PPOAgent(env, config=PPOConfig())
                agent.load(checkpoint)
            Evaluator(env, n_episodes=episodes, deterministic=True, seed=EVAL_SEED_BASE).evaluate(agent)
        finally:
            env.close()
    _logger.info("Videos recorded: %s", video_dir)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a reward ablation experiment.")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "reward.yaml",
                        help="Reward YAML config (the single independent variable).")
    parser.add_argument("--experiment-name", default=None, help="Defaults to the config file stem.")
    parser.add_argument("--timesteps", type=int, default=5_000, help="Training budget (default: small smoke).")
    parser.add_argument("--eval-frequency", type=int, default=5_000)
    parser.add_argument("--train-eval-episodes", type=int, default=5, help="Episodes for in-training checkpoint eval.")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Episodes for the final protocol comparison.")
    parser.add_argument("--video-episodes", type=int, default=1, help="Episodes per stage video (0 disables).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run one reward ablation experiment end to end (smoke budget by default)."""
    args = _parse_args()
    name = args.experiment_name or args.config.stem
    sim_config = default_config()
    sim_config.ensure_directories()

    archive_dir = ensure_directory(EXPERIMENTS_DIR / name)
    checkpoints_dir = ensure_directory(archive_dir / "checkpoints")
    logs_dir = ensure_directory(archive_dir / "logs")
    plots_dir = ensure_directory(archive_dir / "plots")
    videos_dir = archive_dir / "videos"
    analysis_dir = REWARD_ANALYSIS_DIR / name

    reward_config = RewardConfig.from_yaml(args.config)
    manager = RewardManager(reward_config)
    ppo_config = PPOConfig()  # identical to baseline

    spec = _build_spec(name, reward_config, ppo_config)
    save_spec(spec, archive_dir)

    _logger.info("=" * 60)
    _logger.info("Reward experiment: %s (budget=%d steps)", name, args.timesteps)
    _logger.info("Reward config: %s", args.config)
    _logger.info("Enabled components: %s", manager.component_names)
    _logger.info("=" * 60)

    # Training env uses the composed reward; eval env uses the NATIVE reward.
    train_env = make_training_env(sim_config, reward_manager=manager)
    eval_env = make_eval_env(sim_config)

    agent = PPOAgent(train_env, config=ppo_config, seed=args.seed)
    agent.configure_logger(logs_dir, tensorboard=True)

    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        evaluation_frequency=min(args.eval_frequency, args.timesteps),
        checkpoint_frequency=min(args.eval_frequency, args.timesteps),
    )
    eval_config = EvaluationConfig(
        n_eval_episodes=args.train_eval_episodes, deterministic=True,
        record_video=False, success_threshold=900.0,
    )
    trainer = build_trainer(
        agent=agent, eval_env=eval_env, training_config=training_config,
        evaluation_config=eval_config, checkpoint_dir=checkpoints_dir,
        seed=EVAL_SEED_BASE, experiment_name=name, reset_num_timesteps=True,
    )
    summary = trainer.train()

    # --- Final evaluation (native reward, 30-episode protocol) ---
    best_path = trainer.checkpoints.best_path
    best_agent = PPOAgent(eval_env, config=ppo_config)
    best_agent.load(best_path)
    exp_stats, exp_rewards = _evaluate_protocol(best_agent, eval_env, args.eval_episodes)

    train_env.close()
    eval_env.close()

    # --- Comparison, reward analysis, model card, paper ---
    comparison = compare_to_baseline(name, exp_rewards)
    write_json(archive_dir / "comparison.json", comparison.to_dict())
    (archive_dir / "comparison.md").write_text(comparison.to_markdown(), encoding="utf-8")
    write_json(archive_dir / "evaluation.json", exp_stats.to_dict())

    reward_plots = _reward_analysis(sim_config, manager, best_path, analysis_dir)

    # --- Learning curves (training reward, eval reward, LR, entropy, FPS) ---
    learning_curves = generate_learning_curves(
        plots_dir, monitor_csv=None, progress_csv=logs_dir / "progress.csv",
        evaluations=summary.evaluations,
    )

    # --- Videos (random + best policy on the native-reward env) ---
    _record_videos(sim_config, best_path, videos_dir, args.video_episodes)

    hyperparameters = {**asdict(ppo_config), "reward_config": args.config.name,
                       "enabled_components": manager.component_names}
    card = build_model_card(
        algorithm="PPO", environment=sim_config.env_name, training_timesteps=summary.total_timesteps,
        seed=args.seed, hyperparameters=hyperparameters, average_reward=exp_stats.mean,
        best_reward=trainer.checkpoints.best_metric, checkpoint_path=best_path,
        git_commit=get_git_commit(), evaluation_results=exp_stats.to_dict(),
    )
    save_model_card(card, checkpoints_dir)

    abstract = (
        f"Reward ablation '{name}': the enabled reward components are "
        f"{manager.component_names}; all other factors match the frozen baseline. "
        f"Trained for {summary.total_timesteps:,} steps (budget as configured) and "
        "evaluated on the native task reward with the 30-episode paired protocol."
    )
    all_plots = [p.name for p in learning_curves] + [p.name for p in reward_plots]
    paper = build_experiment_paper(
        spec=spec, comparison=comparison, experiment_stats=exp_stats,
        training_timesteps=summary.total_timesteps, elapsed_seconds=summary.elapsed_seconds,
        evaluations=summary.evaluations, plots=all_plots,
        resumed_from=None, abstract=abstract,
    )
    (archive_dir / f"{name}.md").write_text(paper, encoding="utf-8")

    _logger.info("=" * 60)
    _logger.info("Reward experiment complete: %s", name)
    _logger.info("  Enabled components: %s", manager.component_names)
    _logger.info("  Baseline mean:   %.2f", comparison.baseline_mean)
    _logger.info("  Experiment mean: %.2f", comparison.experiment_mean)
    _logger.info("  Verdict:         %s", comparison.verdict)
    _logger.info("  Archive:         %s", archive_dir)
    _logger.info("  Reward analysis: %s", analysis_dir)
    _logger.info("  NOTE: default budget is a smoke value; results are not meaningful until trained longer.")
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""Phase 10: High-confidence head-to-head evaluation of two checkpoints.

Compares ``ppo_1m/best`` against ``ppo_3m/best`` under identical, reproducible
conditions:

    * 30 evaluation episodes (configurable)
    * fixed evaluation seeds (episode i uses ``seed_base + i`` for BOTH agents)
    * identical env / deterministic policy

Reports, per model: mean, median, standard deviation, 95% confidence interval,
best and worst — and saves every episode's reward. Output is written as a
benchmark report (Markdown + JSON) plus a tidy per-episode CSV, so it becomes
part of the benchmark record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.ppo_agent import PPOAgent
from config.rl import PPOConfig
from config.simulator import default_config
from evaluation.benchmark_report import comparison_to_markdown, save_comparison
from evaluation.benchmark_runner import AgentBenchmark, BenchmarkRunner
from simulator.environment_factory import make_eval_env
from utils.io import write_csv
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

# Checkpoints to compare: display name -> path relative to the checkpoints dir.
CHECKPOINTS: dict[str, str] = {
    "ppo_1m_best": "ppo_1m/best.zip",
    "ppo_3m_best": "ppo_3m/best.zip",
}
OUTPUT_SUBDIR = "checkpoint_comparison"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="High-confidence checkpoint comparison.")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per agent.")
    parser.add_argument("--seed", type=int, default=1000, help="Base evaluation seed (fixed).")
    parser.add_argument("--success-threshold", type=float, default=900.0)
    return parser.parse_args()


def _load_agents(checkpoints_dir: Path, eval_env) -> dict[str, PPOAgent]:
    """Load each checkpoint into a PPO agent sharing ``eval_env``."""
    agents: dict[str, PPOAgent] = {}
    for name, relative in CHECKPOINTS.items():
        path = checkpoints_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        agent = PPOAgent(eval_env, config=PPOConfig())
        agent.load(path)
        agents[name] = agent
        _logger.info("Loaded %s from %s", name, path)
    return agents


def _save_episode_rewards(
    output_dir: Path,
    seed_base: int,
    agents: dict[str, AgentBenchmark],
) -> Path:
    """Write a per-episode reward CSV (one column per agent, one row per seed)."""
    names = list(agents)
    n_episodes = agents[names[0]].n_episodes
    rows = []
    for i in range(n_episodes):
        row = {"episode": i, "seed": seed_base + i}
        for name in names:
            row[f"{name}_reward"] = agents[name].episode_rewards[i]
        rows.append(row)
    fieldnames = ["episode", "seed", *(f"{name}_reward" for name in names)]
    path = output_dir / "episode_rewards.csv"
    write_csv(path, rows, fieldnames)
    return path


def _report_line(agent: AgentBenchmark) -> None:
    """Log the required statistics block for one agent."""
    _logger.info("  %s (n=%d)", agent.name, agent.n_episodes)
    _logger.info("    Mean reward:   %.2f", agent.average_reward)
    _logger.info("    Median reward: %.2f", agent.median_reward)
    _logger.info("    Std deviation: %.2f", agent.std_reward)
    _logger.info("    95%% CI:        [%.2f, %.2f]", agent.ci95_low, agent.ci95_high)
    _logger.info("    Best:          %.2f", agent.max_reward)
    _logger.info("    Worst:         %.2f", agent.min_reward)


def main() -> None:
    """Run the high-confidence checkpoint comparison."""
    args = _parse_args()
    sim_config = default_config()
    sim_config.ensure_directories()
    output_dir = ensure_directory(sim_config.evaluation_dir / OUTPUT_SUBDIR)

    _logger.info("=" * 60)
    _logger.info("Phase 10 — High-Confidence Checkpoint Comparison")
    _logger.info("=" * 60)
    _logger.info("Episodes/agent: %d | fixed seed base: %d", args.episodes, args.seed)

    eval_env = make_eval_env(sim_config, record_video=False)
    try:
        agents = _load_agents(sim_config.checkpoints_dir, eval_env)
        runner = BenchmarkRunner(
            eval_env,
            n_episodes=args.episodes,
            deterministic=True,
            success_threshold=args.success_threshold,
            seed=args.seed,
        )
        comparison = runner.run(agents)
    finally:
        eval_env.close()

    md_path, json_path = save_comparison(comparison, output_dir)
    by_name = {a.name: a for a in comparison.agents}
    csv_path = _save_episode_rewards(output_dir, args.seed, by_name)

    _logger.info("=" * 60)
    _logger.info("Comparison Results (identical settings, fixed seeds)")
    _logger.info("=" * 60)
    for agent in comparison.ranked():
        _report_line(agent)
    _logger.info("-" * 60)
    _logger.info("Winner by mean reward: %s", comparison.best_agent)
    _logger.info("Report:          %s", md_path)
    _logger.info("Per-episode CSV: %s", csv_path)
    _logger.info("=" * 60)

    # Also emit the Markdown table to the console for convenience.
    print("\n" + comparison_to_markdown(comparison))


if __name__ == "__main__":
    main()

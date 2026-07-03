"""Phase 6: Comprehensive evaluation of the best trained checkpoint.

Loads the best checkpoint and evaluates over multiple episodes, reporting:
  - Average Reward
  - Std Dev
  - Episode Length
  - Completion Rate (% of episodes exceeding the success threshold)

Run: python -m evaluate_best
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from agents.ppo_agent import PPOAgent
from config.rl import EvaluationConfig, PPOConfig
from config.simulator import default_config
from evaluation.evaluator import Evaluator
from simulator.environment_factory import make_eval_env
from utils.logger import get_logger

_logger = get_logger(__name__)

# Checkpoint directories from our training runs (most recent first)
CHECKPOINT_DIRS = [
    "data/checkpoints/ppo_1m",
    "data/checkpoints/ppo_500k",
    "data/checkpoints/ppo_100k",
]

N_EVAL_EPISODES = 20  # Don't rely on a single run — use 20 episodes


def find_best_checkpoint() -> Path:
    """Find the best.zip checkpoint from the most recent training run."""
    for ckpt_dir in CHECKPOINT_DIRS:
        best_path = Path(ckpt_dir) / "best.zip"
        if best_path.exists():
            return best_path
    raise FileNotFoundError(
        "No best.zip checkpoint found. Run a training script first."
    )


def main() -> None:
    sim_config = default_config()

    # --- Find and load the best checkpoint ---
    checkpoint_path = find_best_checkpoint()
    _logger.info("=" * 60)
    _logger.info("Phase 6 — Comprehensive Evaluation")
    _logger.info("=" * 60)
    _logger.info("Checkpoint: %s", checkpoint_path)
    _logger.info("Episodes:   %d", N_EVAL_EPISODES)

    # --- Build eval environment (no rendering, no video) ---
    eval_env = make_eval_env(sim_config, record_video=False)

    # --- Build a PPO agent and load the checkpoint ---
    ppo_config = PPOConfig()
    agent = PPOAgent(eval_env, config=ppo_config, verbose=0)
    agent.load(checkpoint_path)
    _logger.info("Model loaded successfully.")

    # --- Run evaluation ---
    eval_config = EvaluationConfig(
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        record_video=False,
        success_threshold=900.0,
    )

    evaluator = Evaluator(
        eval_env,
        n_episodes=eval_config.n_eval_episodes,
        deterministic=eval_config.deterministic,
        success_threshold=eval_config.success_threshold,
        seed=42,
    )

    result = evaluator.evaluate(agent)

    # --- Compute Phase 6 metrics ---
    rewards = np.array(result.episode_rewards)
    lengths = np.array(result.episode_lengths)
    std_dev = float(np.std(rewards))
    avg_reward = result.average_reward
    avg_length = result.average_length
    completion_rate = result.success_rate * 100.0

    # --- Report ---
    _logger.info("")
    _logger.info("=" * 60)
    _logger.info("EVALUATION RESULTS")
    _logger.info("=" * 60)
    _logger.info("")
    _logger.info("  %-25s %10.2f", "Average Reward", avg_reward)
    _logger.info("  %-25s %10.2f", "Std Dev", std_dev)
    _logger.info("  %-25s %10.1f", "Episode Length", avg_length)
    _logger.info("  %-25s %9.0f%%", "Completion Rate (≥900)", completion_rate)
    _logger.info("")
    _logger.info("  %-25s %10.2f", "Max Reward", result.max_reward)
    _logger.info("  %-25s %10.2f", "Min Reward", result.min_reward)
    _logger.info("  %-25s %10.2f", "Variance", result.reward_variance)
    _logger.info("")
    _logger.info("  Per-episode rewards:")
    for i, (r, l) in enumerate(zip(rewards, lengths)):
        marker = " ✓" if r >= 900 else ""
        _logger.info("    Episode %2d: reward=%+8.2f  length=%4d%s", i + 1, r, l, marker)
    _logger.info("")
    _logger.info("=" * 60)

    eval_env.close()


if __name__ == "__main__":
    main()

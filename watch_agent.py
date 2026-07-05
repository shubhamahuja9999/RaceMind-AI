"""Phase 8: Watch the Agent — record videos of the trained agent driving.

Records multiple episodes to video files so you can visually inspect:
  - Is the car accelerating?
  - Does it oversteer?
  - Does it hug the inside of corners?
  - Does it spin?
  - Does it oscillate left and right?

Run: python -m watch_agent

Videos are saved to data/videos/watch_agent/
"""

from __future__ import annotations

from pathlib import Path

from agents.ppo_agent import PPOAgent
from config.rl import PPOConfig
from config.simulator import default_config
from simulator.environment_factory import make_eval_env
from training.training_loop import run_episode
from utils.logger import get_logger

_logger = get_logger(__name__)

# Checkpoint directories (most recent first)
CHECKPOINT_DIRS = [
    "data/checkpoints/ppo_1m",
    "data/checkpoints/ppo_500k",
    "data/checkpoints/ppo_100k",
]

N_EPISODES = 5
VIDEO_DIR = Path("data/videos/watch_agent")


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

    checkpoint_path = find_best_checkpoint()
    _logger.info("=" * 60)
    _logger.info("Phase 8 -- Watch the Agent")
    _logger.info("=" * 60)
    _logger.info("Checkpoint: %s", checkpoint_path)
    _logger.info("Episodes:   %d", N_EPISODES)
    _logger.info("Video dir:  %s", VIDEO_DIR)

    # --- Build eval env WITH video recording ---
    eval_env = make_eval_env(
        sim_config,
        record_video=True,
        video_dir=VIDEO_DIR,
        name_prefix="agent",
    )

    # --- Load the trained agent ---
    ppo_config = PPOConfig()
    agent = PPOAgent(eval_env, config=ppo_config, verbose=0)
    agent.load(checkpoint_path)
    _logger.info("Model loaded successfully.")

    # --- Run episodes and record ---
    _logger.info("")
    _logger.info("Recording episodes...")
    for ep in range(N_EPISODES):
        outcome = run_episode(
            eval_env,
            agent,
            deterministic=True,
            seed=42 + ep,
        )
        _logger.info(
            "  Episode %d: reward=%+.2f  length=%d",
            ep + 1, outcome.total_reward, outcome.length,
        )

    eval_env.close()

    # --- List recorded videos ---
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    _logger.info("")
    _logger.info("=" * 60)
    _logger.info("Recorded %d videos in %s:", len(videos), VIDEO_DIR)
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        _logger.info("  -> %s (%.1f MB)", v.name, size_mb)
    _logger.info("=" * 60)
    _logger.info("")
    _logger.info("Open these videos to analyze agent behavior:")
    _logger.info("  - Is the car accelerating?")
    _logger.info("  - Does it oversteer?")
    _logger.info("  - Does it hug the inside of corners?")
    _logger.info("  - Does it spin?")
    _logger.info("  - Does it oscillate left and right?")


if __name__ == "__main__":
    main()

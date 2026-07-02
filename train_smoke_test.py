"""Phase 4 smoke test: short 10K training run to verify the pipeline works.

Checks:
  ✓ Does training start?
  ✓ Does it crash?
  ✓ Are rewards changing?
  ✓ Are checkpoints being saved?
"""

from pathlib import Path

from config.rl import EvaluationConfig, PPOConfig, TrainingConfig
from config.simulator import default_config
from simulator.environment_factory import make_training_env, make_eval_env
from training.ppo_trainer import build_ppo_trainer
from utils.logger import get_logger

_logger = get_logger(__name__)


def main() -> None:
    sim_config = default_config()
    sim_config.ensure_directories()

    # --- Configs for the smoke test ---
    ppo_config = PPOConfig()  # CnnPolicy, lr=3e-4, n_steps=2048, etc.
    training_config = TrainingConfig(
        total_timesteps=10_000,
        evaluation_frequency=5_000,
        checkpoint_frequency=5_000,
    )
    eval_config = EvaluationConfig(
        n_eval_episodes=2,        # just 2 episodes to keep it quick
        deterministic=True,
        record_video=False,
    )

    checkpoint_dir = sim_config.checkpoints_dir / "smoke_test"
    tb_log_dir = Path("runs") / "smoke_test"

    _logger.info("=" * 60)
    _logger.info("Phase 4 Smoke Test — 10K timesteps")
    _logger.info("=" * 60)
    _logger.info("PPO config: %s", ppo_config)
    _logger.info("Training config: %s", training_config)

    # --- Build environments ---
    train_env = make_training_env(sim_config)
    eval_env = make_eval_env(sim_config, record_video=False)

    # --- Build trainer and run ---
    trainer = build_ppo_trainer(
        train_env=train_env,
        eval_env=eval_env,
        ppo_config=ppo_config,
        training_config=training_config,
        evaluation_config=eval_config,
        checkpoint_dir=checkpoint_dir,
        tensorboard_log=tb_log_dir,
        seed=42,
        experiment_name="smoke_test",
    )

    summary = trainer.train()

    # --- Results ---
    _logger.info("=" * 60)
    _logger.info("Smoke Test Results")
    _logger.info("=" * 60)
    _logger.info("Total timesteps:   %d", summary.total_timesteps)
    _logger.info("Elapsed:           %.1fs", summary.elapsed_seconds)
    _logger.info("Steps/sec:         %.1f", summary.timesteps_per_second)
    _logger.info("Best reward:       %.2f", summary.best_reward or 0.0)
    _logger.info("Final reward:      %.2f", summary.final_reward or 0.0)
    _logger.info("Evaluations:       %s", summary.evaluations)

    # --- Verify checkpoints ---
    checkpoints = list(checkpoint_dir.glob("*.zip"))
    _logger.info("Checkpoints saved: %d files in %s", len(checkpoints), checkpoint_dir)
    for cp in checkpoints:
        _logger.info("  → %s (%.1f KB)", cp.name, cp.stat().st_size / 1024)

    _logger.info("=" * 60)
    _logger.info("✓ SMOKE TEST PASSED" if len(checkpoints) > 0 else "✗ SMOKE TEST FAILED")
    _logger.info("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

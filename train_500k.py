"""Phase 4: 500K timestep PPO training run.

Scaled up from the successful 100K run. Evaluates every 50K steps and saves
checkpoints. Expect this to take ~2.5-3 hours on CPU.
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

    ppo_config = PPOConfig()
    training_config = TrainingConfig(
        total_timesteps=500_000,
        evaluation_frequency=50_000,
        checkpoint_frequency=50_000,
    )
    eval_config = EvaluationConfig(
        n_eval_episodes=5,
        deterministic=True,
        record_video=False,
    )

    checkpoint_dir = sim_config.checkpoints_dir / "ppo_500k"
    tb_log_dir = Path("runs") / "ppo_500k"

    _logger.info("=" * 60)
    _logger.info("Phase 4 — PPO 500K Training Run")
    _logger.info("=" * 60)
    _logger.info("PPO config: %s", ppo_config)
    _logger.info("Training config: %s", training_config)

    train_env = make_training_env(sim_config)
    eval_env = make_eval_env(sim_config, record_video=False)

    trainer = build_ppo_trainer(
        train_env=train_env,
        eval_env=eval_env,
        ppo_config=ppo_config,
        training_config=training_config,
        evaluation_config=eval_config,
        checkpoint_dir=checkpoint_dir,
        tensorboard_log=tb_log_dir,
        seed=42,
        experiment_name="ppo_500k",
    )

    summary = trainer.train()

    _logger.info("=" * 60)
    _logger.info("500K Training Results")
    _logger.info("=" * 60)
    _logger.info("Total timesteps:   %d", summary.total_timesteps)
    _logger.info("Elapsed:           %.1fs (%.1f min)", summary.elapsed_seconds, summary.elapsed_seconds / 60)
    _logger.info("Steps/sec:         %.1f", summary.timesteps_per_second)
    _logger.info("Best reward:       %.2f", summary.best_reward or 0.0)
    _logger.info("Final reward:      %.2f", summary.final_reward or 0.0)
    _logger.info("Evaluations:")
    for step, reward in summary.evaluations:
        _logger.info("  step=%6d  reward=%.2f", step, reward)

    checkpoints = list(checkpoint_dir.glob("*.zip"))
    _logger.info("Checkpoints: %d files in %s", len(checkpoints), checkpoint_dir)
    for cp in checkpoints:
        _logger.info("  → %s (%.1f MB)", cp.name, cp.stat().st_size / (1024 * 1024))

    _logger.info("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

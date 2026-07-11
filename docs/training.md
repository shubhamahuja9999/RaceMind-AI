# Training

RaceMind AI trains PPO agents on `CarRacing-v3` through a small, algorithm-agnostic
training stack.

## Components

| Piece | Location | Role |
| --- | --- | --- |
| `BaseAgent` | `agents/base_agent.py` | The interface every agent implements (`act`, `predict`, `learn`, `save`, `load`). |
| `PPOAgent` | `agents/ppo_agent.py` | Wraps Stable-Baselines3 PPO behind `BaseAgent`. |
| `Trainer` | `training/trainer.py` | Trains in fixed-size chunks; evaluates and checkpoints between chunks; supports resume-continuity, plateau early-stopping, and a per-chunk hook. |
| `CheckpointManager` | `training/checkpoint_manager.py` | Saves `latest` + `best` (by mean eval reward) and numbered checkpoints. |
| Environment factory | `simulator/environment_factory.py` | Builds training/eval envs (single creation point). |

## How the trainer works

Training proceeds in chunks of `evaluation_frequency` steps. After each chunk the
trainer runs a short evaluation, checkpoints (`latest`, and `best` if improved),
and optionally stops early if the evaluation reward plateaus. This keeps a live
learning curve and always retains the best model — decoupled from any specific
algorithm because the trainer only calls `agent.learn(...)`.

```python
from simulator.environment_factory import make_training_env, make_eval_env
from agents.ppo_agent import PPOAgent
from config.rl import PPOConfig, TrainingConfig, EvaluationConfig
from training.trainer import build_trainer

train_env = make_training_env()
eval_env = make_eval_env()
agent = PPOAgent(train_env, config=PPOConfig(), seed=42)

trainer = build_trainer(
    agent, eval_env,
    training_config=TrainingConfig(total_timesteps=1_000_000,
                                   evaluation_frequency=100_000,
                                   checkpoint_frequency=100_000),
    evaluation_config=EvaluationConfig(n_eval_episodes=10),
    checkpoint_dir=..., seed=1000,
)
summary = trainer.train()
```

## Ready-made training scripts

| Script | Budget |
| --- | --- |
| `train_smoke_test.py` | tiny — pipeline check |
| `train_100k.py` / `train_500k.py` / `train_1m.py` | scale-up runs |
| `train_3m.py` | resume-and-continue with plateau early-stopping |
| `run_experiment_001.py` / `run_experiment_002.py` | controlled experiments |

## Configuration

Hyperparameters live in `configs/ppo.yaml` (`config/rl.py::PPOConfig`). The frozen
baseline uses `learning_rate=3e-4` (constant), `gamma=0.99`, `n_steps=2048`,
`batch_size=64`, `n_epochs=10`, `clip_range=0.2`, `ent_coef=0.005`, `CnnPolicy`.

## Monitoring

- **TensorBoard:** `tensorboard --logdir runs` (or an experiment's `logs/`).
- **Project log:** `logs/project.log`.
- **Learning curves:** generated to each experiment's `plots/` (training reward,
  evaluation reward, learning rate, entropy, FPS, loss).

> Training is CPU-heavy on this project (~8–20 h for 1M steps). Long runs are best
> launched in a normal terminal so they survive independently of any tool session.

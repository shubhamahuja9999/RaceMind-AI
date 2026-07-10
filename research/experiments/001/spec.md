## Experiment 001: lr_schedule_linear_decay

- **Hypothesis:** Linearly decaying the learning rate from 3e-4 to 0 over the additional 500k steps will stabilise late-stage PPO updates and yield equal-or-higher mean reward than the constant-LR baseline, avoiding the late-training regression seen with a constant rate.
- **Independent variable:** learning_rate schedule: constant 3e-4 -> linear decay 3e-4 to 0 (applied as a stepped schedule, updated each 50k-step evaluation chunk)
- **Expected outcome:** Improved or No significant change (more stable); not Regressed.
- **Baseline:** `baseline_ppo_1m`

**Controlled variables (held identical to baseline):**

| Variable | Value |
| --- | --- |
| `policy` | CnnPolicy |
| `gamma` | 0.99 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.005 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `environment` | CarRacing-v3 |
| `resume_checkpoint` | ppo_1m/best.zip |
| `additional_timesteps` | 500000 |
| `evaluation` | 30 deterministic episodes, seeds 1000-1029 |

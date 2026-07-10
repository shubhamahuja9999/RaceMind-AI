## Experiment 002: framestack_4

- **Hypothesis:** Providing the policy with four consecutive frames will improve driving performance because CarRacing is partially observable from a single frame.
- **Independent variable:** observation: single frame -> FrameStack(4) (temporal, channel-stacked)
- **Expected outcome:** Improved (reject the null hypothesis) if partial observability matters.
- **Baseline:** `baseline_ppo_1m`

**Controlled variables (held identical to baseline):**

| Variable | Value |
| --- | --- |
| `algorithm` | PPO (CnnPolicy) |
| `environment` | CarRacing-v3 |
| `learning_rate` | 0.0003 (constant) |
| `ent_coef` | 0.005 |
| `gamma` | 0.99 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `clip_range` | 0.2 |
| `gae_lambda` | 0.95 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `training_timesteps` | 1000000 |
| `initialisation` | from scratch (no checkpoint loaded) |
| `evaluation` | 30 deterministic episodes, seeds 1000-1029 |

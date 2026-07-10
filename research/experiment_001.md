# Experiment 001: lr_schedule_linear_decay

## Goal

Determine, under strictly controlled conditions, whether replacing the constant learning rate with a linear decay to zero improves the PPO agent's performance on `CarRacing-v3` relative to the frozen baseline.

## Hypothesis

Linearly decaying the learning rate from 3e-4 to 0 over the additional 500k steps will stabilise late-stage PPO updates and yield equal-or-higher mean reward than the constant-LR baseline, avoiding the late-training regression seen with a constant rate.

## Methodology

Training was resumed from `ppo_1m/best.zip` and continued for 500,000 additional timesteps. Exactly one variable was changed relative to the baseline (see below); every other hyperparameter, the environment, and the evaluation protocol were held identical. Evaluation used the frozen protocol: 30 deterministic episodes on fixed seeds (1000–1029), scored on mean reward with a paired t-test against the baseline (identical seeds enable pairing).

### Changed variable (independent)

- learning_rate schedule: constant 3e-4 -> linear decay 3e-4 to 0 (applied as a stepped schedule, updated each 50k-step evaluation chunk)

### Controlled variables (held identical to baseline)

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

## Training Duration

- Additional timesteps: 500,000
- Wall-clock: 5.06 h (303.5 min)

### Evaluation reward during training (every 50k steps)

| Timestep | Mean eval reward |
| --- | --- |
| 50,000 | 434.10 |
| 100,000 | 350.24 |
| 150,000 | 268.86 |
| 200,000 | 169.82 |
| 250,000 | 265.33 |
| 300,000 | 324.28 |
| 350,000 | 341.20 |
| 400,000 | -13.59 |
| 450,000 | 284.97 |
| 500,000 | 328.26 |

## Evaluation (best checkpoint, 30 deterministic episodes)

| Metric | Value |
| --- | --- |
| Mean | 434.10 |
| Median | 426.67 |
| Std | 160.65 |
| 95% CI | [374.12, 494.08] |
| Best | 792.59 |
| Worst | 112.34 |

## Comparison vs Baseline

| Metric | Baseline | Experiment |
| --- | --- | --- |
| Mean reward | 598.42 | 434.10 |
| 95% CI | [544.8, 652.1] | [374.1, 494.1] |

- **Absolute improvement:** -164.32
- **Percentage improvement:** -27.46%
- **Paired t-test** (n=30, df=29): t = -5.499, t-critical(95%) = 2.045
- **95% CIs overlap:** no
- **Statistically significant (p < 0.05):** yes

### Verdict: **Regressed**

## Interpretation

The linear learning-rate decay produced a statistically significant regression of -164.3 (-27.5%). Reducing the learning rate to zero appears to have frozen the policy before it could recover the baseline's performance, or the additional training under any schedule perturbed an already-good policy.

## Lessons Learned

Resuming a converged policy and decaying the LR to zero is not beneficial here. Future experiments should either train the schedule from scratch (not from a converged checkpoint) or decay to a small non-zero floor rather than exactly zero.

## Generated Plots

- `training_reward.png`
- `learning_rate.png`
- `entropy.png`
- `training_fps.png`
- `loss.png`
- `evaluation_reward.png`

## Future Work

Run Experiment 002 changing a different single variable (e.g. entropy coefficient, or enabling frame stacking) against this same frozen baseline, using the identical 30-episode paired protocol so results remain directly comparable.

# Experiment 002: framestack_4

## Abstract

CarRacing-v3 is partially observable from a single frame: velocity and angular momentum are not directly observable. This experiment tests whether providing four stacked consecutive frames (temporal observation) improves a PPO agent relative to the frozen single-frame baseline, changing only the observation representation and holding all other factors constant. The policy is trained from scratch for 1M steps and evaluated with a 30-episode paired protocol against the baseline.

## Hypothesis

Providing the policy with four consecutive frames will improve driving performance because CarRacing is partially observable from a single frame.

## Methodology

The policy was trained from scratch for 1,000,000 timesteps. Exactly one variable was changed relative to the frozen baseline (below); every other hyperparameter, the environment, and the evaluation protocol were held identical. Evaluation used the frozen protocol: 30 deterministic episodes on fixed seeds (1000-1029), compared to the baseline with a paired t-test (identical seeds enable pairing).

### Independent variable (changed)

- observation: single frame -> FrameStack(4) (temporal, channel-stacked)

### Controlled variables (held identical to baseline)

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

## Training

- Timesteps: 1,000,000
- Wall-clock: 22.94 h (1376.5 min)

### Evaluation reward during training

| Timestep | Mean eval reward |
| --- | --- |
| 100,000 | 169.62 |
| 200,000 | 185.05 |
| 300,000 | 169.07 |
| 400,000 | 10.41 |
| 500,000 | 74.90 |
| 600,000 | 278.16 |
| 700,000 | 282.11 |
| 802,464 | 338.47 |
| 902,464 | 373.72 |
| 1,000,000 | 411.07 |

## Evaluation (best checkpoint, 30 deterministic episodes)

| Metric | Value |
| --- | --- |
| Mean | 411.07 |
| Median | 411.87 |
| Std | 177.19 |
| 95% CI | [344.91, 477.23] |
| Best | 829.33 |
| Worst | 124.83 |

## Results & Statistical Analysis

## Comparison vs Baseline

| Metric | Baseline | Experiment |
| --- | --- | --- |
| Mean reward | 598.42 | 411.07 |
| 95% CI | [544.8, 652.1] | [344.9, 477.2] |

- **Absolute improvement:** -187.35
- **Percentage improvement:** -31.31%
- **Paired t-test** (n=30, df=29): t = -5.753, t-critical(95%) = 2.045
- **Effect size (Cohen's d):** -1.161 (large)
- **95% CIs overlap:** no
- **Statistically significant (p < 0.05):** yes

### Verdict: **Regressed**

## Interpretation

FrameStack(4) produced a statistically significant regression. The 4x larger observation (12 channels) increases the input dimensionality and, within the fixed 1M-step budget and unchanged hyperparameters, the policy learned more slowly rather than benefiting from temporal information. The hypothesis is rejected under these conditions.

## Lessons Learned

Adding input channels is not free: it can slow convergence at a fixed budget. Frame stacking should be paired with grayscale (to keep channel count comparable) and/or a larger training budget before judging it.

## Generated Plots

- `training_reward.png`
- `learning_rate.png`
- `entropy.png`
- `training_fps.png`
- `loss.png`
- `evaluation_reward.png`

## Future Work

Re-run FrameStack with grayscale (4 stacked grayscale frames = 4 channels, comparable to the baseline's 3) to isolate temporal information from the channel-count increase, as Experiment 003.

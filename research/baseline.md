# RaceMind AI — Frozen Baseline

> **This document is immutable. It defines the permanent reference point against
> which all controlled experiments are compared. Do not edit or overwrite it,
> and never overwrite the baseline checkpoint.**

## Model

| Field | Value |
| --- | --- |
| Name | `baseline_ppo_1m` |
| Checkpoint | `data/checkpoints/ppo_1m/best.zip` |
| Algorithm | PPO (Stable-Baselines3), `CnnPolicy` |
| Environment | `CarRacing-v3` (continuous, `max_episode_steps=1000`) |

## Hyperparameters

| Parameter | Value |
| --- | --- |
| learning_rate | `3e-4` (**constant**) |
| gamma | `0.99` |
| n_steps | `2048` |
| batch_size | `64` |
| n_epochs | `10` |
| gae_lambda | `0.95` |
| clip_range | `0.2` |
| ent_coef | `0.005` |
| vf_coef | `0.5` |
| max_grad_norm | `0.5` |

## Training

- **Total timesteps:** 1,000,000
- **Best checkpoint at:** 900,000 steps (evaluation reward peaked, then the run
  showed the classic late-PPO variance dip toward 1M).

## Evaluation Protocol

- **30 deterministic episodes**
- **Fixed seeds:** episode *i* uses seed `1000 + i` (seeds 1000–1029)
- Identical environment, deterministic policy (`predict(deterministic=True)`)

## Performance

| Metric | Value |
| --- | --- |
| Mean reward | **598.42** |
| Median reward | 613.73 |
| Std deviation | 143.66 |
| **95% confidence interval** | **[544.78, 652.05]** |
| Best | 816.33 |
| Worst | 296.23 |

The full 30 per-episode rewards are frozen in `research/baseline.json` so that
future experiments can run a **paired** comparison on identical seeds.

## Known Limitations

- **Not solved:** 0% success at the 900-reward threshold; the best single
  episode reached 816. The agent completes laps only partially/inconsistently.
- **High variance:** std ≈ 144 over 30 episodes; individual runs range from 296
  to 816.
- **Late-training instability:** continued training past 900k (the 1M→3M
  extension) *regressed* performance (see `experiment` archive), indicating the
  constant learning rate is too high late in training — the motivation for
  Experiment 001.
- **CPU-trained:** low throughput; results reflect a modest compute budget.

---
*Frozen on establishment of the baseline. Reference only.*

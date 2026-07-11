# Project Assets — Navigation Guide

Where to find each kind of artifact. Generated binaries (model checkpoints `.zip`,
videos `.mp4`, TensorBoard event files) are **git-ignored** to keep the repository
lightweight — they are produced locally when you train. The small, human-readable
reports and plot PNGs are committed.

## Research reports (committed)

| Report | Location |
| --- | --- |
| Frozen baseline | [`research/baseline.md`](../research/baseline.md) |
| Experiment timeline | [`research/experiment_timeline.md`](../research/experiment_timeline.md) |
| Results summary (tables) | [`research/results_summary.md`](../research/results_summary.md) |
| Final research report | [`research/final_report.md`](../research/final_report.md) |
| Per-experiment papers | `research/experiments/<id>/experiment_*.md` |
| Reward framework how-to | [`research/reward_framework.md`](../research/reward_framework.md) |
| Reward ablation report | `research/final_reward_ablation.md` (generated after reward runs) |

## Learning curves & plots (committed PNGs)

`research/experiments/<id>/plots/` — `training_reward.png`, `evaluation_reward.png`,
`learning_rate.png`, `entropy.png`, `training_fps.png`, `loss.png`.

## Reward-breakdown plots (committed PNGs)

`research/reward_analysis/<name>/` — `reward_total.png`, `reward_components.png`,
`reward_contribution_pct.png` (plus a per-component `reward_log.csv`, git-ignored).

## Model cards (committed JSON)

`research/experiments/<id>/checkpoints/model_card.json` — algorithm, environment,
hyperparameters, metrics, training date, git commit.

## Model checkpoints (git-ignored — generated locally)

`research/experiments/<id>/checkpoints/` and `data/checkpoints/<run>/` —
`best.zip`, `latest.zip`, numbered checkpoints. The frozen baseline is
`data/checkpoints/ppo_1m/best.zip`.

## Videos (git-ignored — generated locally)

`research/experiments/<id>/videos/` — `random-*.mp4`, `best-*.mp4` (and, for some
experiments, early/mid stages). Regenerate with `watch_agent.py` or an experiment run.

## TensorBoard logs (git-ignored)

`runs/` or `research/experiments/<id>/logs/` — view with `tensorboard --logdir runs`.

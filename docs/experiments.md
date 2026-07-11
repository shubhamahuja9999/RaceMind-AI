# Experiments

RaceMind AI treats every training run as a **controlled experiment**: exactly one
independent variable changes, everything else matches the frozen baseline, and the
result is judged by a paired statistical test.

## The method

1. **Freeze a baseline.** `research/baseline.md` documents the reference model and
   its performance (mean 598.42, 95% CI [544.78, 652.05]) and is never overwritten.
2. **Change one variable.** An `ExperimentSpec` records the hypothesis, the
   independent variable, the controlled variables and the expected outcome.
3. **Train and evaluate** with the fixed 30-episode protocol.
4. **Compare** against the baseline (paired t-test + Cohen's d) and emit a verdict.
5. **Archive** everything: spec, checkpoints, learning curves, model card,
   evaluation, comparison, videos, and an auto-generated research paper.

## Running an experiment

```bash
# A controlled RL experiment (e.g. the learning-rate schedule study)
python run_experiment_001.py

# A reward ablation (only the reward configuration changes)
python -m research.run_reward_experiment --config configs/reward_smooth_steering.yaml \
    --timesteps 1000000 --eval-frequency 100000 --eval-episodes 30

# Aggregate all reward experiments into one report
python -m research.generate_reward_ablation_report
```

Artifacts land in `research/experiments/<name>/`; reward-analysis plots in
`research/reward_analysis/<name>/`.

## Completed experiments

| # | Name | Independent variable | Verdict |
| --- | --- | --- | --- |
| — | Baseline PPO | — (reference) | reference (598.42) |
| — | Continue Training | +1M→3M continued steps | **Regressed** |
| 001 | Learning-Rate Schedule | constant → linear decay LR | **Regressed** |
| 002 | RGB FrameStack | 1 frame → 4 stacked frames | **Regressed** |

Full numbers are in [`research/results_summary.md`](../research/results_summary.md)
and the per-experiment papers in `research/experiments/<n>/`.

## Reward ablations (framework ready)

The reward research framework supports reward-only ablations. Two are staged:
`configs/reward_smooth_steering.yaml` (baseline + SmoothSteering) and
`configs/reward_smooth_idle.yaml` (+ IdlePenalty). See
[`reward_framework.md`](reward_framework.md).

## Honesty policy

Negative results are reported as first-class outcomes. If a change does not produce
a statistically significant improvement, the hypothesis is **explicitly rejected**
in the experiment's paper and in the results summary. Nothing is cherry-picked.

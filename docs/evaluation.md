# Evaluation & Benchmarking

Evaluation in RaceMind AI is deliberately rigorous and **fixed**, so that every
experiment is directly comparable to the frozen baseline.

## The frozen evaluation protocol

- **30 deterministic episodes** (`predict(deterministic=True)`).
- **Fixed seeds** — episode *i* uses seed `1000 + i` (seeds 1000–1029).
- **Native task reward** — reward shaping is never applied at evaluation time.
- Identical environment configuration to the baseline.

This protocol is treated as immutable. It is what makes a paired statistical
comparison valid: because both the baseline and an experiment are scored on the
same 30 seeds, differences are attributable to the change under test.

## Statistics

`evaluation/statistics.py` computes, from the 30 per-episode rewards: mean, median,
sample standard deviation, standard error, and a **Student's-t 95% confidence
interval** (no SciPy dependency), plus best and worst.

`evaluation/evaluator.py` runs the protocol; `evaluation/benchmark_runner.py`
compares multiple agents and reports the same statistics with the 95% CI.

## Comparison against the baseline

`research/comparison.py` compares an experiment's 30 rewards against the frozen
baseline (`research/baseline.json`) with a **paired t-test** (identical seeds ⇒
pairing), **Cohen's d** effect size, absolute and percentage change, and a verdict:

- **Improved** — statistically significant gain (paired *t* beyond the 95% critical value)
- **No significant change** — the null cannot be rejected
- **Regressed** — statistically significant loss

Non-overlapping 95% confidence intervals are reported as a conservative secondary
check.

## Artifacts produced

- **Model card** (`model_card.json`) — algorithm, environment, hyperparameters,
  metrics, training date, git commit.
- **Evaluation report** (`evaluation.json`) and **comparison** (`comparison.md` / `.json`).
- **Learning curves** and, for reward experiments, **reward-breakdown plots**.
- **Videos** of the trained policy.

## Standalone tools

```bash
python evaluate_best.py                 # evaluate the best checkpoint
python compare_checkpoints.py           # high-confidence head-to-head (30 episodes)
python plot_learning_curves.py          # regenerate learning-curve plots
python watch_agent.py                   # record a video of a trained agent
```

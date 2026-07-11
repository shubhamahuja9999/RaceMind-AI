# RaceMind AI: A Reproducible Platform for Controlled Reinforcement-Learning Experiments on CarRacing-v3

*Technical report — Version 1.0*

## Abstract

We present RaceMind AI, a reproducible reinforcement-learning research platform
built around Gymnasium's `CarRacing-v3`. Rather than chasing a single high score,
the project's contribution is methodological: a modular system in which one
independent variable can be changed at a time, trained under identical conditions,
and evaluated against a frozen baseline with a paired statistical test. Using PPO
(Stable-Baselines3) we establish a baseline of mean reward **598.42** (95% CI
[544.78, 652.05]) over 30 deterministic episodes. We then run three controlled
experiments — continued training, a learning-rate schedule, and RGB frame stacking
— and find that **all three regress** relative to the baseline (Δ ranging −27% to
−31%, all statistically significant). We report these negative results honestly and
analyse their causes. A modular reward-shaping framework is implemented and verified
for future ablations.

## Problem statement

`CarRacing-v3` is a continuous-control, pixel-based driving task that is challenging
on a limited (CPU) compute budget. Beyond training a competent agent, we ask a
methodological question: **can we measure, rigorously and reproducibly, whether a
given change actually improves performance?** Informal single-run comparisons are
unreliable given PPO's high variance, so we build the infrastructure to answer such
questions with statistical discipline.

## Methodology

- **Frozen baseline.** One PPO model is designated the permanent reference and never
  overwritten. Every experiment compares against it.
- **One variable per experiment.** An experiment specification records the
  hypothesis, the single independent variable, the controlled variables, and the
  expected outcome; everything else matches the baseline.
- **Fixed evaluation protocol.** 30 deterministic episodes on fixed seeds
  (1000–1029), scored on the native task reward. Because the seeds are shared, we
  use a **paired t-test** (more powerful than unpaired) and report **Cohen's d** and
  non-overlapping 95% confidence intervals.
- **Verdict.** Each experiment yields exactly one of *Improved*, *No significant
  change*, or *Regressed*.

## System design

The platform is layered (simulator → environment factory → wrappers → optional
reward framework → agent → trainer → evaluation → benchmarking → reports). Training
touches agents only through a `BaseAgent` interface, and evaluation is decoupled from
training (native reward always), which is what keeps experiments comparable. See
[`../docs/architecture.md`](../docs/architecture.md).

## Experiments and results

| Experiment | Independent variable | Mean | Δ vs 598.42 | Verdict |
| --- | --- | ---: | ---: | :---: |
| Continue Training | +1M→3M continued steps | 420.94 | −29.7% | Regressed |
| Learning-Rate Schedule | constant → linear-decay LR | 434.10 | −27.5% | Regressed |
| RGB FrameStack | 1 → 4 stacked frames | 411.07 | −31.3% | Regressed |

## Statistical analysis

All three experiments are statistically significant regressions at the 95% level.
For RGB FrameStack, paired *t* = −5.753 (df=29) with **Cohen's d = −1.16 (large)**;
for the learning-rate schedule, paired *t* = −5.499; for continued training, the 95%
confidence intervals do not overlap. In no case did the experiment's interval reach
the baseline's lower bound of 544.8.

## Discussion

- **Continued training** degraded a converged policy: at a constant full learning
  rate, further updates drifted the policy away from its good solution — the classic
  late-PPO instability.
- **The learning-rate schedule** regressed for the same underlying reason: the drop
  appeared in the first post-resume evaluation, *before* decay took effect, so the
  cause is continuation, not the schedule. Decaying the LR did not rescue it.
- **Frame stacking** is the most nuanced case. It quadrupled the observation to 12
  channels and trained ~5× slower; the evaluation reward was *still rising* at 1M
  steps, indicating the agent was **undertrained** at the fixed budget. The result is
  confounded by channel count and compute, so the temporal-information hypothesis was
  not cleanly isolated.

## Negative results

We state these plainly: **no tested change improved on the baseline.** Every
hypothesis was rejected. We consider this a valuable outcome — the experiments rule
out "just train longer," "just decay the LR," and "just add frames (naively)" as
easy wins on this budget, and they demonstrate that the platform detects regressions
reliably rather than rationalising them away.

## Limitations

- **Single seed / single run per configuration** — no across-seed variance is
  estimated; each verdict reflects one training trajectory.
- **Compute-limited** — CPU training caps the budget; effects requiring longer
  training (e.g. frame stacking) may be undetectable here.
- **Confounds where noted** — frame stacking mixed temporal information with channel
  count.
- **Reward components** — `speed`/`centerline`/`collision` are proxies/placeholders
  where the environment does not expose the underlying signal.

## Future work

- Re-run key configurations across several seeds to estimate variance and increase
  statistical power.
- Isolate the temporal hypothesis with **grayscale + FrameStack(4)** (4 channels,
  comparable to the baseline's 3), removing the channel-count confound.
- Complete the staged reward ablations (SmoothSteering, +IdlePenalty) at full budget.
- Train on a GPU to lift the compute ceiling and revisit whether any change wins with
  more steps.

## Reproducibility

Global seeding, an immutable frozen baseline, archived specs and model cards, and a
paired statistical comparison for every run. All experiment archives (specs,
comparisons, learning curves, papers) are in `research/experiments/`.

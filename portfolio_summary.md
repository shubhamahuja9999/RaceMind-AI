# RaceMind AI — Portfolio Summary

> A reproducible reinforcement-learning research platform for autonomous racing
> (`CarRacing-v3`), built to test — with statistical rigor — whether a change
> actually improves an RL agent.

## Overview

RaceMind AI is an end-to-end RL research platform: a modular simulator, a PPO
training stack, a benchmarking and evaluation suite, a controlled-experiment
framework, and a modular reward-shaping framework. It was built incrementally
(Phases 1 → release) as a from-scratch software-engineering and machine-learning
project, prioritising clean architecture, reproducibility, and honest science over
leaderboard chasing.

## Technical highlights

- **Layered, interface-driven architecture** — simulator → wrappers → reward
  framework → agent → trainer → evaluation → benchmarking → reports. Training depends
  only on a `BaseAgent` interface, so algorithms and rewards are pluggable.
- **PPO training pipeline** (Stable-Baselines3) with a custom algorithm-agnostic
  chunked trainer: periodic evaluation, best/latest checkpointing, resume-continuity,
  and plateau early-stopping.
- **Rigorous evaluation** — a fixed 30-episode, fixed-seed protocol with 95%
  confidence intervals, **paired t-tests**, and **Cohen's d** effect sizes; automatic
  *Improved / No change / Regressed* verdicts against a frozen baseline.
- **Controlled-experiment framework** — one independent variable per experiment,
  auto-generated specs, model cards, learning curves, videos, and research papers.
- **Modular reward framework** — YAML-configurable, composable reward components with
  per-component logging and visualisation; reward shaping is training-only so
  evaluation stays baseline-comparable.
- **Reproducibility engineering** — global seeding, immutable baseline, structured
  archives, TensorBoard logging, and JSON + Markdown reports.
- **Scale:** ~10 Python packages, YAML-driven configuration, and a documented `docs/`
  suite — roughly a small research codebase, not a single script.

## Challenges solved

- **PPO variance vs. reliable conclusions** — solved with a paired, fixed-seed
  evaluation protocol and confidence intervals so single-run noise doesn't drive
  decisions.
- **Frame stacking broke the CNN policy** — Gymnasium's `FrameStackObservation`
  produced a 4-D observation that SB3's `CnnPolicy` rejects; diagnosed and fixed with
  channel-wise stacking `(H,W,C·N)` while preserving the single-env evaluation API.
- **Long CPU training vs. session limits** — added resume-safe checkpointing and
  documented a reliable terminal workflow for multi-hour runs.
- **Serialisable learning-rate schedule** — a schedule closing over the model broke
  SB3's checkpoint pickling; solved with a picklable per-chunk schedule via a generic
  trainer hook.

## Research findings (honest)

Established a frozen PPO baseline of **mean reward 598.42** (95% CI [544.78, 652.05]).
Ran three controlled experiments — continued training, learning-rate decay, and RGB
frame stacking — and found that **all three statistically regressed** (−27% to −31%).
Each negative result was analysed (converged-policy drift; a continuation confound;
an under-training/channel-count confound for frame stacking) and reported without
spin. The platform reliably detects regressions, which is the point.

## Lessons learned

- Rigorous, reproducible **methodology** is more valuable than a lucky high score.
- "Obvious" improvements (train longer, decay LR, add frames) are not free and can
  regress under a fixed compute budget.
- Negative results, clearly reported, are a legitimate and useful contribution.
- Decoupling evaluation from training (native reward, fixed seeds) is what makes any
  comparison trustworthy.

## Impact

A clean, documented, publishable RL research codebase demonstrating end-to-end ML
engineering: environment design, training pipelines, statistical evaluation,
experiment design, reward engineering, and honest scientific reporting — a template
for running controlled RL studies rather than one-off training scripts.

## Tech stack

Python 3.12 · Gymnasium (Box2D) · Stable-Baselines3 · PyTorch · NumPy · Pandas ·
Matplotlib · TensorBoard · PyYAML · PyGame · OpenCV.

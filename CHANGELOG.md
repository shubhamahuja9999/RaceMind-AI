# Changelog

All notable changes to RaceMind AI are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [1.0.0] — 2026-07-11

First public release: a complete, reproducible reinforcement-learning research
platform built around Gymnasium's `CarRacing-v3`.

### Added
- **Simulator framework** — OO `CarRacing-v3` wrapper, environment factory,
  manual driving, telemetry, episode recording and standalone replay.
- **Research infrastructure** — YAML-driven configuration, professional logging,
  global seeding, structured data layout, experiment specs and callbacks.
- **PPO training framework** — pluggable `BaseAgent` interface, `PPOAgent`
  (Stable-Baselines3), an algorithm-agnostic chunked trainer with early-stopping
  and resume support, checkpoint management, TensorBoard logging.
- **Evaluation & benchmarking** — multi-episode evaluator, statistics with 95%
  confidence intervals, multi-agent benchmarks, model cards, learning-curve
  plots, and JSON + Markdown experiment reports.
- **Controlled-experiment layer** — frozen baseline, experiment specifications,
  automatic paired t-test comparison with Cohen's d and a verdict, and
  auto-generated research papers.
- **Reward research framework** — modular, YAML-configurable reward components,
  a reward manager, an opt-in shaping wrapper, per-component logging and plots,
  and a reward-ablation runner.
- **Experiments** — baseline PPO, continued training, learning-rate schedule and
  RGB frame stacking, each with a full archive and statistical comparison.
- **Documentation** — comprehensive README, `docs/` subsystem guides, research
  timeline, results summary, final research report, and portfolio summary.

### Notes
- Every controlled deviation tested so far (continued training, LR decay, frame
  stacking) **regressed** against the frozen baseline (mean 598.42) at the fixed
  CPU budget. These negative results are documented honestly; the platform's
  value is enabling the rigorous test, not guaranteeing an improvement.
- Reward ablation experiments (SmoothSteering, +IdlePenalty) are implemented and
  verified but await full-budget runs.

### Known limitations
- Single-seed, single-run per configuration; no across-seed variance estimates.
- CPU-only training budgets limit the scale of any observable effect.
- `speed`/`centerline`/`collision` reward components are proxies/placeholders
  where `CarRacing-v3` does not expose the underlying signal.

[1.0.0]: https://github.com/shubhamahuja9999/RaceMind-AI/releases/tag/v1.0.0

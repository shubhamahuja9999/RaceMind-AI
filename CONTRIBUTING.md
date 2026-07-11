# Contributing to RaceMind AI

Thanks for your interest in RaceMind AI. This is a research-oriented reinforcement
learning project, so contributions that improve reproducibility, rigor, and clarity
are especially welcome.

## Ground rules

RaceMind AI is a **controlled-experiment platform**. The most important rule is
scientific discipline:

- **Change one independent variable per experiment.** Everything else must match
  the frozen baseline (see [`research/baseline.md`](research/baseline.md)).
- **Never overwrite the frozen baseline** (`data/checkpoints/ppo_1m/best.zip`).
- **Evaluate on the native task reward** with the fixed 30-episode protocol
  (seeds 1000–1029) so results stay comparable.
- **Report negative results honestly.** A rejected hypothesis is a valid result.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Code style

- Python 3.12+, PEP 8, type hints on public functions, docstrings on public APIs.
- Small, focused functions; composition over inheritance.
- No new RL algorithms, wrappers, or reward components unless discussed first —
  the framework is intentionally stable.

## Adding a reward component

See [`docs/reward_framework.md`](docs/reward_framework.md). In short: subclass
`BaseRewardComponent`, register it, add it to a YAML config. No other code changes.

## Running an experiment

```bash
# Controlled RL experiment (change one variable)
python run_experiment_001.py            # example: learning-rate schedule

# Reward ablation (change one reward config)
python -m research.run_reward_experiment --config configs/reward_smooth_steering.yaml
```

Full training runs are CPU-heavy (hours); a small smoke budget is the default for
the reward runner. See [`docs/experiments.md`](docs/experiments.md).

## Pull requests

- Keep PRs focused and describe the hypothesis/variable if it is an experiment.
- Include the evaluation numbers (mean, 95% CI, paired-t, verdict) for any run.
- Ensure `python -m compileall <changed packages>` passes and imports resolve.

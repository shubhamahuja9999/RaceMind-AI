# Reward Framework

A modular framework for **controlled reward-function experiments**. Reward is
composed from independent, individually-configurable components so reward designs
can be tested one change at a time against the frozen baseline. It is
infrastructure — not an attempt to hand-craft a "better" reward.

## Composition

```
total = Σ  weightᵢ · componentᵢ(context)
```

- **`reward/base_reward.py`** — `RewardContext` (per-step inputs), `RewardResult`
  (total + per-component breakdown), `BaseRewardComponent` (the contract).
- **`reward/components/`** — `progress`, `speed`, `centerline`, `smooth_steering`,
  `offtrack`, `collision`, `idle`, plus a name→class registry. Each returns **only
  its own** contribution and knows nothing about the others.
- **`reward/reward_config.py`** — YAML-driven `enabled` / `weight` per component.
- **`reward/reward_manager.py`** — instantiates enabled components, applies weights,
  sums, returns a `RewardResult`.
- **`reward/reward_shaping.py`** — `RewardShapingWrapper`, the integration point;
  applied to **training only**.
- **`reward/reward_logger.py`** / **`reward/reward_plots.py`** — per-component CSV
  logging and visualisation.

## Key properties

- **Disabled components contribute exactly zero** (never instantiated).
- **`configs/reward_baseline.yaml` reproduces the baseline reward exactly**
  (progress only, weight 1.0 = the environment's native reward).
- **Evaluation uses the native reward**, so a reward experiment's score is directly
  comparable to the baseline — a shaped reward that scores its own objective well
  can still drive worse on the real task.
- **Opt-in** — default training does not use reward shaping, so the baseline is
  unaffected.

## Adding a component

1. Create `reward/components/<name>.py` subclassing `BaseRewardComponent`
   (`name = "<name>"`, implement `compute_reward(context) -> float`).
2. Register it in `reward/components/__init__.py`.
3. Reference it in a reward YAML with `enabled` and `weight`.

No other code changes are required.

## Running a reward experiment

```bash
python -m research.run_reward_experiment --config configs/reward_smooth_steering.yaml
```

> The detailed how-to (including the full component reference and interpretation
> notes) lives in [`../research/reward_framework.md`](../research/reward_framework.md).

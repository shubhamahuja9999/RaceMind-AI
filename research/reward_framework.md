# RaceMind AI — Reward Research Framework

A modular framework for **controlled reward-function experiments**. Reward is
composed from independent, individually-configurable components so that reward
designs can be tested scientifically — one change at a time — against the frozen
baseline. It is infrastructure, not an attempt to find a "better" reward.

## Architecture

```
                       configs/*.yaml  (enabled + weight per component)
                                │  load
                                ▼
                        RewardConfig ──► RewardManager
                                            │ instantiates enabled components
                                            │ from COMPONENT_REGISTRY
   env step ──► RewardShapingWrapper ──────►│ compute(context)
   (base_reward, action, obs, info)         │   Σ  weightᵢ · componentᵢ
                                            ▼
                                       RewardResult
                                   { total, weighted{}, raw{} }
                                            │
        training reward = total ◄───────────┘   (breakdown → info, for logging)
```

- **`reward/base_reward.py`** — `RewardContext` (per-step inputs), `RewardResult`
  (total + per-component breakdown), and `BaseRewardComponent` (the contract).
- **`reward/components/`** — one class per component (`progress`, `speed`,
  `centerline`, `smooth_steering`, `offtrack`, `collision`, `idle`) plus a
  name→class `COMPONENT_REGISTRY`. Each returns **only its own** raw contribution
  and knows nothing about the others.
- **`reward/reward_config.py`** — `RewardConfig` loaded from YAML (`enabled`,
  `weight` per component).
- **`reward/reward_manager.py`** — instantiates enabled components, applies
  weights, sums, returns `RewardResult`.
- **`reward/reward_shaping.py`** — `RewardShapingWrapper`: the integration point.
  Replaces the env reward with the composed reward and exposes the breakdown in
  `info`. Applied to **training only**.
- **`reward/reward_logger.py`** / **`reward/reward_plots.py`** — per-component
  logging (CSV) and visualisation.

## Reward composition

`total = Σ weightᵢ · componentᵢ(context)`

- Components return raw contributions; penalties return negative values, so a
  positive weight applies the penalty scale.
- **Disabled components are never instantiated**, so they contribute exactly
  zero and never appear in the breakdown.
- **Baseline reproduction:** with `progress` enabled at weight 1.0 and everything
  else disabled (`configs/reward_baseline.yaml`), the composed reward equals the
  environment's native reward exactly — i.e. the frozen baseline.

## Integration points

- `simulator.wrappers.wrap_environment(..., reward_manager=...)` inserts
  `RewardShapingWrapper` **before** `Monitor` (so logged returns reflect the
  shaped reward).
- `simulator.environment_factory.make_training_env(..., reward_manager=...)`
  forwards it. **`make_eval_env` has no such parameter** — evaluation always uses
  the native task reward, keeping every comparison honest and baseline-relative.
- Default training (`reward_manager=None`) is unchanged, so existing PPO training
  and the frozen baseline are unaffected.

## How to add a new reward component

1. Create `reward/components/<name>.py` with a `BaseRewardComponent` subclass:
   set `name = "<name>"` and implement `compute_reward(context) -> float`
   (return only your own raw contribution; read `context`, never other
   components).
2. Register it in `reward/components/__init__.py` (`COMPONENT_REGISTRY`).
3. Add it to a reward YAML with `enabled` and `weight`.

No other code changes are required.

## How to run a reward experiment

```bash
# Baseline control (progress only) — reproduces the frozen baseline reward:
python -m research.run_reward_experiment --config configs/reward_baseline.yaml

# An ablation (baseline + centerline) — only the reward changes:
python -m research.run_reward_experiment --config configs/reward_centerline.yaml
```

The default budget is a **small smoke value** (`--timesteps 5000`); pass
`--timesteps 1000000` for a full run. Artifacts are written to
`research/experiments/<name>/` (spec, checkpoints, comparison, model card, paper)
and reward analysis to `research/reward_analysis/<name>/` (per-component CSV +
plots).

## How to interpret results

- Evaluation is always on the **native task reward** (30 deterministic episodes,
  seeds 1000–1029), so a reward experiment's number is directly comparable to the
  baseline's 598.42.
- The comparison reports mean/median/std/95% CI, a **paired t-test**, **Cohen's
  d**, and a verdict: `Improved` / `No significant change` / `Regressed`.
- The reward-analysis plots (`reward_total`, `reward_components`,
  `reward_contribution_pct`) show how each component contributed during a
  representative episode — useful for sanity-checking that a component behaves as
  intended, independent of the final score.
- **Caveat:** a shaped reward that scores lower on the native reward is not
  necessarily "worse" — it optimised a different objective. Draw conclusions only
  from the native-reward comparison, and only at a real (non-smoke) budget.

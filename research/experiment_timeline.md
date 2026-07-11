# Experiment Timeline

A chronological, honest record of every completed experiment. Rejected hypotheses
are marked as such. All comparisons are against the frozen baseline (mean **598.42**,
95% CI [544.78, 652.05]) using the fixed 30-episode protocol (seeds 1000–1029).

---

## 0. Baseline PPO — *reference*

- **Role:** establishes the frozen reference point; never overwritten.
- **Setup:** PPO + `CnnPolicy`, single-frame observation, `lr=3e-4` constant,
  `ent_coef=0.005`, `n_steps=2048`, 1,000,000 steps.
- **Result:** mean **598.42**, 95% CI [544.78, 652.05], best single episode 816.
  (A single mid-run evaluation peaked at ~618 near 900k steps.)
- **Notes:** 0% "success" at the 900-reward threshold — the agent drives
  competently but does not solve the track. This is the bar every experiment is
  measured against.

---

## 1. Continue Training — **hypothesis rejected**

- **Hypothesis:** training the converged 1M model further (→3M) will improve it.
- **Independent variable:** additional training steps (continuation from `ppo_1m`).
- **Result:** best checkpoint mean **420.94**, 95% CI [358.43, 483.45].
- **Statistical conclusion:** **Regressed** — Δ −177.48 (−29.7%); the experiment's
  95% CI does not overlap the baseline's. Continuation made the agent *worse*.
- **Lessons learned:** more steps ≠ better. Continuing to train a converged PPO
  policy at the full constant learning rate perturbed it away from its good
  solution; the well-known late-PPO variance manifested as a downward drift.

---

## 2. Learning-Rate Schedule (Experiment 001) — **hypothesis rejected**

- **Hypothesis:** a linear LR decay (3e-4 → 0) stabilises late training and yields
  equal-or-better performance than the constant-LR baseline.
- **Independent variable:** LR schedule (constant → linear decay), resumed from the
  1M checkpoint for +500k steps. Everything else identical.
- **Result:** best checkpoint mean **434.10**, 95% CI [374.12, 494.08].
- **Statistical conclusion:** **Regressed** — Δ −164.32 (−27.46%), paired
  *t* = −5.499 (df=29, critical 2.045); non-overlapping CIs.
- **Lessons learned:** the regression appeared in the *first* evaluation after
  resuming — before decay had meaningful effect — so it is driven by **continuation
  from a converged checkpoint**, not by the schedule. You cannot improve this
  baseline by continuing to train it under any LR schedule.

---

## 3. RGB FrameStack (Experiment 002) — **hypothesis rejected (confounded)**

- **Hypothesis:** four stacked frames improve driving because CarRacing is partially
  observable from a single frame (velocity/heading rate are not directly visible).
- **Independent variable:** observation — single frame → FrameStack(4), trained from
  scratch for 1,000,000 steps.
- **Result:** best checkpoint mean **411.07**, 95% CI [344.91, 477.23].
- **Statistical conclusion:** **Regressed** — Δ −187.35 (−31.31%), paired
  *t* = −5.753, **Cohen's d = −1.16 (large)**; non-overlapping CIs.
- **Lessons learned:** the honest finding is narrower than "temporal info is
  useless." FrameStack(4) quadruples the input to 12 channels and trained ~5×
  slower (6.5 vs ~35 steps/s); evaluation reward was **still rising at 1M**, so the
  agent was *undertrained* at the fixed budget. The result is confounded by channel
  count and compute — the temporal hypothesis was not cleanly isolated.

---

## 4. Reward Ablations — *framework complete, runs pending*

- **Status:** the modular reward framework is implemented and verified. Two
  ablations are staged — baseline + `SmoothSteering` (Experiment 003) and
  + `IdlePenalty` (Experiment 004) — each changing only the reward configuration
  and evaluated on the native reward with the same 30-episode paired protocol.
- **Note:** full-budget runs are pending; no results are claimed here. When run,
  `research/generate_reward_ablation_report.py` aggregates them into
  `research/final_reward_ablation.md`.

---

## Summary of the story so far

Every controlled deviation tested to date — continued training, LR decay, frame
stacking — **regressed** against the baseline at the fixed CPU budget. The baseline
PPO (598.42) remains the champion. The scientific value is in the *method*: each
result is a clean, reproducible, statistically-tested negative, not a guess.

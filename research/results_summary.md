# Results Summary

All results use the fixed 30-episode evaluation protocol (deterministic, seeds
1000–1029, native task reward) and are compared to the frozen baseline with a
paired t-test. Numbers are drawn from each experiment's archived
`comparison.json` / `evaluation.json`.

## Headline comparison

| Experiment | Change (independent variable) | Mean | 95% CI | Δ vs baseline | Δ% | Cohen's d | Verdict |
| --- | --- | ---: | :---: | ---: | ---: | :---: | :---: |
| **Baseline PPO** | — (reference) | **598.42** | [544.8, 652.1] | — | — | — | reference |
| Continue Training | +1M→3M continued steps | 420.94 | [358.4, 483.5] | −177.48 | −29.7% | ≈ −1.1 (large)¹ | **Regressed** |
| Learning-Rate Schedule (001) | constant → linear-decay LR | 434.10 | [374.1, 494.1] | −164.32 | −27.5% | —² | **Regressed** |
| RGB FrameStack (002) | 1 frame → FrameStack(4) | 411.07 | [344.9, 477.2] | −187.35 | −31.3% | −1.16 (large) | **Regressed** |

¹ Non-overlapping 95% CIs; effect size estimated from pooled SD.
² Experiment 001 predates the Cohen's d field; significance from paired *t* = −5.499 (df=29, critical 2.045).

## Training scale-up (context for the baseline)

The baseline emerged from a standard PPO scale-up. Best evaluation reward by budget:

| Budget | Best eval reward |
| --- | ---: |
| 100k | 152.83 |
| 500k | 402.91 |
| 1M | 618.65 (peak ~900k) |

The rigorous 30-episode evaluation of the 1M best checkpoint yields the frozen
baseline of **598.42**.

## Statistical verdicts

| Experiment | Paired *t* (df=29) | Significant (p<0.05) | Verdict |
| --- | ---: | :---: | :---: |
| Continue Training | non-overlapping CIs | yes | Regressed |
| Learning-Rate Schedule | −5.499 | yes | Regressed |
| RGB FrameStack | −5.753 | yes | Regressed |

## Reward framework

The modular reward framework is **implemented and verified**; two reward ablations
(baseline + SmoothSteering; + IdlePenalty) are staged but **not yet run at full
budget**, so no reward results are reported here. See
[`reward_framework.md`](reward_framework.md) and the (pending)
`final_reward_ablation.md`.

## Bottom line

No controlled change tested so far produced a statistically significant improvement
over the frozen PPO baseline; every one **regressed**. This is reported honestly:
the platform's contribution is the rigorous, reproducible methodology — not a claim
of a better agent.

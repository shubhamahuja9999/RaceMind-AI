"""Automatic baseline comparison for controlled RaceMind AI experiments.

Compares an experiment's evaluation against the frozen baseline and decides
whether the change is statistically significant. Because both are evaluated on
identical fixed seeds, a **paired** t-test is used (more powerful than an
unpaired test), with non-overlapping 95% confidence intervals reported as a
secondary, conservative check.

Verdict is one of: ``Improved``, ``No significant change``, ``Regressed``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config.paths import PROJECT_ROOT
from evaluation.statistics import RewardStatistics, compute_reward_statistics, t_critical_95
from utils.io import read_json
from utils.logger import get_logger

_logger = get_logger(__name__)

BASELINE_PATH: Path = PROJECT_ROOT / "research" / "baseline.json"

IMPROVED = "Improved"
NO_CHANGE = "No significant change"
REGRESSED = "Regressed"


@dataclass(frozen=True)
class BaselineComparison:
    """The result of comparing an experiment against the frozen baseline."""

    baseline_name: str
    experiment_name: str
    baseline_mean: float
    experiment_mean: float
    baseline_ci: tuple[float, float]
    experiment_ci: tuple[float, float]
    absolute_improvement: float
    percent_improvement: float
    paired: bool
    n: int
    mean_difference: float
    t_statistic: float
    t_critical_95: float
    degrees_of_freedom: int
    effect_size: float
    effect_size_label: str
    ci_overlap: bool
    significant: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown section."""
        return "\n".join(
            [
                "## Comparison vs Baseline",
                "",
                "| Metric | Baseline | Experiment |",
                "| --- | --- | --- |",
                f"| Mean reward | {self.baseline_mean:.2f} | {self.experiment_mean:.2f} |",
                f"| 95% CI | [{self.baseline_ci[0]:.1f}, {self.baseline_ci[1]:.1f}] | "
                f"[{self.experiment_ci[0]:.1f}, {self.experiment_ci[1]:.1f}] |",
                "",
                f"- **Absolute improvement:** {self.absolute_improvement:+.2f}",
                f"- **Percentage improvement:** {self.percent_improvement:+.2f}%",
                f"- **Paired t-test** (n={self.n}, df={self.degrees_of_freedom}): "
                f"t = {self.t_statistic:.3f}, t-critical(95%) = {self.t_critical_95:.3f}",
                f"- **Effect size (Cohen's d):** {self.effect_size:.3f} ({self.effect_size_label})",
                f"- **95% CIs overlap:** {'yes' if self.ci_overlap else 'no'}",
                f"- **Statistically significant (p < 0.05):** "
                f"{'yes' if self.significant else 'no'}",
                "",
                f"### Verdict: **{self.verdict}**",
                "",
            ]
        )


def load_baseline() -> dict[str, Any]:
    """Load the frozen baseline record from ``research/baseline.json``."""
    return read_json(BASELINE_PATH)


def _cis_overlap(a: RewardStatistics, b: RewardStatistics) -> bool:
    """Return whether two 95% confidence intervals overlap."""
    return not (a.ci95_high < b.ci95_low or b.ci95_high < a.ci95_low)


def _cohens_d(
    base_stats: RewardStatistics,
    exp_stats: RewardStatistics,
    absolute: float,
) -> tuple[float, str]:
    """Return Cohen's d (pooled-SD standardized mean difference) and its label."""
    pooled = math.sqrt((base_stats.std ** 2 + exp_stats.std ** 2) / 2.0)
    d = absolute / pooled if pooled > 0 else 0.0
    magnitude = abs(d)
    if magnitude < 0.2:
        label = "negligible"
    elif magnitude < 0.5:
        label = "small"
    elif magnitude < 0.8:
        label = "medium"
    else:
        label = "large"
    return d, label


def _paired_t(baseline: np.ndarray, experiment: np.ndarray) -> tuple[float, int]:
    """Return the paired t-statistic and degrees of freedom for exp - baseline."""
    diffs = experiment - baseline
    n = int(diffs.size)
    std = float(diffs.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    mean_diff = float(diffs.mean())
    if sem == 0.0:
        t = 0.0 if mean_diff == 0.0 else math.copysign(float("inf"), mean_diff)
    else:
        t = mean_diff / sem
    return t, n - 1


def compare_to_baseline(
    experiment_name: str,
    experiment_rewards: Sequence[float],
    baseline: dict[str, Any] | None = None,
) -> BaselineComparison:
    """Compare an experiment's per-episode rewards against the frozen baseline.

    Args:
        experiment_name: Name of the experiment being compared.
        experiment_rewards: The experiment's per-episode rewards (same seeds and
            order as the baseline for a paired comparison).
        baseline: Optional pre-loaded baseline record; loaded from disk if
            omitted.

    Returns:
        The :class:`BaselineComparison`, including the significance verdict.
    """
    baseline = baseline or load_baseline()
    base_rewards = np.asarray(baseline["episode_rewards"], dtype=np.float64)
    exp_rewards = np.asarray(experiment_rewards, dtype=np.float64)

    base_stats = compute_reward_statistics(base_rewards)
    exp_stats = compute_reward_statistics(exp_rewards)

    absolute = exp_stats.mean - base_stats.mean
    percent = (absolute / base_stats.mean * 100.0) if base_stats.mean != 0 else 0.0

    paired = base_rewards.size == exp_rewards.size
    if paired:
        t_stat, df = _paired_t(base_rewards, exp_rewards)
        mean_diff = float((exp_rewards - base_rewards).mean())
    else:
        # Fall back to a CI-overlap decision when sample sizes differ.
        t_stat, df, mean_diff = 0.0, exp_rewards.size - 1, absolute
        _logger.warning("Baseline/experiment sizes differ; using CI-overlap only.")

    t_crit = t_critical_95(df)
    ci_overlap = _cis_overlap(base_stats, exp_stats)
    significant = paired and abs(t_stat) > t_crit
    verdict = _verdict(significant, absolute, paired, ci_overlap)
    effect_size, effect_label = _cohens_d(base_stats, exp_stats, absolute)

    comparison = BaselineComparison(
        baseline_name=baseline.get("name", "baseline"),
        experiment_name=experiment_name,
        baseline_mean=base_stats.mean,
        experiment_mean=exp_stats.mean,
        baseline_ci=(base_stats.ci95_low, base_stats.ci95_high),
        experiment_ci=(exp_stats.ci95_low, exp_stats.ci95_high),
        absolute_improvement=absolute,
        percent_improvement=percent,
        paired=paired,
        n=int(exp_rewards.size),
        mean_difference=mean_diff,
        t_statistic=t_stat,
        t_critical_95=t_crit,
        degrees_of_freedom=df,
        effect_size=effect_size,
        effect_size_label=effect_label,
        ci_overlap=ci_overlap,
        significant=significant,
        verdict=verdict,
    )
    _logger.info(
        "Baseline comparison: %+.2f (%+.2f%%), t=%.3f, verdict=%s",
        absolute, percent, t_stat, verdict,
    )
    return comparison


def _verdict(significant: bool, absolute: float, paired: bool, ci_overlap: bool) -> str:
    """Decide the verdict from the significance test (or CI overlap fallback)."""
    if paired:
        if not significant:
            return NO_CHANGE
        return IMPROVED if absolute > 0 else REGRESSED
    # Unpaired fallback: rely on CI overlap.
    if ci_overlap:
        return NO_CHANGE
    return IMPROVED if absolute > 0 else REGRESSED

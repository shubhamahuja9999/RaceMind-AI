"""Reward statistics with confidence intervals for RaceMind AI.

Computes the summary statistics used to compare agents with confidence: mean,
median, sample standard deviation, standard error, a Student's-t 95% confidence
interval for the mean, plus best/worst. Kept dependency-light (NumPy only) — the
t critical values are tabulated so SciPy is not required.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

# Two-sided 95% Student's-t critical values by degrees of freedom (df = n - 1).
# For df > 30 the normal approximation (1.96) is within ~2% and used as fallback.
_T_CRITICAL_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}
_NORMAL_95: float = 1.96


def t_critical_95(df: int) -> float:
    """Return the two-sided 95% t critical value for ``df`` degrees of freedom."""
    if df < 1:
        return float("nan")
    return _T_CRITICAL_95.get(df, _NORMAL_95)


@dataclass(frozen=True)
class RewardStatistics:
    """Summary statistics for a set of per-episode rewards."""

    n: int
    mean: float
    median: float
    std: float          # sample standard deviation (ddof=1)
    sem: float          # standard error of the mean
    ci95_low: float
    ci95_high: float
    best: float
    worst: float
    rewards: tuple[float, ...]

    @property
    def ci95_margin(self) -> float:
        """Half-width of the 95% confidence interval."""
        return (self.ci95_high - self.ci95_low) / 2.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data = asdict(self)
        data["rewards"] = list(self.rewards)
        return data


def compute_reward_statistics(rewards: Sequence[float]) -> RewardStatistics:
    """Compute :class:`RewardStatistics` from per-episode rewards.

    Args:
        rewards: The per-episode rewards (must be non-empty).

    Returns:
        The computed statistics, including a Student's-t 95% CI for the mean.

    Raises:
        ValueError: If ``rewards`` is empty.
    """
    values = np.asarray(rewards, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot compute statistics from an empty reward set.")

    mean = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    margin = t_critical_95(n - 1) * sem if n > 1 else 0.0

    return RewardStatistics(
        n=n,
        mean=mean,
        median=median,
        std=std,
        sem=sem,
        ci95_low=mean - margin,
        ci95_high=mean + margin,
        best=float(values.max()),
        worst=float(values.min()),
        rewards=tuple(float(v) for v in values),
    )

"""Core abstractions for the modular reward framework.

A reward is composed from independent components. Each component receives a
:class:`RewardContext` (everything available at a single environment step) and
returns **only its own raw contribution** — it knows nothing about other
components or their weights. The :class:`~reward.reward_manager.RewardManager`
applies weights and sums the contributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    """Everything available to a reward component at one environment step."""

    step: int
    observation: np.ndarray
    action: np.ndarray
    base_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    prev_action: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RewardResult:
    """The outcome of composing all enabled components for one step."""

    total: float
    weighted: dict[str, float]  # weighted contribution per component
    raw: dict[str, float]       # raw (pre-weight) value per component

    def contribution_percentages(self) -> dict[str, float]:
        """Return each component's share of the total absolute contribution."""
        magnitude = sum(abs(v) for v in self.weighted.values())
        if magnitude == 0.0:
            return {name: 0.0 for name in self.weighted}
        return {name: 100.0 * abs(value) / magnitude for name, value in self.weighted.items()}


class BaseRewardComponent(ABC):
    """Abstract base class for an independent reward component.

    Subclasses set :attr:`name` and implement :meth:`compute_reward`, returning
    only their own raw contribution. Components must remain independent: they may
    read the :class:`RewardContext` but must not reference other components.
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def compute_reward(self, context: RewardContext) -> float:
        """Return this component's raw (unweighted) contribution for the step."""

    def reset(self) -> None:
        """Reset any per-episode state. Default: nothing to reset."""
        return None

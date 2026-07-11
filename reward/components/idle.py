"""Idle penalty component."""

from __future__ import annotations

from reward.base_reward import BaseRewardComponent, RewardContext

_THROTTLE_EPS = 0.01


class IdlePenalty(BaseRewardComponent):
    """Penalises idling — no throttle and no progress.

    Returns ``-1.0`` when the throttle command is near zero and the step made no
    positive progress (the environment's base reward is not positive), else
    ``0.0``. This discourages the policy from stopping or coasting.
    """

    name = "idle"

    def compute_reward(self, context: RewardContext) -> float:
        """Return ``-1.0`` when idle (no throttle and no progress), else 0.0."""
        throttle = float(context.action[1])
        is_idle = throttle < _THROTTLE_EPS and context.base_reward <= 0.0
        return -1.0 if is_idle else 0.0

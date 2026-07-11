"""Smooth-steering reward component."""

from __future__ import annotations

from reward.base_reward import BaseRewardComponent, RewardContext


class SmoothSteeringReward(BaseRewardComponent):
    """Penalises abrupt steering changes to encourage smooth control.

    Returns the negative absolute change in the steering command (action index
    0) between consecutive steps. On the first step of an episode (no previous
    action) it contributes ``0.0``.
    """

    name = "smooth_steering"

    def compute_reward(self, context: RewardContext) -> float:
        """Return ``-|steering_t - steering_{t-1}|``."""
        if context.prev_action is None:
            return 0.0
        delta = float(context.action[0]) - float(context.prev_action[0])
        return -abs(delta)

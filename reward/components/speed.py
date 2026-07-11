"""Speed reward component."""

from __future__ import annotations

from reward.base_reward import BaseRewardComponent, RewardContext


class SpeedReward(BaseRewardComponent):
    """Rewards forward speed, approximated by throttle application.

    ``CarRacing-v3`` does not expose the vehicle's scalar speed in ``info``, so
    this component uses the throttle command (action index 1, range ``[0, 1]``)
    as a proxy for speed. When a true speed signal becomes available it can be
    read from ``context.info`` instead — the interface does not change.
    """

    name = "speed"

    def compute_reward(self, context: RewardContext) -> float:
        """Return the throttle magnitude as a speed proxy."""
        speed = context.info.get("speed")
        if speed is not None:
            return float(speed)
        return float(context.action[1])

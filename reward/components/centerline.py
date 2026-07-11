"""Centerline reward component."""

from __future__ import annotations

from reward.base_reward import BaseRewardComponent, RewardContext


class CenterlineReward(BaseRewardComponent):
    """Rewards staying near the track centerline.

    ``CarRacing-v3`` does not expose the signed distance from the track
    centerline, so this component reads it from ``context.info['centerline_dist']``
    when a wrapper provides it, and otherwise contributes ``0.0``. It is a
    placeholder for a real geometric signal and is disabled by default; the
    interface is ready for when that signal exists.
    """

    name = "centerline"

    def compute_reward(self, context: RewardContext) -> float:
        """Return a centerline-proximity reward, or 0.0 if unavailable."""
        distance = context.info.get("centerline_dist")
        if distance is None:
            return 0.0
        # Closer to the centerline -> higher reward (negative absolute distance).
        return -abs(float(distance))

"""Progress reward component."""

from __future__ import annotations

from reward.base_reward import BaseRewardComponent, RewardContext


class ProgressReward(BaseRewardComponent):
    """Rewards track progress using the environment's native reward.

    ``CarRacing-v3``'s built-in reward is the progress signal (positive for
    visiting new track tiles, a small negative per-frame cost). Returning it
    here — with weight 1.0 and every other component disabled — reproduces the
    baseline reward exactly.
    """

    name = "progress"

    def compute_reward(self, context: RewardContext) -> float:
        """Return the environment's native (base) reward."""
        return float(context.base_reward)

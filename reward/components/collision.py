"""Collision penalty component."""

from __future__ import annotations

from reward.base_reward import BaseRewardComponent, RewardContext


class CollisionPenalty(BaseRewardComponent):
    """Penalises collisions.

    ``CarRacing-v3`` has no obstacles or collisions, so this reads
    ``context.info['collision']`` when a wrapper provides it and otherwise
    contributes ``0.0``. It is a placeholder that keeps the framework ready for
    environments (or future track variants) that do report collisions.
    """

    name = "collision"

    def compute_reward(self, context: RewardContext) -> float:
        """Return ``-1.0`` on a reported collision, else ``0.0``."""
        return -1.0 if bool(context.info.get("collision", False)) else 0.0

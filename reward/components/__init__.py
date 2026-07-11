"""Reward components package.

Exposes the concrete components and a name-keyed registry so the
:class:`~reward.reward_manager.RewardManager` can instantiate exactly the
components named in a configuration.
"""

from __future__ import annotations

from typing import Type

from reward.base_reward import BaseRewardComponent
from reward.components.centerline import CenterlineReward
from reward.components.collision import CollisionPenalty
from reward.components.idle import IdlePenalty
from reward.components.offtrack import OffTrackPenalty
from reward.components.progress import ProgressReward
from reward.components.smooth_steering import SmoothSteeringReward
from reward.components.speed import SpeedReward

# Single source of truth mapping config names -> component classes.
COMPONENT_REGISTRY: dict[str, Type[BaseRewardComponent]] = {
    ProgressReward.name: ProgressReward,
    SpeedReward.name: SpeedReward,
    CenterlineReward.name: CenterlineReward,
    SmoothSteeringReward.name: SmoothSteeringReward,
    OffTrackPenalty.name: OffTrackPenalty,
    CollisionPenalty.name: CollisionPenalty,
    IdlePenalty.name: IdlePenalty,
}

__all__ = [
    "COMPONENT_REGISTRY",
    "ProgressReward",
    "SpeedReward",
    "CenterlineReward",
    "SmoothSteeringReward",
    "OffTrackPenalty",
    "CollisionPenalty",
    "IdlePenalty",
]

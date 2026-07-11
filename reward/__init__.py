"""Modular reward-function research framework for RaceMind AI.

Reward is composed from independent, individually-configurable components. This
package provides the component base class and context, the concrete components
and their registry, the reward manager (composition), the shaping wrapper
(integration point), and logging/plotting for analysis.

Nothing here changes the default training path — reward shaping is opt-in and
used only by reward experiments, so the frozen baseline is unaffected.
"""

from reward.base_reward import BaseRewardComponent, RewardContext, RewardResult
from reward.components import COMPONENT_REGISTRY
from reward.reward_config import RewardComponentConfig, RewardConfig
from reward.reward_manager import RewardManager, build_reward_manager

__all__ = [
    "BaseRewardComponent",
    "RewardContext",
    "RewardResult",
    "COMPONENT_REGISTRY",
    "RewardComponentConfig",
    "RewardConfig",
    "RewardManager",
    "build_reward_manager",
]

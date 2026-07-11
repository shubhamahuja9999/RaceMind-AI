"""Reward composition: combine enabled components into a single reward.

The :class:`RewardManager` instantiates the enabled components from the registry,
applies their weights, sums the contributions and returns both the total reward
and a per-component breakdown. The simulator receives only the total; the
breakdown is available for logging and analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

from reward.base_reward import BaseRewardComponent, RewardContext, RewardResult
from reward.components import COMPONENT_REGISTRY
from reward.reward_config import RewardConfig
from utils.logger import get_logger

_logger = get_logger(__name__)


class RewardManager:
    """Composes enabled reward components with their configured weights."""

    def __init__(
        self,
        config: RewardConfig,
        registry: Optional[dict[str, Type[BaseRewardComponent]]] = None,
    ) -> None:
        """Instantiate the enabled components from ``config``.

        Args:
            config: The reward configuration.
            registry: Name -> component-class map; defaults to the built-in
                :data:`~reward.components.COMPONENT_REGISTRY`.

        Raises:
            KeyError: If a configured component name is not in the registry.
        """
        registry = registry or COMPONENT_REGISTRY
        self._components: dict[str, BaseRewardComponent] = {}
        self._weights: dict[str, float] = {}
        for name, component_config in config.enabled_components().items():
            if name not in registry:
                raise KeyError(f"Unknown reward component: {name!r}")
            self._components[name] = registry[name]()
            self._weights[name] = component_config.weight
        _logger.info(
            "RewardManager: %d enabled component(s): %s",
            len(self._components), ", ".join(self._components) or "(none)",
        )

    @property
    def component_names(self) -> list[str]:
        """Names of the enabled components, in configuration order."""
        return list(self._components)

    def reset(self) -> None:
        """Reset every component's per-episode state."""
        for component in self._components.values():
            component.reset()

    def compute(self, context: RewardContext) -> RewardResult:
        """Compute the total reward and per-component breakdown for a step.

        Args:
            context: The per-step reward context.

        Returns:
            A :class:`RewardResult` with the total and per-component values.
        """
        raw: dict[str, float] = {}
        weighted: dict[str, float] = {}
        total = 0.0
        for name, component in self._components.items():
            value = float(component.compute_reward(context))
            contribution = value * self._weights[name]
            raw[name] = value
            weighted[name] = contribution
            total += contribution
        return RewardResult(total=total, weighted=weighted, raw=raw)


def build_reward_manager(config_path: Path) -> RewardManager:
    """Build a :class:`RewardManager` from a reward YAML file."""
    return RewardManager(RewardConfig.from_yaml(config_path))

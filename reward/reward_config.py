"""Reward configuration loaded from YAML.

Each component is independently configurable via ``enabled`` and ``weight``.
Changing the reward function is a matter of editing ``configs/*.yaml`` — never
Python code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.io import read_yaml


@dataclass(frozen=True)
class RewardComponentConfig:
    """Enable flag and weight for a single reward component."""

    enabled: bool = False
    weight: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    """A full reward configuration: one entry per named component."""

    components: dict[str, RewardComponentConfig]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RewardConfig":
        """Build a config from a mapping of ``name -> {enabled, weight}``."""
        components = {
            name: RewardComponentConfig(
                enabled=bool(entry.get("enabled", False)),
                weight=float(entry.get("weight", 0.0)),
            )
            for name, entry in data.items()
        }
        return cls(components=components)

    @classmethod
    def from_yaml(cls, path: Path) -> "RewardConfig":
        """Load a reward configuration from a YAML file."""
        return cls.from_mapping(read_yaml(path))

    def enabled_components(self) -> dict[str, RewardComponentConfig]:
        """Return only the enabled components (name -> config)."""
        return {name: cfg for name, cfg in self.components.items() if cfg.enabled}

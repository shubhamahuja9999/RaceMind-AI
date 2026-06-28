"""Reinforcement-learning configuration for RaceMind AI (Phase 2).

Defines the immutable config dataclasses loaded from ``configs/*.yaml`` so that
experiment parameters live in YAML rather than hardcoded Python. Each dataclass
can be built from a plain mapping (ignoring unknown keys) or directly from a
YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from config.simulator import SimulatorConfig
from utils.io import read_yaml

T = TypeVar("T")


def _from_mapping(cls: Type[T], data: dict[str, Any]) -> T:
    """Construct ``cls`` from ``data``, ignoring keys that are not fields."""
    names = {field.name for field in fields(cls)}  # type: ignore[arg-type]
    return cls(**{key: value for key, value in data.items() if key in names})


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for the Stable-Baselines3 PPO algorithm."""

    policy: str = "CnnPolicy"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PPOConfig":
        """Build a :class:`PPOConfig` from a mapping (unknown keys ignored)."""
        return _from_mapping(cls, data)


@dataclass(frozen=True)
class TrainingConfig:
    """Algorithm-agnostic training schedule."""

    seed: int = 42
    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    evaluation_frequency: int = 20_000
    checkpoint_frequency: int = 20_000

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Build a :class:`TrainingConfig` from a mapping."""
        return _from_mapping(cls, data)


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for the multi-episode evaluation pipeline."""

    n_eval_episodes: int = 5
    deterministic: bool = True
    record_video: bool = True
    success_threshold: float = 900.0
    max_steps: Optional[int] = None
    eval_seeds: list[int] = field(default_factory=lambda: [42, 123, 999, 2024, 7])

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvaluationConfig":
        """Build an :class:`EvaluationConfig` from a mapping."""
        return _from_mapping(cls, data)


def load_ppo_config(path: Path) -> PPOConfig:
    """Load PPO hyperparameters from ``path`` (``configs/ppo.yaml``)."""
    return PPOConfig.from_mapping(read_yaml(path))


def load_training_config(path: Path) -> TrainingConfig:
    """Load the training schedule from ``path`` (``configs/ppo.yaml``)."""
    return TrainingConfig.from_mapping(read_yaml(path))


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load evaluation settings from ``path`` (``configs/evaluation.yaml``)."""
    return EvaluationConfig.from_mapping(read_yaml(path))


def load_simulator_config(path: Path) -> SimulatorConfig:
    """Load simulator settings from ``path`` (``configs/simulator.yaml``)."""
    return _from_mapping(SimulatorConfig, read_yaml(path))

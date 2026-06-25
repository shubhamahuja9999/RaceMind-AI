"""Per-frame telemetry logging for the RaceMind AI simulator.

The :class:`TelemetryLogger` accumulates one :class:`TelemetryRecord` per frame
and flushes the buffer to a CSV file at the end of an episode. The data is the
foundation for later analysis and reward shaping, so the schema is kept stable
and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from config.config import SimulatorConfig, default_config
from simulator.utils import (
    generate_timestamp,
    make_episode_name,
    write_csv,
)


@dataclass
class TelemetryRecord:
    """A single frame of telemetry.

    The field order defines the CSV column order.
    """

    episode: int
    timestamp: str
    step: int
    reward: float
    steering: float
    throttle: float
    brake: float
    cumulative_reward: float
    done: bool


def _decompose_action(action: Sequence[float]) -> tuple[float, float, float]:
    """Split a ``CarRacing`` action into steering, throttle and brake.

    ``CarRacing-v3`` uses a continuous action of ``[steering, gas, brake]``.

    Args:
        action: The action passed to the environment.

    Returns:
        A ``(steering, throttle, brake)`` tuple of plain floats.
    """
    values = np.asarray(action, dtype=np.float32).ravel()
    if values.size < 3:
        raise ValueError(
            f"Expected an action with 3 components, got {values.size}"
        )
    return float(values[0]), float(values[1]), float(values[2])


class TelemetryLogger:
    """Collects per-frame telemetry for one episode and writes it to CSV."""

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        """Initialise the logger.

        Args:
            config: Simulator configuration; the default is used when omitted.
        """
        self._config: SimulatorConfig = config or default_config()
        self._records: list[TelemetryRecord] = []
        self._episode: int = 0
        self._cumulative_reward: float = 0.0

    @property
    def records(self) -> list[TelemetryRecord]:
        """The telemetry records collected so far for the current episode."""
        return self._records

    @property
    def cumulative_reward(self) -> float:
        """Running sum of rewards for the current episode."""
        return self._cumulative_reward

    def start_episode(self, episode: int) -> None:
        """Begin a new episode, clearing any previously buffered records.

        Args:
            episode: Index of the episode about to be logged.
        """
        self._episode = episode
        self._records = []
        self._cumulative_reward = 0.0

    def log(
        self,
        action: Sequence[float],
        reward: float,
        done: bool,
        step: Optional[int] = None,
    ) -> TelemetryRecord:
        """Record a single frame of telemetry.

        Args:
            action: The action applied this frame.
            reward: The reward returned this frame.
            done: Whether the episode terminated or was truncated this frame.
            step: Optional explicit step index; defaults to the running count.

        Returns:
            The :class:`TelemetryRecord` that was appended.
        """
        steering, throttle, brake = _decompose_action(action)
        self._cumulative_reward += float(reward)
        record = TelemetryRecord(
            episode=self._episode,
            timestamp=generate_timestamp(),
            step=step if step is not None else len(self._records),
            reward=float(reward),
            steering=steering,
            throttle=throttle,
            brake=brake,
            cumulative_reward=self._cumulative_reward,
            done=bool(done),
        )
        self._records.append(record)
        return record

    def save(self) -> Path:
        """Write the buffered records to a CSV file.

        Returns:
            The path of the written CSV file.
        """
        fieldnames = [field.name for field in fields(TelemetryRecord)]
        name = make_episode_name(self._config.telemetry_prefix, self._episode)
        path = self._config.telemetry_dir / f"{name}.csv"
        return write_csv(path, self._records, fieldnames)

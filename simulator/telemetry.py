"""Per-frame telemetry logging for the RaceMind AI simulator.

The :class:`TelemetryLogger` accumulates one :class:`TelemetryRecord` per frame
and flushes the buffer to a CSV file at the end of an episode.

The schema is intentionally wide: alongside the always-available fields (reward,
action components, cumulative reward) it carries optional placeholders for
metrics that the base ``CarRacing-v3`` environment does not expose directly —
speed, track position, distance travelled, lap progress, FPS, off-track and
collision flags. When a metric is unavailable it is recorded as ``None`` rather
than a fabricated value, so downstream analysis can distinguish "missing" from
"zero". New metrics can be added by extending :class:`TelemetryRecord` and the
optional keyword arguments of :meth:`TelemetryLogger.log`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from config.simulator import SimulatorConfig, default_config
from utils.io import write_csv
from utils.logger import get_logger
from utils.paths import generate_timestamp, make_episode_name

_logger = get_logger(__name__)


@dataclass
class TelemetryRecord:
    """A single frame of telemetry.

    The field order defines the CSV column order. Optional fields default to
    ``None`` and are only populated when the corresponding metric is available.
    """

    # Always available.
    episode: int
    timestamp: str
    step: int
    reward: float
    steering: float
    throttle: float
    brake: float
    cumulative_reward: float
    done: bool

    # Optional metrics — recorded as None when unavailable (never fabricated).
    speed: Optional[float] = None
    track_position: Optional[float] = None
    distance_travelled: Optional[float] = None
    lap_progress: Optional[float] = None
    fps: Optional[float] = None
    off_track: Optional[bool] = None
    collision: Optional[bool] = None
    observation_checksum: Optional[str] = None


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


def _observation_checksum(observation: Optional[np.ndarray]) -> Optional[str]:
    """Return a short, stable checksum of ``observation`` (or ``None``)."""
    if observation is None:
        return None
    array = np.ascontiguousarray(observation)
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


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
        *,
        speed: Optional[float] = None,
        track_position: Optional[float] = None,
        distance_travelled: Optional[float] = None,
        lap_progress: Optional[float] = None,
        fps: Optional[float] = None,
        off_track: Optional[bool] = None,
        collision: Optional[bool] = None,
        observation: Optional[np.ndarray] = None,
    ) -> TelemetryRecord:
        """Record a single frame of telemetry.

        Only ``action``, ``reward`` and ``done`` are required; every other
        metric is optional and recorded as ``None`` when not supplied.

        Args:
            action: The action applied this frame.
            reward: The reward returned this frame.
            done: Whether the episode terminated or was truncated this frame.
            step: Optional explicit step index; defaults to the running count.
            speed: Optional vehicle speed.
            track_position: Optional position along the track.
            distance_travelled: Optional cumulative distance travelled.
            lap_progress: Optional lap completion fraction.
            fps: Optional frames-per-second measurement.
            off_track: Optional flag for being off the track.
            collision: Optional collision flag.
            observation: Optional observation array; if given, its checksum is
                stored (the array itself is not).

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
            speed=speed,
            track_position=track_position,
            distance_travelled=distance_travelled,
            lap_progress=lap_progress,
            fps=fps,
            off_track=off_track,
            collision=collision,
            observation_checksum=_observation_checksum(observation),
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
        write_csv(path, self._records, fieldnames)
        _logger.info("Telemetry saved: %s", path)
        return path

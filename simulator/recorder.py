"""Episode recording for the RaceMind AI simulator.

The :class:`EpisodeRecorder` captures the raw materials needed to reconstruct or
replay an episode — the rendered frames, the actions taken, the rewards
received and a small metadata block — and persists them as a single compressed
``.npz`` file per episode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from config.config import SimulatorConfig, default_config
from simulator.utils import generate_timestamp, make_episode_name

# Keys used inside the ``.npz`` archive. Centralised so the recorder and the
# replay loader cannot drift apart.
FRAMES_KEY: str = "frames"
ACTIONS_KEY: str = "actions"
REWARDS_KEY: str = "rewards"
METADATA_KEY: str = "metadata"


@dataclass
class EpisodeMetadata:
    """Lightweight descriptive metadata stored alongside an episode."""

    episode: int
    env_name: str
    timestamp: str
    num_frames: int = 0
    total_reward: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    """Buffers frames, actions and rewards, then saves them as ``.npz``."""

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        """Initialise the recorder.

        Args:
            config: Simulator configuration; the default is used when omitted.
        """
        self._config: SimulatorConfig = config or default_config()
        self._episode: int = 0
        self._frames: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []

    @property
    def num_frames(self) -> int:
        """Number of frames recorded for the current episode."""
        return len(self._frames)

    @property
    def total_reward(self) -> float:
        """Sum of rewards recorded for the current episode."""
        return float(sum(self._rewards))

    def start_episode(self, episode: int) -> None:
        """Begin recording a new episode, discarding any previous buffer.

        Args:
            episode: Index of the episode about to be recorded.
        """
        self._episode = episode
        self._frames = []
        self._actions = []
        self._rewards = []

    def record(
        self,
        frame: np.ndarray,
        action: Sequence[float],
        reward: float,
    ) -> None:
        """Append one frame's worth of data to the buffer.

        Args:
            frame: The rendered RGB frame (an ``H x W x 3`` array).
            action: The action applied to produce this frame.
            reward: The reward received for this frame.
        """
        self._frames.append(np.asarray(frame, dtype=np.uint8))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._rewards.append(float(reward))

    def _build_metadata(self) -> EpisodeMetadata:
        """Assemble the metadata block for the current episode."""
        return EpisodeMetadata(
            episode=self._episode,
            env_name=self._config.env_name,
            timestamp=generate_timestamp(),
            num_frames=self.num_frames,
            total_reward=self.total_reward,
        )

    def save(self, extra_metadata: Optional[dict[str, Any]] = None) -> Path:
        """Persist the recorded episode to a compressed ``.npz`` file.

        Args:
            extra_metadata: Optional additional key/value pairs to store in the
                metadata block (e.g. the control mode used).

        Returns:
            The path of the written ``.npz`` file.

        Raises:
            RuntimeError: If no frames have been recorded.
        """
        if not self._frames:
            raise RuntimeError("Cannot save an episode with no recorded frames.")

        metadata = self._build_metadata()
        if extra_metadata:
            metadata.extra.update(extra_metadata)

        self._config.recordings_dir.mkdir(parents=True, exist_ok=True)
        name = make_episode_name(self._config.episode_prefix, self._episode)
        path = self._config.recordings_dir / f"{name}.npz"

        np.savez_compressed(
            path,
            **{
                FRAMES_KEY: np.stack(self._frames),
                ACTIONS_KEY: np.stack(self._actions),
                REWARDS_KEY: np.asarray(self._rewards, dtype=np.float32),
                METADATA_KEY: np.asarray(json.dumps(asdict(metadata))),
            },
        )
        return path

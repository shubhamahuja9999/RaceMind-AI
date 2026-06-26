"""Episode summaries for RaceMind AI.

Bundles computed :class:`~evaluation.metrics.EpisodeMetrics` with identifying
information into a serialisable :class:`EpisodeSummary`, and provides a helper to
summarise a recorded ``.npz`` episode directly. This is the seam Phase 2
evaluation reporting will build on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evaluation.metrics import EpisodeMetrics, compute_episode_metrics


@dataclass(frozen=True)
class EpisodeSummary:
    """A single episode's metrics plus identifying metadata."""

    episode: int
    metrics: EpisodeMetrics
    experiment_name: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serialisable representation of the summary."""
        data: dict[str, Any] = {
            "episode": self.episode,
            "experiment_name": self.experiment_name,
            "source": self.source,
        }
        data.update(asdict(self.metrics))
        return data


def summarize_episode(
    episode: int,
    rewards: np.ndarray,
    actions: np.ndarray,
    fps: Optional[float] = None,
    experiment_name: Optional[str] = None,
    source: Optional[str] = None,
) -> EpisodeSummary:
    """Build an :class:`EpisodeSummary` from raw rewards and actions.

    Args:
        episode: Episode index.
        rewards: 1-D array of per-step rewards.
        actions: 2-D array of per-step ``[steering, gas, brake]`` actions.
        fps: Optional FPS used to derive reward-per-second.
        experiment_name: Optional owning experiment name.
        source: Optional source identifier (e.g. a recording path).

    Returns:
        The assembled summary.
    """
    metrics = compute_episode_metrics(rewards, actions, fps=fps)
    return EpisodeSummary(
        episode=episode,
        metrics=metrics,
        experiment_name=experiment_name,
        source=source,
    )


def summarize_recording(
    path: Path,
    fps: Optional[float] = None,
    experiment_name: Optional[str] = None,
) -> EpisodeSummary:
    """Summarise a recorded ``.npz`` episode produced by ``EpisodeRecorder``.

    Args:
        path: Path to the ``.npz`` recording.
        fps: Optional FPS used to derive reward-per-second.
        experiment_name: Optional owning experiment name.

    Returns:
        The episode summary computed from the recording.
    """
    # Imported here so this module stays usable without the simulator package.
    from simulator.replay import load_episode

    recorded = load_episode(path)
    return summarize_episode(
        episode=int(recorded.metadata.get("episode", 0)),
        rewards=recorded.rewards,
        actions=recorded.actions,
        fps=fps,
        experiment_name=experiment_name,
        source=str(path),
    )

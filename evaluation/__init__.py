"""Evaluation package for RaceMind AI.

Utilities to compute episode metrics and build serialisable summaries from
recorded episodes or raw rollout data.
"""

from evaluation.episode_summary import (
    EpisodeSummary,
    summarize_episode,
    summarize_recording,
)
from evaluation.metrics import EpisodeMetrics, compute_episode_metrics

__all__ = [
    "EpisodeMetrics",
    "compute_episode_metrics",
    "EpisodeSummary",
    "summarize_episode",
    "summarize_recording",
]

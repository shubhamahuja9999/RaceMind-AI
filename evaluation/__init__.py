"""Evaluation package for RaceMind AI.

Utilities to compute episode metrics and build serialisable summaries from
recorded episodes or raw rollout data.
"""

from evaluation.benchmark import BenchmarkResult, aggregate_outcomes
from evaluation.episode_summary import (
    EpisodeSummary,
    summarize_episode,
    summarize_recording,
)
from evaluation.evaluator import Evaluator
from evaluation.metrics import EpisodeMetrics, compute_episode_metrics

__all__ = [
    "EpisodeMetrics",
    "compute_episode_metrics",
    "EpisodeSummary",
    "summarize_episode",
    "summarize_recording",
    "BenchmarkResult",
    "aggregate_outcomes",
    "Evaluator",
]

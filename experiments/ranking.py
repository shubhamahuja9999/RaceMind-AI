"""Cross-experiment ranking for RaceMind AI.

Ranks completed experiments (from the :class:`~experiments.registry.ExperimentRegistry`)
by average reward and exports the leaderboard as CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

from experiments.registry import ExperimentRecord
from utils.io import write_csv
from utils.logger import get_logger

_logger = get_logger(__name__)

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class RankRow:
    """One row of the experiment leaderboard."""

    rank: int
    experiment: str
    average_reward: Optional[float]
    best_reward: Optional[float]
    training_time: Optional[float]
    checkpoint: Optional[str]


def rank_experiments(records: Sequence[ExperimentRecord]) -> list[RankRow]:
    """Rank ``records`` by average reward (descending).

    Args:
        records: The experiment records to rank.

    Returns:
        Ranked rows, best first.
    """
    ordered = sorted(
        records,
        key=lambda r: r.average_reward if r.average_reward is not None else _NEG_INF,
        reverse=True,
    )
    return [
        RankRow(
            rank=index,
            experiment=record.experiment_name,
            average_reward=record.average_reward,
            best_reward=record.best_reward,
            training_time=record.training_seconds,
            checkpoint=record.checkpoint_path,
        )
        for index, record in enumerate(ordered, start=1)
    ]


def export_rankings_csv(records: Sequence[ExperimentRecord], path: Path) -> Path:
    """Rank ``records`` and write the leaderboard to ``path`` as CSV.

    Args:
        records: The experiment records to rank.
        path: Destination CSV path.

    Returns:
        The written path.
    """
    rows = rank_experiments(records)
    fieldnames = [f.name for f in fields(RankRow)]
    write_csv(path, rows, fieldnames)
    _logger.info("Rankings exported: %s (%d experiments)", path, len(rows))
    return path

"""Learning-curve plotting for RaceMind AI.

Generates training/evaluation plots with matplotlib and saves them under
``data/plots/<experiment>/``. Data is read from the artifacts produced during
training:

* ``monitor.csv``  — per-episode reward and length (SB3 ``Monitor`` wrapper).
* ``progress.csv`` — per-update metrics incl. FPS and loss (SB3 CSV logger).
* evaluation points — ``(timestep, reward)`` pairs from the trainer.

Every plot is optional: if its source data is missing, it is skipped (so the
"Loss (if available)" requirement is honoured gracefully).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required.

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils.logger import get_logger  # noqa: E402
from utils.paths import ensure_directory  # noqa: E402

_logger = get_logger(__name__)


def _save_line(
    output_dir: Path,
    filename: str,
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    """Plot a single line series and save it as a PNG."""
    path = output_dir / filename
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(x, y, linewidth=1.5)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _read_csv(path: Optional[Path], comment: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Read a CSV into a DataFrame, returning ``None`` if missing or empty."""
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_csv(path, comment=comment)
    except (pd.errors.EmptyDataError, OSError):
        return None
    return frame if not frame.empty else None


def generate_learning_curves(
    output_dir: Path,
    monitor_csv: Optional[Path] = None,
    progress_csv: Optional[Path] = None,
    evaluations: Optional[Sequence[tuple[int, float]]] = None,
    moving_average_window: int = 10,
) -> list[Path]:
    """Generate all available learning-curve plots.

    Args:
        output_dir: Directory to save plots into (created if missing).
        monitor_csv: Optional path to an SB3 ``Monitor`` CSV.
        progress_csv: Optional path to an SB3 CSV-logger ``progress.csv``.
        evaluations: Optional ``(timestep, reward)`` evaluation points.
        moving_average_window: Window for the moving-average reward curve.

    Returns:
        The paths of the plots that were generated.
    """
    ensure_directory(output_dir)
    saved: list[Path] = []

    saved += _episode_plots(output_dir, _read_csv(monitor_csv, comment="#"), moving_average_window)
    saved += _progress_plots(output_dir, _read_csv(progress_csv))
    saved += _evaluation_plot(output_dir, evaluations)

    if not saved:
        _logger.warning("No learning-curve data available to plot in %s", output_dir)
    else:
        _logger.info("Generated %d learning-curve plot(s) in %s", len(saved), output_dir)
    return saved


def _episode_plots(
    output_dir: Path,
    monitor: Optional[pd.DataFrame],
    window: int,
) -> list[Path]:
    """Episode reward, moving-average reward and episode-length plots."""
    if monitor is None or "r" not in monitor.columns:
        return []

    rewards = monitor["r"].to_numpy()
    episodes = range(1, len(rewards) + 1)
    saved = [
        _save_line(
            output_dir, "episode_reward.png", episodes, rewards,
            "Episode Reward", "Episode", "Reward",
        )
    ]

    effective_window = max(1, min(window, len(rewards)))
    moving = monitor["r"].rolling(effective_window, min_periods=1).mean().to_numpy()
    saved.append(
        _save_line(
            output_dir, "moving_average_reward.png", episodes, moving,
            f"Moving-Average Reward (window={effective_window})", "Episode", "Reward",
        )
    )

    if "l" in monitor.columns:
        saved.append(
            _save_line(
                output_dir, "episode_length.png", episodes, monitor["l"].to_numpy(),
                "Episode Length", "Episode", "Steps",
            )
        )
    return saved


def _progress_plots(output_dir: Path, progress: Optional[pd.DataFrame]) -> list[Path]:
    """Training FPS and loss plots from the SB3 progress log."""
    if progress is None:
        return []

    x_column = "time/total_timesteps"
    x = progress[x_column] if x_column in progress.columns else range(len(progress))
    saved: list[Path] = []

    for column, filename, title, ylabel in (
        ("time/fps", "training_fps.png", "Training FPS", "FPS"),
        ("train/loss", "loss.png", "Training Loss", "Loss"),
    ):
        if column in progress.columns:
            series = progress[[column]].copy()
            series["x"] = list(x)
            series = series.dropna()
            if not series.empty:
                saved.append(
                    _save_line(
                        output_dir, filename, series["x"].to_numpy(),
                        series[column].to_numpy(), title, "Timesteps", ylabel,
                    )
                )
    return saved


def _evaluation_plot(
    output_dir: Path,
    evaluations: Optional[Sequence[tuple[int, float]]],
) -> list[Path]:
    """Evaluation-reward-vs-timestep plot."""
    if not evaluations:
        return []
    steps = [step for step, _ in evaluations]
    rewards = [reward for _, reward in evaluations]
    return [
        _save_line(
            output_dir, "evaluation_reward.png", steps, rewards,
            "Evaluation Reward", "Timesteps", "Reward",
        )
    ]

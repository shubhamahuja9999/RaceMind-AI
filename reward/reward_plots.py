"""Reward visualisation.

Given a per-component reward log CSV (from :class:`~reward.reward_logger.RewardLogger`),
generate plots of the total reward over time, each component over time, and the
overall contribution percentages. Saved under ``research/reward_analysis/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils.logger import get_logger  # noqa: E402
from utils.paths import ensure_directory  # noqa: E402

_logger = get_logger(__name__)


def generate_reward_plots(
    csv_path: Path,
    output_dir: Path,
    component_names: Sequence[str],
) -> list[Path]:
    """Generate reward-analysis plots from a reward log CSV.

    Args:
        csv_path: Path to the reward log CSV.
        output_dir: Directory to save plots into (created if missing).
        component_names: The component columns to plot.

    Returns:
        The paths of the generated plots.
    """
    ensure_directory(output_dir)
    frame = pd.read_csv(csv_path)
    present = [name for name in component_names if name in frame.columns]
    saved: list[Path] = []

    saved.append(_plot_total(output_dir, frame))
    if present:
        saved.append(_plot_components(output_dir, frame, present))
        saved.append(_plot_contribution_pct(output_dir, frame, present))
    _logger.info("Generated %d reward plot(s) in %s", len(saved), output_dir)
    return saved


def _plot_total(output_dir: Path, frame: pd.DataFrame) -> Path:
    """Total (and base) reward over time."""
    path = output_dir / "reward_total.png"
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.plot(frame["step"], frame["total_reward"], label="total (shaped)", linewidth=1.5)
    if "base_reward" in frame.columns:
        axes.plot(frame["step"], frame["base_reward"], label="base (env)", linewidth=1.0, alpha=0.7)
    axes.set_title("Reward over time")
    axes.set_xlabel("Step")
    axes.set_ylabel("Reward")
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _plot_components(output_dir: Path, frame: pd.DataFrame, names: Sequence[str]) -> Path:
    """Each component's weighted contribution over time."""
    path = output_dir / "reward_components.png"
    figure, axes = plt.subplots(figsize=(9, 5))
    for name in names:
        axes.plot(frame["step"], frame[name], label=name, linewidth=1.2)
    axes.set_title("Reward components over time (weighted)")
    axes.set_xlabel("Step")
    axes.set_ylabel("Contribution")
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _plot_contribution_pct(output_dir: Path, frame: pd.DataFrame, names: Sequence[str]) -> Path:
    """Overall contribution share (mean absolute contribution per component)."""
    path = output_dir / "reward_contribution_pct.png"
    magnitudes = {name: float(frame[name].abs().mean()) for name in names}
    total = sum(magnitudes.values())
    percentages = {n: (100.0 * v / total if total > 0 else 0.0) for n, v in magnitudes.items()}

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.bar(list(percentages.keys()), list(percentages.values()))
    axes.set_title("Mean contribution share per component")
    axes.set_ylabel("Share of |contribution| (%)")
    axes.tick_params(axis="x", rotation=30)
    axes.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path

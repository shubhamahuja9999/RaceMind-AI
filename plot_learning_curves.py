"""Phase 7: Generate learning curve plots from TensorBoard logs.

Produces three plots for the README and reports:
  1. Reward vs. timesteps
  2. Episode length vs. timesteps
  3. Evaluation reward vs. checkpoints

Run: python -m plot_learning_curves
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

# Use a modern font if available
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
})

PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def read_tb_events(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    """Read TensorBoard event files and extract scalar data.

    Returns a dict mapping tag names to lists of (step, value) tuples.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    data: dict[str, list[tuple[int, float]]] = {}

    # Find all event files and read them
    for events_dir in sorted(logdir.rglob("PPO_*")):
        ea = EventAccumulator(str(events_dir))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag not in data:
                data[tag] = []
            for event in ea.Scalars(tag):
                data[tag].append((event.step, event.value))

    # Sort each tag by step
    for tag in data:
        data[tag].sort(key=lambda x: x[0])

    return data


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward_vs_timesteps(data: dict, save_path: Path) -> None:
    """Plot 1: Reward vs. timesteps."""
    tag = "rollout/ep_rew_mean"
    if tag not in data:
        print(f"  [SKIP] Tag '{tag}' not found in logs.")
        return

    steps, values = zip(*data[tag])
    steps = np.array(steps)
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw data (transparent)
    ax.plot(steps, values, alpha=0.25, color="#58a6ff", linewidth=1)
    # Smoothed
    smoothed = smooth(values, window=15)
    smooth_steps = steps[:len(smoothed)]
    ax.plot(smooth_steps, smoothed, color="#58a6ff", linewidth=2.5, label="Mean Episode Reward (smoothed)")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward vs. Timesteps - PPO on CarRacing-v3")
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, linestyle="--")
    ax.axhline(y=0, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.5, label="Zero baseline")
    ax.axhline(y=900, color="#3fb950", linewidth=0.8, linestyle="--", alpha=0.5, label="Success (900)")

    # Fill between zero and reward
    ax.fill_between(smooth_steps, 0, smoothed, alpha=0.08, color="#58a6ff")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {save_path}")


def plot_episode_length_vs_timesteps(data: dict, save_path: Path) -> None:
    """Plot 2: Episode length vs. timesteps."""
    tag = "rollout/ep_len_mean"
    if tag not in data:
        print(f"  [SKIP] Tag '{tag}' not found in logs.")
        return

    steps, values = zip(*data[tag])
    steps = np.array(steps)
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, values, alpha=0.25, color="#d2a8ff", linewidth=1)
    smoothed = smooth(values, window=15)
    smooth_steps = steps[:len(smoothed)]
    ax.plot(smooth_steps, smoothed, color="#d2a8ff", linewidth=2.5, label="Mean Episode Length (smoothed)")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Length vs. Timesteps - PPO on CarRacing-v3")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, linestyle="--")
    ax.set_ylim(0, 1100)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {save_path}")


def plot_eval_reward_vs_checkpoints(save_path: Path) -> None:
    """Plot 3: Evaluation reward vs. checkpoints (from our known eval data)."""
    # Data from our training runs
    eval_data = {
        "10K": [(-26.47, 5000), (-37.91, 10000)],
        "100K": [(-24.44, 20000), (152.83, 100000)],
        "500K": [
            (-20.51, 50000), (152.83, 100000), (132.58, 150000),
            (42.28, 200000), (151.65, 250000), (393.74, 300000),
            (168.08, 350000), (402.91, 400000), (148.59, 450000), (201.16, 500000),
        ],
        "1M": [
            (229.84, 100000), (87.82, 200000), (241.42, 300000),
            (319.47, 400000), (89.33, 500000), (339.62, 600000),
            (172.46, 700000), (548.76, 800000), (618.65, 900000), (358.83, 1000000),
        ],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the 1M run as the primary curve
    rewards_1m, steps_1m = zip(*eval_data["1M"])
    steps_1m = np.array(steps_1m)
    rewards_1m = np.array(rewards_1m)

    ax.plot(steps_1m, rewards_1m, "o-", color="#f0883e", linewidth=2.5,
            markersize=8, markerfacecolor="#f0883e", markeredgecolor="white",
            markeredgewidth=1.5, label="1M Run — Eval Reward", zorder=3)

    # Add 500K run
    rewards_500k, steps_500k = zip(*eval_data["500K"])
    ax.plot(steps_500k, rewards_500k, "s--", color="#8b949e", linewidth=1.5,
            markersize=6, alpha=0.6, label="500K Run — Eval Reward")

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Evaluation Reward (5-episode avg)")
    ax.set_title("Evaluation Reward vs. Checkpoints - PPO on CarRacing-v3")
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, linestyle="--")
    ax.axhline(y=0, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(y=900, color="#3fb950", linewidth=0.8, linestyle="--", alpha=0.5)

    # Annotate best
    best_idx = np.argmax(rewards_1m)
    ax.annotate(
        f"Best: {rewards_1m[best_idx]:.0f}",
        xy=(steps_1m[best_idx], rewards_1m[best_idx]),
        xytext=(steps_1m[best_idx] - 150000, rewards_1m[best_idx] + 80),
        arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.5),
        color="#3fb950", fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {save_path}")


def main() -> None:
    print("=" * 60)
    print("Phase 7 — Generating Learning Curves")
    print("=" * 60)

    # Read TensorBoard data from the 1M run
    logdir = Path("runs/ppo_1m")
    print(f"\nReading TensorBoard logs from: {logdir}")
    data = read_tb_events(logdir)
    print(f"  Found {len(data)} scalar tags: {list(data.keys())[:8]}...")

    # Plot 1: Reward vs. timesteps
    print("\nPlot 1: Reward vs. timesteps")
    plot_reward_vs_timesteps(data, PLOTS_DIR / "reward_vs_timesteps.png")

    # Plot 2: Episode length vs. timesteps
    print("Plot 2: Episode length vs. timesteps")
    plot_episode_length_vs_timesteps(data, PLOTS_DIR / "episode_length_vs_timesteps.png")

    # Plot 3: Evaluation reward vs. checkpoints
    print("Plot 3: Evaluation reward vs. checkpoints")
    plot_eval_reward_vs_checkpoints(PLOTS_DIR / "eval_reward_vs_checkpoints.png")

    print("\n" + "=" * 60)
    print(f"All plots saved to: {PLOTS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

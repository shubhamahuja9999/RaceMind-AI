"""Aggregate reward-ablation report generator.

Reads each completed reward experiment's ``comparison.json`` / ``evaluation.json``
and the frozen baseline, then writes ``research/final_reward_ablation.md`` — a
Results-and-Discussion style summary of the whole ablation study: a summary
table, best/worst performers, effect sizes, honest lessons, limitations and
recommendations.

Usage:
    python -m research.generate_reward_ablation_report
    python -m research.generate_reward_ablation_report --experiments reward_smooth_steering reward_smooth_idle
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config.paths import PROJECT_ROOT
from reward.reward_config import RewardConfig
from research.comparison import IMPROVED, NO_CHANGE, REGRESSED, load_baseline
from utils.io import read_json
from utils.logger import get_logger
from utils.paths import ensure_directory
from utils.tables import markdown_table

_logger = get_logger(__name__)

EXPERIMENTS_DIR = PROJECT_ROOT / "research" / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORT_PATH = PROJECT_ROOT / "research" / "final_reward_ablation.md"
DEFAULT_EXPERIMENTS = ["reward_smooth_steering", "reward_smooth_idle"]


@dataclass
class ExperimentRow:
    """One experiment's aggregated results (or a not-run marker)."""

    name: str
    components: str
    comparison: Optional[dict[str, Any]]
    evaluation: Optional[dict[str, Any]]

    @property
    def ran(self) -> bool:
        return self.comparison is not None and self.evaluation is not None


def _components_label(name: str) -> str:
    """List enabled reward components for an experiment from its config."""
    config_path = CONFIGS_DIR / f"{name}.yaml"
    if not config_path.exists():
        return name
    enabled = RewardConfig.from_yaml(config_path).enabled_components()
    return " + ".join(enabled) if enabled else "(none)"


def _load_row(name: str) -> ExperimentRow:
    """Load an experiment's comparison + evaluation, if present."""
    archive = EXPERIMENTS_DIR / name
    comparison = read_json(archive / "comparison.json") if (archive / "comparison.json").exists() else None
    evaluation = read_json(archive / "evaluation.json") if (archive / "evaluation.json").exists() else None
    return ExperimentRow(name, _components_label(name), comparison, evaluation)


def _summary_table(baseline: dict[str, Any], rows: list[ExperimentRow]) -> str:
    """Render the summary comparison table (baseline first)."""
    headers = ["Experiment", "Components", "Mean", "Median", "Std", "95% CI", "Δ", "Δ%", "Cohen's d", "Verdict"]
    stats = baseline["statistics"]
    table_rows = [[
        "**baseline_ppo**", "progress (native)", f"{stats['mean']:.2f}", f"{stats['median']:.2f}",
        f"{stats['std']:.2f}", f"[{stats['ci95_low']:.1f}, {stats['ci95_high']:.1f}]", "—", "—", "—", "reference",
    ]]
    for row in rows:
        if not row.ran:
            table_rows.append([row.name, row.components, "—", "—", "—", "—", "—", "—", "—", "_not run_"])
            continue
        cmp, ev = row.comparison, row.evaluation
        table_rows.append([
            row.name, row.components, f"{ev['mean']:.2f}", f"{ev['median']:.2f}", f"{ev['std']:.2f}",
            f"[{ev['ci95_low']:.1f}, {ev['ci95_high']:.1f}]", f"{cmp['absolute_improvement']:+.1f}",
            f"{cmp['percent_improvement']:+.1f}%", f"{cmp['effect_size']:.2f}", f"**{cmp['verdict']}**",
        ])
    return markdown_table(headers, table_rows)


def _discussion(rows: list[ExperimentRow]) -> list[str]:
    """Generate honest, data-driven discussion prose from the verdicts."""
    ran = [r for r in rows if r.ran]
    if not ran:
        return ["No reward experiments have completed yet. Run them, then regenerate this report."]

    improved = [r for r in ran if r.comparison["verdict"] == IMPROVED]
    regressed = [r for r in ran if r.comparison["verdict"] == REGRESSED]
    unchanged = [r for r in ran if r.comparison["verdict"] == NO_CHANGE]

    best = max(ran, key=lambda r: r.evaluation["mean"])
    worst = min(ran, key=lambda r: r.evaluation["mean"])

    lines = [
        f"- **Best performing:** `{best.name}` ({best.components}) — mean "
        f"{best.evaluation['mean']:.2f}, verdict {best.comparison['verdict']}.",
        f"- **Worst performing:** `{worst.name}` ({worst.components}) — mean "
        f"{worst.evaluation['mean']:.2f}, verdict {worst.comparison['verdict']}.",
        "",
    ]
    if improved:
        names = ", ".join(f"`{r.name}`" for r in improved)
        lines.append(
            f"{len(improved)} configuration(s) significantly improved on the baseline: {names}. "
            "This is evidence that the corresponding reward shaping helps, but see the limitations "
            "before generalising (single seed, fixed weights)."
        )
    else:
        lines.append(
            "**No reward configuration produced a statistically significant improvement over the "
            "frozen PPO baseline.** The reward-shaping hypothesis is therefore rejected for the "
            "components and weights tested: at this budget, adding these shaping terms did not help, "
            f"and {len(regressed)} regressed while {len(unchanged)} showed no significant change."
        )
    return lines


def _effect_sizes(rows: list[ExperimentRow]) -> str:
    """Render an effect-size list for completed experiments."""
    ran = [r for r in rows if r.ran]
    if not ran:
        return "_No completed experiments._"
    items = [
        f"- `{r.name}`: Cohen's d = {r.comparison['effect_size']:.3f} "
        f"({r.comparison['effect_size_label']}), paired t = {r.comparison['t_statistic']:.3f}"
        for r in ran
    ]
    return "\n".join(items)


def build_report(baseline: dict[str, Any], rows: list[ExperimentRow]) -> str:
    """Assemble the full Markdown ablation report."""
    return "\n".join([
        "# Reward Ablation Study — Results and Discussion",
        "",
        "## Results",
        "",
        "All configurations were trained under identical conditions (PPO, CnnPolicy, single-frame "
        "observation, fixed hyperparameters) — the reward composition was the only variable — and "
        "evaluated on the **native task reward** with 30 deterministic episodes (seeds 1000–1029), "
        "compared to the frozen baseline with a paired t-test.",
        "",
        _summary_table(baseline, rows),
        "",
        "### Effect sizes",
        "",
        _effect_sizes(rows),
        "",
        "## Discussion",
        "",
        *_discussion(rows),
        "",
        "## Lessons Learned",
        "",
        "- Evaluating on the native reward (not the shaped reward) is essential: a shaped reward can "
        "score well on its own objective while driving worse on the real task.",
        "- The framework isolates the reward as a single variable cleanly; verdicts are backed by a "
        "paired test on identical seeds, so differences are attributable to the reward alone.",
        "",
        "## Limitations",
        "",
        "- **Single seed / single run per configuration** — no across-seed variance is estimated, so "
        "results reflect one training trajectory each.",
        "- **Fixed, un-tuned weights** — each component used one hand-chosen weight; a different weight "
        "could change the outcome. Weight sensitivity was deliberately not explored (no tuning).",
        "- **Compute-limited budget** — CPU training at a modest budget; effects that need longer "
        "training may be undetectable here.",
        "",
        "## Recommendations for Future Work",
        "",
        "- Re-run the most promising configuration across several seeds to estimate variance.",
        "- Study weight sensitivity for any component that showed a non-negligible effect.",
        "- Add a true speed / centerline signal (currently a proxy / placeholder) before drawing "
        "conclusions about those components.",
        "",
    ])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the aggregate reward-ablation report.")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS,
                        help="Experiment (archive) names to include.")
    return parser.parse_args()


def main() -> None:
    """Generate ``research/final_reward_ablation.md`` from completed experiments."""
    args = _parse_args()
    baseline = load_baseline()
    rows = [_load_row(name) for name in args.experiments]
    ensure_directory(REPORT_PATH.parent)
    REPORT_PATH.write_text(build_report(baseline, rows), encoding="utf-8")

    ran = sum(1 for r in rows if r.ran)
    _logger.info("Ablation report written: %s (%d/%d experiments included)", REPORT_PATH, ran, len(rows))


if __name__ == "__main__":
    main()

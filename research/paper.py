"""Auto-generated research write-ups for controlled RaceMind AI experiments.

Renders a short research-paper-style Markdown report from the structured outputs
of an experiment (spec, training summary, evaluation statistics and the baseline
comparison). Interpretation and lessons are change-agnostic by default (driven by
the significance verdict) but can be overridden with experiment-specific prose.
"""

from __future__ import annotations

from typing import Optional, Sequence

from evaluation.statistics import RewardStatistics
from research.comparison import IMPROVED, NO_CHANGE, REGRESSED, BaselineComparison
from research.experiment import ExperimentSpec


def _default_interpretation(comparison: BaselineComparison) -> str:
    """Verdict-based, change-agnostic interpretation prose."""
    common = (
        f"mean change {comparison.absolute_improvement:+.1f} "
        f"({comparison.percent_improvement:+.1f}%), "
        f"Cohen's d = {comparison.effect_size:.2f} ({comparison.effect_size_label})"
    )
    if comparison.verdict == IMPROVED:
        return (
            f"The change produced a statistically significant improvement ({common}). "
            "The hypothesis is supported."
        )
    if comparison.verdict == REGRESSED:
        return (
            f"The change produced a statistically significant regression ({common}). "
            "The hypothesis is rejected."
        )
    return (
        f"The change produced no statistically significant difference ({common}); "
        f"the 95% confidence intervals {'overlap' if comparison.ci_overlap else 'are inconclusive'}. "
        "The null hypothesis cannot be rejected on this budget."
    )


def _default_lessons(comparison: BaselineComparison) -> str:
    """Verdict-based, change-agnostic lessons prose."""
    if comparison.verdict == IMPROVED:
        return (
            "The tested change is beneficial and should be carried forward into "
            "the default configuration and subsequent experiments."
        )
    if comparison.verdict == REGRESSED:
        return (
            "The tested change is harmful under these conditions and should not be "
            "adopted; the baseline configuration remains preferable."
        )
    return (
        "With a large per-episode variance, a single 1M-step run may be "
        "underpowered to detect a modest effect. A larger training budget or more "
        "evaluation episodes would be needed to resolve small differences."
    )


def build_experiment_paper(
    spec: ExperimentSpec,
    comparison: BaselineComparison,
    experiment_stats: RewardStatistics,
    training_timesteps: int,
    elapsed_seconds: float,
    evaluations: Sequence[tuple[int, float]],
    plots: Sequence[str],
    resumed_from: Optional[str] = None,
    abstract: Optional[str] = None,
    interpretation: Optional[str] = None,
    lessons: Optional[str] = None,
    future_work: Optional[str] = None,
) -> str:
    """Render the experiment write-up as Markdown.

    Args:
        spec: The experiment specification.
        comparison: The baseline comparison (with significance + effect size).
        experiment_stats: Statistics of the experiment's best-checkpoint eval.
        training_timesteps: Timesteps trained in this experiment.
        elapsed_seconds: Wall-clock training time.
        evaluations: ``(timestep, mean_reward)`` points recorded during training.
        plots: Names of the generated plot files.
        resumed_from: Checkpoint resumed from, or ``None`` if trained from scratch.
        abstract: Optional abstract paragraph.
        interpretation: Optional interpretation override (else verdict-based).
        lessons: Optional lessons override (else verdict-based).
        future_work: Optional future-work override.
    """
    controlled_rows = "\n".join(
        f"| `{key}` | {value} |" for key, value in spec.controlled_variables.items()
    )
    eval_curve = "\n".join(f"| {step:,} | {reward:.2f} |" for step, reward in evaluations)
    plot_list = "\n".join(f"- `{name}`" for name in plots) or "_none_"
    methodology = (
        f"Training was resumed from `{resumed_from}` and continued for "
        f"{training_timesteps:,} timesteps."
        if resumed_from
        else f"The policy was trained from scratch for {training_timesteps:,} timesteps."
    )

    lines: list[str] = [f"# Experiment {spec.experiment_id}: {spec.name}", ""]

    if abstract:
        lines += ["## Abstract", "", abstract, ""]

    lines += [
        "## Hypothesis",
        "",
        spec.hypothesis,
        "",
        "## Methodology",
        "",
        f"{methodology} Exactly one variable was changed relative to the frozen "
        "baseline (below); every other hyperparameter, the environment, and the "
        "evaluation protocol were held identical. Evaluation used the frozen "
        "protocol: 30 deterministic episodes on fixed seeds (1000-1029), compared "
        "to the baseline with a paired t-test (identical seeds enable pairing).",
        "",
        "### Independent variable (changed)",
        "",
        f"- {spec.independent_variable}",
        "",
        "### Controlled variables (held identical to baseline)",
        "",
        "| Variable | Value |",
        "| --- | --- |",
        controlled_rows,
        "",
        "## Training",
        "",
        f"- Timesteps: {training_timesteps:,}",
        f"- Wall-clock: {elapsed_seconds / 3600:.2f} h ({elapsed_seconds / 60:.1f} min)",
        "",
        "### Evaluation reward during training",
        "",
        "| Timestep | Mean eval reward |",
        "| --- | --- |",
        eval_curve,
        "",
        "## Evaluation (best checkpoint, 30 deterministic episodes)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean | {experiment_stats.mean:.2f} |",
        f"| Median | {experiment_stats.median:.2f} |",
        f"| Std | {experiment_stats.std:.2f} |",
        f"| 95% CI | [{experiment_stats.ci95_low:.2f}, {experiment_stats.ci95_high:.2f}] |",
        f"| Best | {experiment_stats.best:.2f} |",
        f"| Worst | {experiment_stats.worst:.2f} |",
        "",
        "## Results & Statistical Analysis",
        "",
        comparison.to_markdown(),
        "## Interpretation",
        "",
        interpretation or _default_interpretation(comparison),
        "",
        "## Lessons Learned",
        "",
        lessons or _default_lessons(comparison),
        "",
        "## Generated Plots",
        "",
        plot_list,
        "",
        "## Future Work",
        "",
        future_work
        or (
            "Run the next experiment changing a different single variable against "
            "this same frozen baseline and identical 30-episode paired protocol."
        ),
        "",
    ]
    return "\n".join(lines)

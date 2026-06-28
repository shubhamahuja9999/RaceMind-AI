"""Built-in grid-search hyperparameter sweep for RaceMind AI.

A small, dependency-free sweep framework (no Optuna, no Ray). It enumerates the
Cartesian product of a parameter grid over PPO hyperparameters, runs a full
experiment per combination via :func:`experiments.runner.execute_experiment`,
then ranks the results and writes a sweep summary + ranking CSV.

Example:
    python -m experiments.sweep --sweep-name lr_gamma \\
        --param learning_rate=0.0003,0.0001 --param gamma=0.99,0.95 \\
        --total-timesteps 5000
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional, Sequence

from config.paths import PROJECT_ROOT
from config.rl import (
    EvaluationConfig,
    PPOConfig,
    TrainingConfig,
    load_evaluation_config,
    load_ppo_config,
    load_simulator_config,
    load_training_config,
)
from config.simulator import SimulatorConfig
from experiments.runner import execute_experiment
from utils.io import write_csv, write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
_PPO_FIELDS = {f.name for f in fields(PPOConfig)}


@dataclass(frozen=True)
class SweepRunResult:
    """The outcome of a single sweep combination."""

    rank: int
    experiment_name: str
    params: dict[str, Any]
    average_reward: Optional[float]
    best_reward: Optional[float]
    checkpoint_path: Optional[str]


def _grid_combinations(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of concrete parameter mappings."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _validate_grid(grid: dict[str, Sequence[Any]]) -> None:
    """Ensure every grid key is a real PPO hyperparameter."""
    unknown = set(grid) - _PPO_FIELDS
    if unknown:
        raise ValueError(f"Unknown PPO hyperparameters in sweep grid: {sorted(unknown)}")


def run_sweep(
    grid: dict[str, Sequence[Any]],
    simulator_config: SimulatorConfig,
    base_ppo_config: PPOConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    sweep_name: str = "sweep",
    multi_seed: bool = False,
) -> list[SweepRunResult]:
    """Run a grid sweep and return ranked results.

    Args:
        grid: Mapping of PPO hyperparameter name to the list of values to try.
        simulator_config: Shared simulator configuration.
        base_ppo_config: Base PPO config; each combination overrides fields on it.
        training_config: Shared training schedule.
        evaluation_config: Shared evaluation settings.
        sweep_name: Name prefix for the sweep and its experiments.
        multi_seed: Whether each run also performs multi-seed evaluation.

    Returns:
        Ranked :class:`SweepRunResult` entries, best average reward first.
    """
    _validate_grid(grid)
    combinations = _grid_combinations(grid)
    _logger.info("Starting sweep '%s' with %d combination(s)", sweep_name, len(combinations))

    scored: list[tuple[dict[str, Any], Any]] = []
    for index, params in enumerate(combinations):
        experiment_name = f"{sweep_name}__{index:03d}"
        _logger.info("Sweep run %d/%d: %s %s", index + 1, len(combinations), experiment_name, params)
        ppo_config = replace(base_ppo_config, **params)
        result = execute_experiment(
            experiment_name=experiment_name,
            simulator_config=simulator_config,
            ppo_config=ppo_config,
            training_config=training_config,
            evaluation_config=evaluation_config,
            multi_seed=multi_seed,
        )
        scored.append((params, result))

    ranked = _rank(scored)
    _persist(sweep_name, simulator_config, grid, ranked)
    return ranked


def _rank(scored: Sequence[tuple[dict[str, Any], Any]]) -> list[SweepRunResult]:
    """Rank scored runs by average reward (descending)."""
    ordered = sorted(
        scored,
        key=lambda item: item[1].benchmark.get("average_reward", float("-inf")),
        reverse=True,
    )
    return [
        SweepRunResult(
            rank=rank,
            experiment_name=result.experiment_name,
            params=params,
            average_reward=result.benchmark.get("average_reward"),
            best_reward=result.benchmark.get("max_reward"),
            checkpoint_path=result.checkpoint_path,
        )
        for rank, (params, result) in enumerate(ordered, start=1)
    ]


def _persist(
    sweep_name: str,
    simulator_config: SimulatorConfig,
    grid: dict[str, Sequence[Any]],
    ranked: Sequence[SweepRunResult],
) -> tuple[Path, Path]:
    """Write the sweep summary (JSON) and ranking (CSV)."""
    output_dir = simulator_config.evaluation_dir / sweep_name
    ensure_directory(output_dir)

    json_path = output_dir / "sweep_results.json"
    write_json(
        json_path,
        {
            "sweep_name": sweep_name,
            "grid": {key: list(values) for key, values in grid.items()},
            "runs": [asdict(run) for run in ranked],
        },
    )

    csv_path = output_dir / "sweep_ranking.csv"
    rows = [
        {
            "rank": run.rank,
            "experiment": run.experiment_name,
            "params": ";".join(f"{k}={v}" for k, v in run.params.items()),
            "average_reward": run.average_reward,
            "best_reward": run.best_reward,
            "checkpoint": run.checkpoint_path,
        }
        for run in ranked
    ]
    write_csv(csv_path, rows, ["rank", "experiment", "params", "average_reward", "best_reward", "checkpoint"])
    _logger.info("Sweep results written: %s", json_path)
    return json_path, csv_path


def _parse_value(raw: str) -> Any:
    """Parse a CLI grid value as int, then float, else leave as string."""
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def _parse_grid(param_specs: Sequence[str]) -> dict[str, list[Any]]:
    """Parse repeated ``name=v1,v2`` CLI specs into a grid mapping."""
    grid: dict[str, list[Any]] = {}
    for spec in param_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --param '{spec}'; expected name=v1,v2,...")
        name, raw_values = spec.split("=", 1)
        grid[name.strip()] = [_parse_value(v.strip()) for v in raw_values.split(",") if v.strip()]
    return grid


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sweep."""
    parser = argparse.ArgumentParser(description="Grid-search hyperparameter sweep.")
    parser.add_argument("--sweep-name", default="sweep")
    parser.add_argument(
        "--param", action="append", default=[],
        help="Grid spec 'name=v1,v2'. Repeatable. e.g. --param learning_rate=3e-4,1e-4",
    )
    parser.add_argument("--ppo-config", type=Path, default=CONFIGS_DIR / "ppo.yaml")
    parser.add_argument("--simulator-config", type=Path, default=CONFIGS_DIR / "simulator.yaml")
    parser.add_argument("--eval-config", type=Path, default=CONFIGS_DIR / "evaluation.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--multi-seed", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point for the sweep."""
    args = _parse_args()
    simulator_config = load_simulator_config(args.simulator_config)
    base_ppo_config = load_ppo_config(args.ppo_config)
    training_config = load_training_config(args.ppo_config)
    evaluation_config = load_evaluation_config(args.eval_config)
    if args.total_timesteps is not None:
        training_config = replace(training_config, total_timesteps=args.total_timesteps)

    grid = _parse_grid(args.param) or {"learning_rate": [3e-4, 1e-4]}
    run_sweep(
        grid=grid,
        simulator_config=simulator_config,
        base_ppo_config=base_ppo_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        sweep_name=args.sweep_name,
        multi_seed=args.multi_seed,
    )


if __name__ == "__main__":
    main()

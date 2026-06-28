"""Generic training orchestration for RaceMind AI.

:class:`Trainer` drives training in fixed-size chunks, running evaluation and
checkpointing between chunks. It is deliberately algorithm-agnostic: it interacts
only with the :class:`~agents.base_agent.BaseAgent` interface, an
:class:`~evaluation.evaluator.Evaluator` and a
:class:`~training.checkpoint_manager.CheckpointManager`. PPO-specific behaviour
lives entirely inside the agent.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym

from agents.base_agent import BaseAgent
from config.rl import EvaluationConfig, TrainingConfig
from evaluation.evaluator import Evaluator
from training.checkpoint_manager import CheckpointManager
from utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingSummary:
    """A structured summary of a completed training run."""

    experiment_name: str
    algorithm: str
    total_timesteps: int
    elapsed_seconds: float
    timesteps_per_second: float
    best_reward: Optional[float]
    final_reward: Optional[float]
    evaluations: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the summary."""
        data = asdict(self)
        data["evaluations"] = [list(item) for item in self.evaluations]
        return data


class Trainer:
    """Algorithm-agnostic trainer: train in chunks, eval and checkpoint between."""

    def __init__(
        self,
        agent: BaseAgent,
        evaluator: Evaluator,
        checkpoint_manager: CheckpointManager,
        total_timesteps: int,
        evaluation_frequency: int,
        checkpoint_frequency: int,
        experiment_name: str = "experiment",
        algorithm: str = "unknown",
    ) -> None:
        """Initialise the trainer.

        Args:
            agent: The agent to train (interface only).
            evaluator: Runs multi-episode evaluations between chunks.
            checkpoint_manager: Persists latest/best checkpoints.
            total_timesteps: Total environment steps to train for.
            evaluation_frequency: Steps between evaluations.
            checkpoint_frequency: Steps between checkpoints.
            experiment_name: Name used in logs and the summary.
            algorithm: Algorithm label used in the summary.
        """
        if min(total_timesteps, evaluation_frequency, checkpoint_frequency) < 1:
            raise ValueError("timesteps and frequencies must be positive integers.")
        self._agent = agent
        self._evaluator = evaluator
        self._checkpoints = checkpoint_manager
        self._total_timesteps = total_timesteps
        self._eval_frequency = evaluation_frequency
        self._checkpoint_frequency = checkpoint_frequency
        self._experiment_name = experiment_name
        self._algorithm = algorithm

    @property
    def agent(self) -> BaseAgent:
        """The agent being trained (for resume / final evaluation)."""
        return self._agent

    @property
    def evaluator(self) -> Evaluator:
        """The evaluator used between chunks (reused for the final benchmark)."""
        return self._evaluator

    @property
    def checkpoints(self) -> CheckpointManager:
        """The checkpoint manager (used to resume from disk)."""
        return self._checkpoints

    def train(self) -> TrainingSummary:
        """Run the full training schedule and return a summary.

        Returns:
            A :class:`TrainingSummary` describing the run.
        """
        chunk = min(self._eval_frequency, self._checkpoint_frequency)
        steps_done = 0
        first_chunk = True
        last_reward: Optional[float] = None
        evaluations: list[tuple[int, float]] = []

        _logger.info(
            "Training started: %s (%s) for %d timesteps",
            self._experiment_name,
            self._algorithm,
            self._total_timesteps,
        )
        start = time.perf_counter()

        while steps_done < self._total_timesteps:
            step = min(chunk, self._total_timesteps - steps_done)
            self._agent.learn(step, reset_num_timesteps=first_chunk)
            first_chunk = False
            steps_done += step

            at_end = steps_done >= self._total_timesteps
            do_eval = at_end or steps_done % self._eval_frequency == 0
            do_checkpoint = at_end or steps_done % self._checkpoint_frequency == 0

            if do_eval or do_checkpoint:
                last_reward = self._evaluator.evaluate(self._agent).average_reward
                evaluations.append((steps_done, last_reward))
                self._log_progress(steps_done, last_reward, start)

            if do_checkpoint and last_reward is not None:
                self._checkpoints.save(self._agent, metric=last_reward, step=steps_done)

        elapsed = time.perf_counter() - start
        summary = TrainingSummary(
            experiment_name=self._experiment_name,
            algorithm=self._algorithm,
            total_timesteps=steps_done,
            elapsed_seconds=elapsed,
            timesteps_per_second=steps_done / elapsed if elapsed > 0 else 0.0,
            best_reward=self._checkpoints.best_metric,
            final_reward=last_reward,
            evaluations=tuple(evaluations),
        )
        _logger.info(
            "Training finished: %d steps in %.1fs (%.1f steps/s), best=%.2f",
            summary.total_timesteps,
            summary.elapsed_seconds,
            summary.timesteps_per_second,
            summary.best_reward if summary.best_reward is not None else float("nan"),
        )
        return summary

    def _log_progress(self, steps_done: int, reward: float, start: float) -> None:
        """Log a concise per-evaluation progress line."""
        elapsed = time.perf_counter() - start
        fps = steps_done / elapsed if elapsed > 0 else 0.0
        _logger.info(
            "step=%d eval_reward=%.2f elapsed=%.1fs fps=%.1f",
            steps_done,
            reward,
            elapsed,
            fps,
        )


def build_trainer(
    agent: BaseAgent,
    eval_env: gym.Env,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    checkpoint_dir: Path,
    seed: Optional[int] = None,
    experiment_name: str = "experiment",
) -> Trainer:
    """Assemble a :class:`Trainer` around an already-constructed agent.

    Generic over the agent, so it serves PPO, the random baseline and any future
    algorithm identically.

    Args:
        agent: The agent to train.
        eval_env: Environment used for evaluation.
        training_config: Training schedule (timesteps, frequencies, algorithm).
        evaluation_config: Evaluation settings.
        checkpoint_dir: Directory for latest/best checkpoints.
        seed: Optional base seed for evaluation episodes.
        experiment_name: Name used in logs and the summary.

    Returns:
        A ready-to-run :class:`Trainer`.
    """
    evaluator = Evaluator(
        eval_env,
        n_episodes=evaluation_config.n_eval_episodes,
        deterministic=evaluation_config.deterministic,
        success_threshold=evaluation_config.success_threshold,
        max_steps=evaluation_config.max_steps,
        seed=seed,
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    return Trainer(
        agent=agent,
        evaluator=evaluator,
        checkpoint_manager=checkpoint_manager,
        total_timesteps=training_config.total_timesteps,
        evaluation_frequency=training_config.evaluation_frequency,
        checkpoint_frequency=training_config.checkpoint_frequency,
        experiment_name=experiment_name,
        algorithm=training_config.algorithm,
    )

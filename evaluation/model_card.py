"""Model cards for RaceMind AI.

A model card is a small metadata JSON describing a trained model: its algorithm,
environment, training budget, hyperparameters, headline metrics, checkpoint
location and provenance (training date, git commit). It is saved beside the
checkpoint so a model is always self-describing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.io import write_json
from utils.logger import get_logger
from utils.paths import ensure_directory

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelCard:
    """Self-describing metadata for a trained model."""

    algorithm: str
    environment: str
    training_timesteps: int
    seed: int
    hyperparameters: dict[str, Any]
    average_reward: Optional[float]
    best_reward: Optional[float]
    checkpoint_path: Optional[str]
    training_date: str
    git_commit: Optional[str]
    evaluation_results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def build_model_card(
    algorithm: str,
    environment: str,
    training_timesteps: int,
    seed: int,
    hyperparameters: dict[str, Any],
    average_reward: Optional[float],
    best_reward: Optional[float],
    checkpoint_path: Optional[Path],
    git_commit: Optional[str],
    evaluation_results: Optional[dict[str, Any]] = None,
) -> ModelCard:
    """Assemble a :class:`ModelCard` (stamps the current date)."""
    return ModelCard(
        algorithm=algorithm,
        environment=environment,
        training_timesteps=training_timesteps,
        seed=seed,
        hyperparameters=hyperparameters,
        average_reward=average_reward,
        best_reward=best_reward,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        training_date=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit,
        evaluation_results=evaluation_results or {},
    )


def save_model_card(card: ModelCard, checkpoint_dir: Path) -> Path:
    """Write the model card to ``model_card.json`` beside the checkpoints."""
    ensure_directory(checkpoint_dir)
    path = checkpoint_dir / "model_card.json"
    write_json(path, card.to_dict())
    _logger.info("Model card written: %s", path)
    return path

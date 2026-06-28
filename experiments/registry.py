"""Experiment registry for RaceMind AI.

Maintains a persistent index of completed experiments (a JSON file) and supports
lookup by name, algorithm, date, seed, environment or checkpoint. The registry
is the backbone of cross-experiment ranking and reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

from config.simulator import SimulatorConfig, default_config
from utils.io import read_json, write_json
from utils.logger import get_logger

_logger = get_logger(__name__)

REGISTRY_FILENAME: str = "registry.json"


@dataclass(frozen=True)
class ExperimentRecord:
    """A single completed-experiment entry in the registry."""

    experiment_name: str
    algorithm: str
    environment: str
    seed: int
    date: str
    total_timesteps: int
    average_reward: Optional[float] = None
    best_reward: Optional[float] = None
    training_seconds: Optional[float] = None
    checkpoint_path: Optional[str] = None
    report_path: Optional[str] = None
    model_card_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        """Build a record from a mapping, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def default_registry_path(config: Optional[SimulatorConfig] = None) -> Path:
    """Return the default registry path under the evaluation directory."""
    config = config or default_config()
    return config.evaluation_dir / REGISTRY_FILENAME


class ExperimentRegistry:
    """A persistent, JSON-backed index of completed experiments."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        """Load the registry from disk (or start empty).

        Args:
            registry_path: Path to the registry JSON; a sensible default under
                the evaluation directory is used when omitted.
        """
        self._path = registry_path or default_registry_path()
        self._records: list[ExperimentRecord] = self._load()

    def _load(self) -> list[ExperimentRecord]:
        """Load records from disk, returning an empty list if absent."""
        if not self._path.exists():
            return []
        payload = read_json(self._path)
        return [ExperimentRecord.from_dict(item) for item in payload.get("experiments", [])]

    def register(self, record: ExperimentRecord) -> None:
        """Add ``record`` (replacing any existing entry with the same name)."""
        self._records = [r for r in self._records if r.experiment_name != record.experiment_name]
        self._records.append(record)
        self._save()
        _logger.info("Experiment registered: %s", record.experiment_name)

    def _save(self) -> None:
        """Persist the registry to disk."""
        write_json(self._path, {"experiments": [r.to_dict() for r in self._records]})

    def all(self) -> list[ExperimentRecord]:
        """Return all registered records."""
        return list(self._records)

    def find(self, experiment_name: str) -> Optional[ExperimentRecord]:
        """Return the record with ``experiment_name``, or ``None``."""
        return next((r for r in self._records if r.experiment_name == experiment_name), None)

    def find_by_checkpoint(self, checkpoint_path: str) -> Optional[ExperimentRecord]:
        """Return the record referencing ``checkpoint_path``, or ``None``."""
        return next((r for r in self._records if r.checkpoint_path == checkpoint_path), None)

    def filter(
        self,
        algorithm: Optional[str] = None,
        seed: Optional[int] = None,
        environment: Optional[str] = None,
        date: Optional[str] = None,
    ) -> list[ExperimentRecord]:
        """Return records matching every provided (non-``None``) criterion.

        Args:
            algorithm: Match on algorithm (case-insensitive).
            seed: Match on seed.
            environment: Match on environment (case-insensitive).
            date: Match records whose date starts with this prefix (e.g. a day).

        Returns:
            The matching records.
        """
        def matches(record: ExperimentRecord) -> bool:
            if algorithm is not None and record.algorithm.lower() != algorithm.lower():
                return False
            if seed is not None and record.seed != seed:
                return False
            if environment is not None and record.environment.lower() != environment.lower():
                return False
            if date is not None and not record.date.startswith(date):
                return False
            return True

        return [record for record in self._records if matches(record)]

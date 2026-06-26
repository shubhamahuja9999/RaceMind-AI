"""Configuration package for RaceMind AI.

Configuration is split into focused modules: :mod:`config.simulator`,
:mod:`config.experiment`, :mod:`config.logging` and :mod:`config.paths`. The
most common names are re-exported here for convenience.
"""

from config.experiment import ExperimentConfig
from config.logging import LoggingConfig
from config.paths import PROJECT_ROOT
from config.simulator import SimulatorConfig, default_config

__all__ = [
    "PROJECT_ROOT",
    "SimulatorConfig",
    "default_config",
    "ExperimentConfig",
    "LoggingConfig",
]

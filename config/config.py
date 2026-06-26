"""Backwards-compatibility shim for the original configuration module.

The configuration was split into focused modules during Phase 1.5
(:mod:`config.simulator`, :mod:`config.experiment`, :mod:`config.logging`,
:mod:`config.paths`). This module re-exports the original public names so that
existing imports such as ``from config.config import SimulatorConfig`` keep
working. New code should import from the dedicated modules instead.
"""

from __future__ import annotations

from config.paths import PROJECT_ROOT
from config.simulator import SimulatorConfig, default_config

__all__ = ["PROJECT_ROOT", "SimulatorConfig", "default_config"]

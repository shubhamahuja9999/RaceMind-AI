"""Environment factory for RaceMind AI.

:func:`make_env` is the single, canonical entry point for creating simulator
environments. It builds the raw Gymnasium environment, applies the standard
wrapper stack and returns a :class:`~simulator.env.RaceEnv`. Centralising
construction here means future changes to wrapping or environment setup happen
in exactly one place — which is what Phase 2 (PPO training) will depend on.
"""

from __future__ import annotations

from typing import Optional

from config.simulator import SimulatorConfig, default_config
from simulator.env import RaceEnv, build_gym_env
from simulator.wrappers import wrap_environment
from utils.logger import get_logger

_logger = get_logger(__name__)


def make_env(
    config: Optional[SimulatorConfig] = None,
    apply_wrappers: bool = True,
) -> RaceEnv:
    """Create a fully wired simulator environment.

    Args:
        config: Simulator configuration; the framework default is used when
            omitted.
        apply_wrappers: When ``True`` (default), apply the standard wrapper
            stack from :func:`simulator.wrappers.wrap_environment`.

    Returns:
        A :class:`RaceEnv` wrapping the (optionally wrapped) Gymnasium env.
    """
    config = config or default_config()
    env = build_gym_env(config)
    if apply_wrappers:
        env = wrap_environment(env)
        _logger.debug("Applied standard wrapper stack.")
    return RaceEnv(config, env=env)

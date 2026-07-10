"""Environment factory for RaceMind AI.

:func:`make_env` is the single, canonical entry point for creating simulator
environments. It builds the raw Gymnasium environment, applies the standard
wrapper stack and returns a :class:`~simulator.env.RaceEnv`. Centralising
construction here means future changes to wrapping or environment setup happen
in exactly one place — which is what Phase 2 (PPO training) will depend on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from config.simulator import SimulatorConfig, default_config
from simulator.env import RaceEnv, build_gym_env
from simulator.wrappers import wrap_environment
from utils.logger import get_logger
from utils.paths import ensure_directory

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


def make_training_env(
    config: Optional[SimulatorConfig] = None,
    frame_stack: int = 0,
) -> gym.Env:
    """Create a raw (gym.Env) environment for RL training.

    Returns a wrapped Gymnasium environment — not a :class:`RaceEnv` — because
    Stable-Baselines3 operates on ``gym.Env``. Rendering is disabled for speed.

    Args:
        config: Simulator configuration; the default is used when omitted.
        frame_stack: When > 0, stack this many consecutive frames (temporal
            observation). ``0`` (default) matches the baseline single-frame env.

    Returns:
        The wrapped training environment.
    """
    config = config or default_config()
    env = build_gym_env(config, render_mode=None)
    return wrap_environment(env, frame_stack=frame_stack)


def make_eval_env(
    config: Optional[SimulatorConfig] = None,
    record_video: bool = False,
    video_dir: Optional[Path] = None,
    name_prefix: str = "eval",
    frame_stack: int = 0,
) -> gym.Env:
    """Create a raw (gym.Env) environment for evaluation, optionally recording.

    Uses ``render_mode="rgb_array"`` so frames are available to Gymnasium's
    :class:`~gymnasium.wrappers.RecordVideo` wrapper. Video frames are captured
    by the wrapper — never recorded manually.

    Args:
        config: Simulator configuration; the default is used when omitted.
        record_video: When ``True``, wrap the env in ``RecordVideo``.
        video_dir: Directory for saved videos (required if ``record_video``).
        name_prefix: Filename prefix for recorded videos.
        frame_stack: When > 0, stack this many consecutive frames. Must match the
            value used at training time.

    Returns:
        The wrapped (and optionally recording) evaluation environment.
    """
    config = config or default_config()
    env = build_gym_env(config, render_mode="rgb_array")
    env = wrap_environment(env, frame_stack=frame_stack)

    if record_video:
        if video_dir is None:
            raise ValueError("video_dir is required when record_video is True.")
        ensure_directory(video_dir)
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: True,
            name_prefix=name_prefix,
        )
        _logger.info("Evaluation video recording enabled: %s", video_dir)

    return env

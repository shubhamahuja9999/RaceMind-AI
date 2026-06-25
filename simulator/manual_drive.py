"""Keyboard-driven manual control for the RaceMind AI simulator.

Run this module to drive the ``CarRacing-v3`` car yourself with the arrow keys.
Telemetry is always logged; episode recording is optional. The recorded frame
is the environment's own RGB observation, so recording works regardless of the
render mode.

Controls:
    Arrow Up    -> accelerate
    Arrow Down  -> brake
    Arrow Left  -> steer left
    Arrow Right -> steer right
    ESC         -> quit
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pygame

from config.config import SimulatorConfig, default_config
from simulator.env import RaceEnv
from simulator.recorder import EpisodeRecorder
from simulator.telemetry import TelemetryLogger

# Magnitudes applied to each control axis when a key is held.
STEER_MAGNITUDE: float = 1.0
THROTTLE_MAGNITUDE: float = 1.0
BRAKE_MAGNITUDE: float = 0.8


class KeyboardController:
    """Translates held arrow keys into a ``CarRacing`` action vector."""

    def __init__(self) -> None:
        """Initialise the controller with all keys released."""
        self._left = False
        self._right = False
        self._up = False
        self._down = False
        self._quit = False

    @property
    def quit_requested(self) -> bool:
        """Whether the user has asked to quit (ESC or window close)."""
        return self._quit

    def process_events(self) -> None:
        """Drain the pygame event queue and update internal key state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
            elif event.type == pygame.KEYDOWN:
                self._set_key(event.key, pressed=True)
            elif event.type == pygame.KEYUP:
                self._set_key(event.key, pressed=False)

    def _set_key(self, key: int, pressed: bool) -> None:
        """Update the pressed state for a single key."""
        if key == pygame.K_ESCAPE:
            self._quit = pressed
        elif key == pygame.K_LEFT:
            self._left = pressed
        elif key == pygame.K_RIGHT:
            self._right = pressed
        elif key == pygame.K_UP:
            self._up = pressed
        elif key == pygame.K_DOWN:
            self._down = pressed

    def current_action(self) -> np.ndarray:
        """Build the current action vector from the held keys.

        Returns:
            A ``[steering, gas, brake]`` float32 action.
        """
        steering = 0.0
        if self._left:
            steering -= STEER_MAGNITUDE
        if self._right:
            steering += STEER_MAGNITUDE

        throttle = THROTTLE_MAGNITUDE if self._up else 0.0
        brake = BRAKE_MAGNITUDE if self._down else 0.0
        return np.array([steering, throttle, brake], dtype=np.float32)


class ManualDriveSession:
    """Runs an interactive manual-driving episode with logging and recording."""

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        """Initialise the session.

        Args:
            config: Simulator configuration; the default is used when omitted.
        """
        self._config: SimulatorConfig = config or default_config()
        self._controller = KeyboardController()
        self._telemetry = TelemetryLogger(self._config)
        self._recorder = EpisodeRecorder(self._config)

    def run(self, episode: int = 0, record: bool = False) -> None:
        """Drive a single episode interactively.

        Args:
            episode: Episode index used for naming telemetry/recording files.
            record: When ``True``, save the episode as a ``.npz`` recording.
        """
        # The human render mode is required so the player can see the track.
        config = self._config
        if config.render_mode != "human":
            print("[manual_drive] Forcing render_mode='human' for manual control.")

        with RaceEnv(config) as env:
            observation, _ = env.reset()
            self._telemetry.start_episode(episode)
            self._recorder.start_episode(episode)
            print("Manual driving started. Arrow keys to drive, ESC to quit.")

            self._drive_loop(env, observation, record)

        self._persist(record)

    def _drive_loop(
        self,
        env: RaceEnv,
        observation: np.ndarray,
        record: bool,
    ) -> None:
        """Step the environment until the episode ends or the user quits."""
        clock = pygame.time.Clock()
        done = False
        while not done and not self._controller.quit_requested:
            self._controller.process_events()
            action = self._controller.current_action()

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self._telemetry.log(action, reward, done, step=env.step_count)
            if record:
                self._recorder.record(observation, action, reward)

            env.render()
            clock.tick(self._config.fps)

    def _persist(self, record: bool) -> None:
        """Write telemetry (always) and the recording (when requested)."""
        telemetry_path = self._telemetry.save()
        print(f"Telemetry saved to: {telemetry_path}")
        if record and self._recorder.num_frames:
            recording_path = self._recorder.save(extra_metadata={"mode": "manual"})
            print(f"Recording saved to: {recording_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the manual-driving entry point."""
    parser = argparse.ArgumentParser(description="Drive CarRacing-v3 with the keyboard.")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index used when naming output files.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save the episode as a compressed .npz recording.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point for manual driving."""
    args = _parse_args()
    session = ManualDriveSession(default_config())
    session.run(episode=args.episode, record=args.record)


if __name__ == "__main__":
    main()

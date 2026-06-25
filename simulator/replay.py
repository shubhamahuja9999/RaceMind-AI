"""Standalone replay viewer for recorded RaceMind AI episodes.

This module is deliberately independent of the live simulator: it does not
import Gymnasium or :mod:`simulator.env`. It only needs the ``.npz`` files
produced by :class:`~simulator.recorder.EpisodeRecorder`, so recorded episodes
can be reviewed on a machine that cannot run the environment at all.

Each frame is shown with an overlay reporting the frame number, the reward for
that frame and the steering input that produced it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pygame is only needed for on-screen display, not for loading.
    import pygame
except ImportError:  # pragma: no cover - display is optional
    pygame = None  # type: ignore[assignment]

from config.config import SimulatorConfig, default_config
from simulator.recorder import (
    ACTIONS_KEY,
    FRAMES_KEY,
    METADATA_KEY,
    REWARDS_KEY,
)


@dataclass
class RecordedEpisode:
    """In-memory representation of a single recorded episode."""

    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    metadata: dict

    @property
    def num_frames(self) -> int:
        """Number of frames in the recording."""
        return int(self.frames.shape[0])


def load_episode(path: Path) -> RecordedEpisode:
    """Load a recorded episode from a ``.npz`` file.

    Args:
        path: Path to the ``.npz`` recording.

    Returns:
        The decoded :class:`RecordedEpisode`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Recording not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        frames = archive[FRAMES_KEY]
        actions = archive[ACTIONS_KEY]
        rewards = archive[REWARDS_KEY]
        metadata = json.loads(str(archive[METADATA_KEY]))

    return RecordedEpisode(
        frames=frames,
        actions=actions,
        rewards=rewards,
        metadata=metadata,
    )


class ReplayPlayer:
    """Renders a :class:`RecordedEpisode` to a pygame window with an overlay."""

    def __init__(self, config: Optional[SimulatorConfig] = None) -> None:
        """Initialise the player.

        Args:
            config: Simulator configuration; the default is used when omitted.
        """
        self._config: SimulatorConfig = config or default_config()

    def _draw_overlay(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        frame_index: int,
        total_frames: int,
        reward: float,
        steering: float,
    ) -> None:
        """Draw the informational overlay onto ``surface``."""
        lines = [
            f"Frame   : {frame_index + 1}/{total_frames}",
            f"Reward  : {reward:+.3f}",
            f"Steering: {steering:+.3f}",
        ]
        for row, text in enumerate(lines):
            shadow = font.render(text, True, (0, 0, 0))
            label = font.render(text, True, (255, 255, 255))
            position = (10, 10 + row * 24)
            surface.blit(shadow, (position[0] + 1, position[1] + 1))
            surface.blit(label, position)

    def play(self, episode: RecordedEpisode) -> None:
        """Replay every frame of ``episode`` until it ends or the user quits.

        Controls: ``ESC`` or closing the window stops playback.

        Args:
            episode: The recording to play back.

        Raises:
            RuntimeError: If pygame is not installed.
        """
        if pygame is None:
            raise RuntimeError(
                "pygame is required for replay display. "
                "Install dependencies with: pip install -r requirements.txt"
            )
        pygame.init()
        try:
            screen = pygame.display.set_mode(self._config.window_size)
            pygame.display.set_caption("RaceMind AI :: Replay")
            font = pygame.font.SysFont("monospace", 18, bold=True)
            clock = pygame.time.Clock()

            for index in range(episode.num_frames):
                if self._should_quit():
                    break

                reward = float(episode.rewards[index])
                steering = float(episode.actions[index][0])
                self._render_frame(screen, episode.frames[index])
                self._draw_overlay(
                    screen,
                    font,
                    index,
                    episode.num_frames,
                    reward,
                    steering,
                )
                pygame.display.flip()
                clock.tick(self._config.fps)
        finally:
            pygame.quit()

    @staticmethod
    def _should_quit() -> bool:
        """Return ``True`` if the user requested to stop playback."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def _render_frame(self, screen: pygame.Surface, frame: np.ndarray) -> None:
        """Scale ``frame`` to the window and blit it onto ``screen``."""
        # pygame expects (width, height, 3); the recorded frame is (H, W, 3).
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surface, self._config.window_size)
        screen.blit(scaled, (0, 0))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the replay entry point."""
    parser = argparse.ArgumentParser(description="Replay a recorded RaceMind AI episode.")
    parser.add_argument(
        "recording",
        type=Path,
        help="Path to the .npz recording produced by EpisodeRecorder.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: load a recording and play it back."""
    args = _parse_args()
    config = default_config()
    episode = load_episode(args.recording)
    print(
        f"Loaded episode {episode.metadata.get('episode')} "
        f"with {episode.num_frames} frames "
        f"(total reward = {episode.metadata.get('total_reward'):+.3f})"
    )
    ReplayPlayer(config).play(episode)


if __name__ == "__main__":
    main()

"""Inference engine for the RaceMind AI Research Lab showcase.

Loads the trained PPO model **once** and runs single deterministic episodes on
``CarRacing-v3``, collecting rendered frames and encoding them to MP4. Results are
cached by ``(policy, seed)`` so repeated demos are instant.

The engine depends only on Stable-Baselines3 + Gymnasium (not the parent RaceMind
package), so the backend deploys independently — you only need the model file.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

ENV_ID = "CarRacing-v3"
MAX_EPISODE_STEPS = 1000
VIDEO_FPS = 30
FRAME_STRIDE = 2  # keep every 2nd frame in the MP4 (faster, smaller)


@dataclass
class EpisodeResult:
    """Metrics and artifact for one rolled-out episode."""

    policy: str
    seed: int
    reward: float
    length: int
    inference_seconds: float
    steps_per_second: float
    video_filename: str

    def to_dict(self, video_base: str = "/videos") -> dict:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "reward": round(self.reward, 2),
            "length": self.length,
            "inference_seconds": round(self.inference_seconds, 2),
            "steps_per_second": round(self.steps_per_second, 1),
            "video_url": f"{video_base}/{self.video_filename}",
        }


@dataclass
class InferenceEngine:
    """Holds the loaded model and a result cache."""

    model_path: Path
    video_dir: Path
    default_max_steps: int = 600
    model_url: Optional[str] = None
    _model: Optional[PPO] = field(default=None, init=False)
    _cache: dict[tuple[str, int], EpisodeResult] = field(default_factory=dict, init=False)

    def load(self) -> None:
        """Load the PPO model once (called at startup).

        The model file is git-ignored, so on a fresh deploy it may be absent. If
        ``MODEL_URL`` is set (e.g. a GitHub release asset), it is downloaded to
        ``model_path`` on first startup.
        """
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists() and self.model_url:
            self._download_model()
        self._model = PPO.load(str(self.model_path), device="cpu")

    def _download_model(self) -> None:
        """Download the model from ``model_url`` to ``model_path``."""
        import urllib.request

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[startup] downloading model from {self.model_url}")
        urllib.request.urlretrieve(self.model_url, str(self.model_path))
        print(f"[startup] model saved to {self.model_path}")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _make_env(self) -> gym.Env:
        return gym.make(ENV_ID, render_mode="rgb_array", continuous=True,
                        max_episode_steps=MAX_EPISODE_STEPS)

    def run_episode(self, policy: str = "best", seed: int = 1028,
                    max_steps: Optional[int] = None) -> EpisodeResult:
        """Run one deterministic episode; cache and return the result."""
        key = (policy, seed)
        if key in self._cache and (self.video_dir / self._cache[key].video_filename).exists():
            return self._cache[key]

        if policy != "random" and self._model is None:
            raise RuntimeError("Model not loaded.")

        max_steps = max_steps or self.default_max_steps
        env = self._make_env()
        rng = np.random.default_rng(seed)
        observation, _ = env.reset(seed=seed)

        frames: list[np.ndarray] = []
        total_reward = 0.0
        steps = 0
        done = False
        start = time.perf_counter()
        while not done and steps < max_steps:
            frames.append(env.render())
            if policy == "random":
                action = env.action_space.sample()
            else:
                action, _ = self._model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        elapsed = time.perf_counter() - start
        env.close()

        filename = f"{policy}_{seed}.mp4"
        self._encode(frames, self.video_dir / filename)
        result = EpisodeResult(
            policy=policy, seed=seed, reward=total_reward, length=steps,
            inference_seconds=elapsed,
            steps_per_second=(steps / elapsed if elapsed > 0 else 0.0),
            video_filename=filename,
        )
        self._cache[key] = result
        return result

    @staticmethod
    def _encode(frames: list[np.ndarray], path: Path) -> None:
        """Encode frames to an MP4 (H.264 via imageio-ffmpeg)."""
        selected = frames[::FRAME_STRIDE] or frames
        imageio.mimsave(str(path), selected, fps=VIDEO_FPS, codec="libx264",
                        macro_block_size=None, quality=7)


def build_engine() -> InferenceEngine:
    """Construct the engine from environment variables."""
    default_model = Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "ppo_1m" / "best.zip"
    model_path = Path(os.getenv("MODEL_PATH", str(default_model)))
    video_dir = Path(os.getenv("VIDEO_DIR", str(Path(__file__).resolve().parent / "videos")))
    max_steps = int(os.getenv("MAX_STEPS", "600"))
    model_url = os.getenv("MODEL_URL")  # optional: download the model on startup
    return InferenceEngine(model_path=model_path, video_dir=video_dir,
                           default_max_steps=max_steps, model_url=model_url)

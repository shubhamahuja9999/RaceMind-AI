"""RaceMind AI Research Lab — FastAPI inference backend.

Loads the trained PPO model once at startup and exposes a small inference API:
run the best agent or a random agent for one deterministic episode, returning a
rendered MP4 plus metrics. No database, no auth — a pure inference showcase.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import build_engine

engine = build_engine()

MODEL_INFO = {
    "name": "ppo_1m/best.zip",
    "algorithm": "PPO",
    "policy": "CnnPolicy",
    "environment": "CarRacing-v3",
    "training_timesteps": 1_000_000,
    "mean_reward": 598.42,
    "ci95": [544.78, 652.05],
    "eval_protocol": "30 deterministic episodes, fixed seeds 1000-1029",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, in a background thread.

    Loading (and possibly downloading) the model can take tens of seconds. Doing
    it in the background lets the HTTP port open immediately, so the platform's
    health check passes right away instead of timing out during startup. Until
    the model is ready, ``/api/health`` reports ``model_loaded: false`` and
    ``/api/run`` returns 503 (the frontend handles this gracefully).
    """
    def _load() -> None:
        try:
            engine.load()
            print("[startup] model loaded")
        except Exception as exc:  # noqa: BLE001 — surface a clear startup error
            print(f"[startup] model load failed: {exc}")

    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="RaceMind AI Research Lab", version="1.0.0", lifespan=lifespan)

_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine.video_dir.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(engine.video_dir)), name="videos")


class RunRequest(BaseModel):
    """Optional parameters for POST /api/run."""

    policy: str = Field(default="best", description="'best' or 'random'")
    seed: int = Field(default=1028, description="Evaluation seed (deterministic).")


@app.get("/api/health")
def health() -> dict:
    """Liveness + whether the model is loaded."""
    return {"status": "ok", "model_loaded": engine.loaded}


@app.get("/api/model")
def model_info() -> dict:
    """Static metadata about the deployed model."""
    return {**MODEL_INFO, "model_loaded": engine.loaded}


# Only one inference at a time — prevents abuse / GPU/CPU overload.
_inference_lock = threading.Lock()


def _run(policy: str, seed: int) -> dict:
    """Run one episode and return the serialised result."""
    if policy not in ("best", "random"):
        raise HTTPException(status_code=400, detail="policy must be 'best' or 'random'")
    if policy == "best" and not engine.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not _inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="An inference is already running. Please wait and try again.",
        )
    try:
        return engine.run_episode(policy=policy, seed=seed).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _inference_lock.release()


@app.post("/api/run")
def run(request: RunRequest) -> dict:
    """Run one deterministic episode for the requested policy."""
    return _run(request.policy, request.seed)


@app.get("/api/baseline")
def baseline(seed: int = 1028) -> dict:
    """Run the best PPO agent for one deterministic episode."""
    return _run("best", seed)


@app.get("/api/random")
def random_agent(seed: int = 1028) -> dict:
    """Run a random agent for one deterministic episode."""
    return _run("random", seed)

# RaceMind AI

> An autonomous racing intelligence platform — built incrementally toward
> Reinforcement Learning.

**RaceMind AI** is a long-term machine learning project. Its end goal is an
autonomous racing agent trained with Reinforcement Learning on top of
Gymnasium's `CarRacing-v3` environment.

This repository currently contains **Phase 1**: a clean, modular **simulator
framework** that everything else will be built on. No RL, neural networks or
training code exists yet — Phase 1 is purely the simulator boundary, manual
control, data capture and replay tooling.

---

## Project Overview

Phase 1 delivers five focused capabilities, each in its own module:

| Capability        | Module                      | Description                                                        |
| ----------------- | --------------------------- | ------------------------------------------------------------------ |
| Environment       | `simulator/env.py`          | Object-oriented wrapper around `CarRacing-v3` (`reset/step/close`). |
| Manual driving    | `simulator/manual_drive.py` | Drive the car yourself with the arrow keys (PyGame).               |
| Telemetry         | `simulator/telemetry.py`    | Per-frame logging of reward and controls to CSV.                  |
| Recording         | `simulator/recorder.py`     | Save frames, actions, rewards and metadata to compressed `.npz`.  |
| Replay            | `simulator/replay.py`       | Replay a recorded episode independently of the simulator.         |

Cross-cutting concerns are isolated too: **all configuration** lives in
`config/config.py` (no magic numbers, no hardcoded paths) and **reusable
helpers** live in `simulator/utils.py`.

---

## Architecture

```
                ┌────────────────────────┐
                │   config/config.py     │  Single source of truth:
                │  SimulatorConfig       │  paths, FPS, window size,
                └───────────┬────────────┘  env name, render mode.
                            │ (injected everywhere)
        ┌───────────────────┼───────────────────────┐
        │                   │                       │
┌───────▼───────┐   ┌───────▼────────┐      ┌───────▼────────┐
│  env.py       │   │ telemetry.py   │      │  recorder.py   │
│  RaceEnv      │   │ TelemetryLogger│      │ EpisodeRecorder│
│ (Gymnasium)   │   │  -> CSV        │      │  -> .npz       │
└───────┬───────┘   └───────┬────────┘      └───────┬────────┘
        │                   │                       │
        │            ┌──────▼───────────────────────▼──────┐
        └───────────▶│        manual_drive.py              │
                     │  ManualDriveSession (keyboard)      │
                     └─────────────────────────────────────┘

        ┌─────────────────────────────────────────────────┐
        │  replay.py — reads .npz only, NO Gymnasium import │
        │  ReplayPlayer (independent of the live simulator) │
        └─────────────────────────────────────────────────┘

        simulator/utils.py — timestamps, dirs, CSV, naming (shared)
```

**Design principles**

- Configuration is **injected**, never imported ad-hoc — every component takes a
  `SimulatorConfig`.
- `replay.py` depends only on the recording format, so recordings can be viewed
  on machines that cannot run the environment.
- OOP for stateful components (`RaceEnv`, loggers), dataclasses for plain data
  (`TelemetryRecord`, `EpisodeMetadata`, `RecordedEpisode`), small typed
  functions everywhere.

---

## Folder Structure

```
racemind-ai/
│
├── simulator/
│   ├── __init__.py
│   ├── env.py             # CarRacing-v3 wrapper
│   ├── manual_drive.py    # keyboard control
│   ├── telemetry.py       # per-frame CSV logging
│   ├── recorder.py        # .npz episode recording
│   ├── replay.py          # standalone replay viewer
│   └── utils.py           # shared helpers
│
├── config/
│   ├── __init__.py
│   └── config.py          # SimulatorConfig (no magic numbers)
│
├── data/
│   ├── telemetry/         # generated CSV files (git-ignored)
│   └── recordings/        # generated .npz files (git-ignored)
│
├── notebooks/             # analysis notebooks (later)
├── docs/                  # documentation (later)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Python **3.12+** is recommended (the code also runs on 3.10+).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Box2D note:** `CarRacing-v3` requires the Box2D backend, which is installed
> via the `gymnasium[box2d]` and `box2d-py` requirements. On some platforms a C++
> build toolchain (e.g. SWIG / build tools) may be needed to compile it.

All commands below are run from the project root and use Python's module syntax
(`-m`) so package imports resolve correctly.

---

## Running the Environment

Run a random-action smoke test that prints debugging information (observation
shape, action/observation spaces, per-step and cumulative reward):

```bash
python -m simulator.env
```

This trains nothing — it simply confirms the environment is wired correctly.

---

## Manual Driving

Drive the car yourself:

```bash
python -m simulator.manual_drive
```

**Controls**

| Key          | Action       |
| ------------ | ------------ |
| Arrow Up     | Accelerate   |
| Arrow Down   | Brake        |
| Arrow Left   | Steer left   |
| Arrow Right  | Steer right  |
| ESC          | Quit         |

Telemetry is written automatically to `data/telemetry/` at the end of the run.

---

## Recording Episodes

Add the `--record` flag to also save a compressed `.npz` recording (frames,
actions, rewards and metadata) to `data/recordings/`:

```bash
python -m simulator.manual_drive --record --episode 1
```

`--episode` sets the index used to name the generated files.

---

## Replaying Episodes

Replay a recorded episode. The replay viewer is **independent of the
simulator** — it only reads the `.npz` file and shows each frame with an overlay
displaying the frame number, reward and steering input:

```bash
python -m simulator.replay data/recordings/episode_001_<timestamp>.npz
```

Press **ESC** or close the window to stop playback.

---

## Future Roadmap

Phase 1 (this repository) is the foundation. Planned next phases:

- **Phase 2 — Observation & reward engineering:** preprocessing wrappers
  (frame stacking, grayscale, resizing), custom reward shaping, vectorized envs.
- **Phase 3 — RL training:** integrate Stable-Baselines3 / PyTorch (already
  installed) to train PPO/SAC agents; checkpointing and evaluation.
- **Phase 4 — Analysis & visualization:** telemetry dashboards and notebooks for
  comparing runs (using the data captured in Phase 1).
- **Phase 5 — Autonomous racing intelligence:** self-improving agents, track
  generalization and benchmarking.

> The Phase 1 simulator framework is intentionally decoupled so each later phase
> can build on it without rewrites.

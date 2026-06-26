# RaceMind AI

> An autonomous racing intelligence platform — built incrementally toward
> Reinforcement Learning.

**RaceMind AI** is a long-term machine learning project. Its end goal is an
autonomous racing agent trained with Reinforcement Learning on top of
Gymnasium's `CarRacing-v3` environment.

The repository currently spans **Phase 1** (the simulator framework) and
**Phase 1.5** (research-platform infrastructure). No RL, neural networks or
training code exists yet — the focus is a clean, reproducible foundation that
makes Phase 2 (PPO training) straightforward.

- **Phase 1** — simulator boundary, manual driving, telemetry, recording, replay.
- **Phase 1.5** — environment factory, Gymnasium wrappers, professional logging,
  extended telemetry, evaluation metrics, experiment configuration, callback
  interfaces, a structured data layout and global seeding.

---

## Project Overview

| Capability         | Module                                    | Description                                                  |
| ------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| Environment        | `simulator/env.py`                        | OO wrapper around `CarRacing-v3` (`reset/step/close`).       |
| Environment factory| `simulator/environment_factory.py`        | `make_env(config)` — the only place envs are created.        |
| Wrappers           | `simulator/wrappers/`                     | Gymnasium-convention observation/reward/action wrappers.     |
| Manual driving     | `simulator/manual_drive.py`               | Drive with the arrow keys (PyGame).                          |
| Telemetry          | `simulator/telemetry.py`                  | Per-frame logging (wide, optional-field schema) to CSV.      |
| Recording          | `simulator/recorder.py`                   | Frames + actions + rewards + metadata to compressed `.npz`.  |
| Replay             | `simulator/replay.py`                     | Replay a recording independently of the simulator.           |
| Evaluation         | `evaluation/`                             | Episode metrics and serialisable summaries (dataclasses).    |
| Logging            | `utils/logger.py`                         | Console + `logs/project.log`, configured once.               |
| Seeding            | `utils/seed.py`                           | `set_global_seed` — Python / NumPy / PyTorch (if installed). |
| Configuration      | `config/`                                 | Simulator, experiment, logging and path configuration.       |
| Callbacks          | `training/callbacks/`                     | Reusable training-callback interfaces (no RL logic yet).     |

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
                 │                config/                     │
                 │  paths.py · simulator.py · experiment.py   │
                 │  logging.py   (config.py = compat shim)    │
                 └───────────────────┬────────────────────────┘
                                     │ (configuration injected)
        ┌───────────────────────────┼────────────────────────────┐
        │                           │                            │
┌───────▼────────────┐   ┌──────────▼───────────┐    ┌───────────▼─────────┐
│ environment_factory │   │      utils/          │    │     evaluation/     │
│   make_env(config)  │   │ logger · seed · io   │    │ metrics · summaries │
└───────┬─────────────┘   │ paths                │    └─────────────────────┘
        │ builds + wraps   └──────────────────────┘
        ▼
┌────────────────────┐     ┌───────────────────────────────────────────────┐
│  simulator/env.py  │     │  simulator/wrappers/                            │
│      RaceEnv       │◀────│  Observation · Reward · Action (identity today) │
└───────┬────────────┘     └───────────────────────────────────────────────┘
        │
        ▼
┌────────────────────┐   telemetry.py (CSV) · recorder.py (.npz) · replay.py
│  manual_drive.py   │
└────────────────────┘   training/callbacks/ — interfaces for Phase 2

  simulator/utils.py → re-exports utils/ (backwards compatibility shim)
```

**Design principles**

- **Single creation point:** environments are built only via `make_env`.
- **Configuration is injected**, never imported ad-hoc.
- **Low coupling:** `replay.py` and `evaluation/` depend only on data formats, not
  on the live environment; `utils/` is dependency-light and import-cycle-free.
- **Backwards compatibility:** `config/config.py` and `simulator/utils.py` remain
  as thin re-export shims so Phase 1 imports keep working.
- OOP for stateful components, dataclasses for plain data, full type hints.

---

## Folder Structure

```
racemind-ai/
│
├── config/
│   ├── paths.py            # PROJECT_ROOT + canonical data layout
│   ├── simulator.py        # SimulatorConfig
│   ├── experiment.py       # ExperimentConfig (composes SimulatorConfig)
│   ├── logging.py          # LoggingConfig
│   └── config.py           # backwards-compat shim
│
├── utils/
│   ├── paths.py            # dirs, timestamps, naming
│   ├── io.py               # CSV / JSON / atomic writes
│   ├── logger.py           # project logging setup
│   └── seed.py             # set_global_seed
│
├── simulator/
│   ├── env.py              # RaceEnv + build_gym_env
│   ├── environment_factory.py
│   ├── wrappers/           # base / observation / reward / action
│   ├── manual_drive.py
│   ├── telemetry.py
│   ├── recorder.py
│   ├── replay.py
│   └── utils.py            # backwards-compat shim → utils/
│
├── evaluation/
│   ├── metrics.py          # EpisodeMetrics + compute_episode_metrics
│   └── episode_summary.py  # EpisodeSummary + summarize_recording
│
├── training/
│   └── callbacks/          # base / checkpoint / logging / video
│
├── experiments/            # ppo/ sac/ a2c/ imitation/ (placeholders)
│
├── data/                   # telemetry/ recordings/ videos/ plots/
│                           # evaluation/ models/ checkpoints/ (git-ignored)
├── logs/                   # project.log (git-ignored)
├── notebooks/ · docs/
│
├── requirements.txt · README.md · .gitignore
```

---

## Installation

Python **3.12+** is recommended.

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

> Box2D ships as a pre-built wheel (`box2d==2.3.10`), so no C++ compiler is
> required. All commands below run from the project root and use Python's `-m`
> module syntax so package imports resolve correctly.

---

## Developer Setup & Usage

**Run the environment** (random-action smoke test with debug logging):

```bash
python -m simulator.env
```

**Drive manually** (telemetry is always logged; `--record` saves an `.npz`):

```bash
python -m simulator.manual_drive
python -m simulator.manual_drive --record --episode 1
```

| Key | Action | | Key | Action |
| --- | ------ | --- | --- | ------ |
| ↑ | Accelerate | | ← / → | Steer |
| ↓ | Brake | | ESC | Quit |

**Replay a recording** (independent of the simulator):

```bash
python -m simulator.replay data/recordings/episode_001_<timestamp>.npz
```

**Create an environment in code** (always via the factory):

```python
from simulator.environment_factory import make_env
from config.simulator import SimulatorConfig

config = SimulatorConfig(render_mode="rgb_array")
with make_env(config) as env:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.sample_action())
```

**Logging** is configured automatically on first use and writes to both the
console and `logs/project.log`:

```
2026-06-25 18:41:22 INFO Environment initialized: CarRacing-v3
```

---

## Future Roadmap

- **Phase 2 — RL training:** PPO/SAC/A2C runners under `experiments/`, using the
  environment factory, `ExperimentConfig`, callbacks and seeding already in place.
- **Phase 2 — Observation/reward engineering:** promote the identity wrappers to
  real transforms (normalization, frame stacking, resizing, reward shaping).
- **Phase 3 — Analysis & evaluation:** expand `evaluation/` reporting and plots
  over the telemetry and recordings captured here.
- **Phase 4 — Autonomous racing intelligence:** self-improving agents, track
  generalization and benchmarking.

> The Phase 1 / 1.5 framework is intentionally decoupled so each later phase can
> build on it without rewrites.

# RaceMind AI

> An autonomous racing intelligence platform — built incrementally toward
> Reinforcement Learning.

**RaceMind AI** is a long-term machine learning project. Its end goal is an
autonomous racing agent trained with Reinforcement Learning on top of
Gymnasium's `CarRacing-v3` environment.

The repository spans **Phase 1** (the simulator framework), **Phase 1.5**
(research-platform infrastructure) and **Phase 2** (the baseline RL framework).

- **Phase 1** — simulator boundary, manual driving, telemetry, recording, replay.
- **Phase 1.5** — environment factory, Gymnasium wrappers, professional logging,
  extended telemetry, evaluation metrics, experiment configuration, callback
  interfaces, a structured data layout and global seeding.
- **Phase 2** — a pluggable RL framework: a common agent interface, a PPO agent
  (Stable-Baselines3), an algorithm-agnostic trainer, a multi-episode evaluation
  pipeline, checkpoint management, YAML experiment configs, automatic evaluation
  videos and TensorBoard logging. PPO is the only algorithm implemented; SAC/A2C
  plug into the same interfaces later.

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
| Configuration      | `config/`                                 | Simulator, experiment, logging, path and RL configuration.   |
| Callbacks          | `training/callbacks/`                     | Reusable training-callback interfaces.                       |
| Agents             | `agents/`                                 | `BaseAgent` interface, `RandomAgent`, `PPOAgent` (SB3).      |
| Trainer            | `training/trainer.py`                     | Algorithm-agnostic train → evaluate → checkpoint loop.       |
| Evaluation pipeline| `evaluation/evaluator.py` · `benchmark.py`| Multi-episode benchmark with structured metrics.            |
| Checkpoints        | `training/checkpoint_manager.py`          | Latest + best checkpoint save/load.                          |
| Experiment runner  | `experiments/run_experiment.py`           | Main entry point: config → env → agent → train → evaluate.   |

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
├── agents/
│   ├── base_agent.py       # BaseAgent interface (act/predict/learn/save/load)
│   ├── random_agent.py     # RandomAgent baseline
│   └── ppo_agent.py        # PPOAgent (wraps Stable-Baselines3 PPO)
│
├── training/
│   ├── trainer.py          # generic Trainer + build_trainer + TrainingSummary
│   ├── ppo_trainer.py      # build_ppo_trainer (PPO agent + Trainer assembly)
│   ├── training_loop.py    # run_episode rollout (agent-agnostic)
│   ├── checkpoint_manager.py  # latest + best checkpoints
│   └── callbacks/          # base / checkpoint / logging / video
│
├── evaluation/
│   ├── metrics.py          # EpisodeMetrics + compute_episode_metrics
│   ├── episode_summary.py  # EpisodeSummary + summarize_recording
│   ├── benchmark.py        # BenchmarkResult + aggregate_outcomes
│   └── evaluator.py        # Evaluator (multi-episode)
│
├── configs/                # ppo.yaml · simulator.yaml · evaluation.yaml
│
├── experiments/
│   ├── run_experiment.py   # main entry point
│   └── ppo/ sac/ a2c/ imitation/   # per-algorithm output folders
│
├── data/                   # telemetry/ recordings/ videos/ plots/
│                           # evaluation/ models/ checkpoints/ (git-ignored)
├── runs/                   # TensorBoard logs (git-ignored)
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

## Reinforcement Learning (Phase 2)

All experiment parameters live in `configs/*.yaml` (`ppo.yaml`, `simulator.yaml`,
`evaluation.yaml`) — edit those rather than hardcoding values. The experiment
runner is the single entry point.

**Train PPO:**

```bash
python -m experiments.run_experiment --experiment-name ppo_carracing
```

This loads the configs, seeds everything, creates the training and evaluation
environments, builds the PPO agent and trainer, then trains while periodically
evaluating and checkpointing. CLI overrides are available, e.g.:

```bash
python -m experiments.run_experiment --total-timesteps 200000 --seed 123
```

**Sanity-check with the random baseline** (same framework, no learning):

```bash
python -m experiments.run_experiment --experiment-name rnd --algorithm random --total-timesteps 2000
```

**Evaluate** — a multi-episode benchmark runs automatically at the end of
training and writes `data/evaluation/<experiment-name>.json` (average / max / min
reward, reward variance, episode length, success rate). To evaluate a saved
agent without training:

```bash
python -m experiments.run_experiment --experiment-name ppo_carracing --eval-only --resume
```

**Resume training** from the latest checkpoint:

```bash
python -m experiments.run_experiment --experiment-name ppo_carracing --resume
```

Checkpoints are written to `data/checkpoints/<experiment-name>/` as `latest.zip`
and `best.zip` (best by mean evaluation reward).

**Evaluation videos** are recorded automatically (Gymnasium's `RecordVideo`) to
`data/videos/<experiment-name>/` — disable with `--no-video`. They are standard
`.mp4` files; open them in any player. (The Phase 1 `.npz` replay viewer is
separate and used for manually recorded episodes.)

**TensorBoard** logs are written to `runs/<experiment-name>/`:

```bash
tensorboard --logdir runs
```

---

## Future Roadmap

- **Phase 2 (done) — Baseline RL framework:** pluggable agents, PPO via SB3, a
  generic trainer, evaluation pipeline, checkpoints, videos and TensorBoard.
- **Phase 3 — More algorithms:** add SAC and A2C as new `BaseAgent`
  implementations (the trainer and evaluator need no changes).
- **Phase 3 — Observation/reward engineering:** promote the identity wrappers to
  real transforms (normalization, frame stacking, resizing, reward shaping).
- **Phase 4 — Autonomous racing intelligence:** self-improving agents, track
  generalization, hyperparameter sweeps and benchmarking.

> The framework is intentionally decoupled — algorithm code depends only on the
> `BaseAgent` interface, so each later phase builds on it without rewrites.

# Architecture

RaceMind AI is layered so that each stage depends only on the interface below it.
The reward framework and the agent are both **pluggable**, and evaluation is
decoupled from training — which is what makes controlled experiments possible.

## System pipeline

```mermaid
flowchart TD
    A["Simulator<br/>(CarRacing-v3)"] --> B["Environment factory<br/>make_training_env / make_eval_env"]
    B --> C["Wrappers<br/>Observation · Reward · Action · Monitor<br/>(optional: Grayscale, FrameStack)"]
    C --> D["Reward framework<br/>RewardManager + components<br/>(opt-in, training only)"]
    D --> E["PPO Agent<br/>BaseAgent → PPOAgent (SB3)"]
    E --> F["Trainer<br/>chunked train · checkpoint · early-stop"]
    F --> G["Evaluation<br/>30-episode protocol · statistics + 95% CI"]
    G --> H["Benchmarking<br/>multi-agent · paired t-test · Cohen's d"]
    H --> I["Research reports<br/>model cards · learning curves · papers"]
```

## Package map

```mermaid
flowchart LR
    subgraph Core
        sim[simulator/]
        cfg[config/]
        utils[utils/]
    end
    subgraph RL
        agents[agents/]
        training[training/]
    end
    subgraph Science
        evaluation[evaluation/]
        research[research/]
        reward[reward/]
    end
    experiments[experiments/] --> agents
    experiments --> training
    training --> agents
    training --> evaluation
    agents --> sim
    research --> evaluation
    research --> reward
    reward --> sim
    agents --> cfg
    sim --> cfg
    evaluation --> utils
```

## Design principles

- **Single environment creation point** (`simulator.environment_factory`) — every
  env is built here, so wrapping and configuration change in exactly one place.
- **Interface-driven training** — the trainer, evaluator and benchmark suite touch
  only `BaseAgent`; PPO specifics live inside `PPOAgent`. A new algorithm plugs in
  without touching them.
- **Evaluation is decoupled from training** — always the native task reward, 30
  deterministic episodes, fixed seeds. Reward shaping applies to *training only*,
  so every experiment stays comparable to the frozen baseline.
- **Config over code** — simulator, RL, evaluation and reward settings live in
  YAML; changing an experiment does not require editing Python.
- **Reproducibility first** — global seeding, immutable frozen baseline, archived
  specs, model cards, and paired statistical comparison for every run.

See [`training.md`](training.md), [`evaluation.md`](evaluation.md),
[`experiments.md`](experiments.md) and [`reward_framework.md`](reward_framework.md)
for each subsystem.

# RaceMind AI Research Lab — Interactive Showcase

An interactive web app where anyone can **understand, explore, and test** the
RaceMind AI reinforcement-learning project in under three minutes — watch the
trained PPO agent drive, compare it to a random policy, browse the experiments,
and view the training curves.

> Not a landing page — an inference-only "AI research lab" for the project.

```
showcase/
├── backend/      # FastAPI — loads ppo_1m/best.zip once, runs inference, returns MP4 + metrics
├── frontend/     # Next.js (App Router) + Tailwind + Framer Motion + Recharts
├── docker-compose.yml
└── deployment.md
```

## Features

- **Run the AI** — one click runs a deterministic episode on the backend and streams
  back the rendered video, reward, episode length and inference time.
- **Compare agents** — random vs. best PPO, side-by-side.
- **Experiment timeline** — expandable cards (hypothesis · result · verdict · confidence).
- **Training dashboard** — interactive reward / loss / FPS / learning-rate curves.
- **Research findings** — the honest negative results, with evidence and interpretation.

## Quick start (local, no Docker)

```bash
# 1. Backend
cd showcase/backend
pip install -r requirements.txt
cp ../../data/checkpoints/ppo_1m/best.zip ./best.zip     # or set MODEL_PATH
uvicorn main:app --reload --port 8000

# 2. Frontend (new terminal)
cd showcase/frontend
npm install
cp .env.example .env.local                                # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                                               # http://localhost:3000
```

Or the whole stack with Docker:

```bash
cd showcase && cp ../data/checkpoints/ppo_1m/best.zip backend/best.zip && docker compose up --build
```

## Backend API

| Method | Endpoint | Returns |
| --- | --- | --- |
| GET | `/api/health` | `{ status, model_loaded }` |
| GET | `/api/model` | model metadata (name, mean reward, CI, protocol) |
| POST | `/api/run` | run one episode `{ policy, seed }` → video + reward + length + time |
| GET | `/api/baseline` | run the best PPO agent |
| GET | `/api/random` | run a random agent |

The model is loaded **once at startup** and never reloaded. Results are cached by
`(policy, seed)`.

## Tech

**Frontend:** Next.js · TypeScript · TailwindCSS · Framer Motion · Recharts · Lucide.
**Backend:** FastAPI · Stable-Baselines3 · Gymnasium · imageio (MP4).

See [`deployment.md`](deployment.md) for Vercel + Render/Railway/Fly.io deployment.

## Design

Clean, light "AI research lab" aesthetic (white / slate / light blue, a subtle
orange telemetry accent) — inspired by Weights & Biases, Hugging Face, Linear and
Vercel. Subtle motion only; no flashy effects.

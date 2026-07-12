# Deployment — RaceMind AI Research Lab

The showcase is two independently deployable pieces:

- **Frontend** — a static-ish Next.js app → **Vercel**.
- **Backend** — a FastAPI inference server that loads `ppo_1m/best.zip` once →
  **Render / Railway / Fly.io** (CPU only, no GPU).

```
 Browser ──▶ Vercel (Next.js)  ──▶  Render/Railway/Fly (FastAPI)  ──▶  PPO model
```

---

## 1. Backend (Render / Railway / Fly.io)

The backend needs the model file. Either **bake it into the image** (copy
The model file (`ppo_1m/best.zip`, ~26 MB) is **git-ignored**, so it is not in the
repo. Make it available to the backend in one of two ways:

- **Recommended — download on startup.** Attach `best.zip` to the GitHub release and
  set `MODEL_URL` to its asset URL; the backend downloads it on first boot.
- **Or bake it in.** Copy it next to the backend and set `MODEL_PATH=./best.zip`.

### Environment variables

| Variable | Example | Purpose |
| --- | --- | --- |
| `PYTHON_VERSION` | `3.12.7` | **Required for native** — pins Python 3.12 (pygame has no 3.14 wheel) |
| `MODEL_URL` | `https://github.com/shubhamahuja9999/RaceMind-AI/releases/download/v1.0.0/best.zip` | Download the model on startup (the model is git-ignored) |
| `MODEL_PATH` | `/tmp/best.zip` | Where the model is downloaded to |
| `VIDEO_DIR` | `/tmp/videos` | Where MP4s are written/served |
| `SDL_VIDEODRIVER` | `dummy` | Headless rendering (also set in code) |
| `MAX_STEPS` | `600` | Steps per demo episode (snappier demo) |
| `CORS_ORIGINS` | `https://your-app.vercel.app` | Allowed frontend origin(s) |
| `PORT` | (auto) | Injected by the host |

### Render — native Python, free tier (no Docker, no Blueprint)

Create a **Web Service** manually in the dashboard (Blueprints are not needed).
The single most important setting is the **Root Directory** — without it, Render
builds the repo-root `requirements.txt` and stays on Python 3.14, which fails.

- **New → Web Service** → connect the repo
- **Root Directory:** `showcase/backend`   ← critical (uses the backend's files)
- **Language:** Python 3   ·   **Instance Type:** Free
- **Build Command:** `bash build.sh`
  *(runs `showcase/backend/build.sh`: upgrades pip, installs swig, CPU-torch, then requirements)*
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment:** `PYTHON_VERSION=3.12.7`, `SDL_VIDEODRIVER=dummy`, `MODEL_URL=…`,
  `CORS_ORIGINS=…`, `MODEL_PATH=/tmp/best.zip`, `VIDEO_DIR=/tmp/videos`

Why this works: `PYTHON_VERSION=3.12.7` makes pip use pygame's **prebuilt wheel**
(instead of compiling it and needing SDL), installing **`swig` first** lets Box2D
compile, and CPU-only torch avoids a multi-GB CUDA download.

> **Free-tier memory note:** the build succeeds on free, but running an episode
> (PyTorch + CarRacing) is memory-heavy and may occasionally OOM on the 512 MB free
> instance. If it does, upgrade to a paid instance for the live demo — or just rely
> on the frontend's committed `demo.mp4`, which shows the agent driving even when
> the backend is asleep/unavailable. The free service also sleeps after inactivity
> (~30–60 s cold start), which the frontend handles with a "waking up" hint.

### Railway / other hosts

- **Railway:** new service → root `showcase/backend`. Use the same **Build Command**
  and `PYTHON_VERSION=3.12.7` as above (Nixpacks native build).
- Any host with a native Python buildpack works the same way: pin Python 3.12,
  install `swig` before the requirements, and set the environment variables above.

> First request runs an episode on CPU (~10–20 s) and caches the MP4; subsequent
> requests for the same `(policy, seed)` are instant. On free tiers the service may
> sleep — the frontend shows a "waking up" hint and falls back to a cached demo.

---

## 2. Frontend (Vercel)

1. Import the repo into Vercel, set **Root Directory** to `showcase/frontend`.
2. Framework preset: **Next.js** (auto-detected).
3. Environment variable:

   | Variable | Value |
   | --- | --- |
   | `NEXT_PUBLIC_API_URL` | `https://racemind-api.onrender.com` |

4. Deploy. Update the backend's `CORS_ORIGINS` to the Vercel URL.

---

## 3. Local development

```bash
# Backend
cd showcase/backend
pip install -r requirements.txt
cp ../../data/checkpoints/ppo_1m/best.zip ./best.zip   # or set MODEL_PATH / MODEL_URL
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd showcase/frontend
npm install
cp .env.example .env.local        # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                        # → http://localhost:3000
```

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

### Render — native Python (no Docker) ✅ recommended

The included `render.yaml` does this automatically: **New → Blueprint → select the
repo** → it creates the service with the right Python version and build command.
Then set the two secret env vars (`MODEL_URL`, `CORS_ORIGINS`) in the dashboard.

To configure a service **by hand** instead, use these settings — they are the fix
for the pygame/box2d/Python-3.14 build failures:

- **Runtime / Language:** Python
- **Root Directory:** `showcase/backend`
- **Build Command:**
  ```
  pip install --upgrade pip && pip install swig && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
  ```
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment:** `PYTHON_VERSION=3.12.7`, `SDL_VIDEODRIVER=dummy`, `MODEL_URL=…`,
  `CORS_ORIGINS=…`, `MODEL_PATH=/tmp/best.zip`, `VIDEO_DIR=/tmp/videos`
- **Instance:** 512 MB+ (PyTorch + CarRacing is memory-heavy; the free tier may OOM)

Why this works: `PYTHON_VERSION=3.12.7` makes pip use pygame's **prebuilt wheel**
(instead of compiling it and needing SDL), and installing **`swig` first** lets
Box2D compile. CPU-only torch avoids a multi-GB CUDA download.

### Docker (alternative)

A `showcase/backend/Dockerfile` is also provided if you prefer a container: set the
service **Runtime = Docker**. It installs the same deps with system libraries baked
in. Not required — the native path above is the recommended one here.

### Railway / Fly.io

- **Railway:** new service → root `showcase/backend`. Use the same Build Command and
  `PYTHON_VERSION=3.12.7` as above (Nixpacks/native), or the `Dockerfile`.
- **Fly.io:** `fly launch` in `showcase/backend`; set secrets with
  `fly secrets set CORS_ORIGINS=... MODEL_URL=...`.

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

## 3. Local (Docker Compose)

```bash
cd showcase
cp ../data/checkpoints/ppo_1m/best.zip backend/best.zip
docker compose up --build
# frontend → http://localhost:3000   backend → http://localhost:8000
```

## 4. Local (without Docker)

```bash
# Backend
cd showcase/backend
pip install -r requirements.txt
cp ../../data/checkpoints/ppo_1m/best.zip ./best.zip   # or set MODEL_PATH
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd showcase/frontend
npm install
cp .env.example .env.local        # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                        # → http://localhost:3000
```

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

> ⚠️ **Use the Docker runtime, not the native Python buildpack.** The native
> buildpack defaults to a too-new Python and must compile Box2D from source — that
> is exactly what caused the `box2d`/Python-3.14 build failure. The provided
> `Dockerfile` pins Python 3.12 and installs `swig` + `ffmpeg`, so it just works.
> (`requirements.txt` no longer pins a bare `box2d`; `gymnasium[box2d]` supplies the
> physics backend. A `runtime.txt` pins Python 3.12 if you must use the native path.)

### Environment variables

| Variable | Example | Purpose |
| --- | --- | --- |
| `MODEL_URL` | `https://github.com/shubhamahuja9999/RaceMind-AI/releases/download/v1.0.0/best.zip` | Download the model on startup (if `MODEL_PATH` is missing) |
| `MODEL_PATH` | `/app/best.zip` | Where the model lives / is downloaded to |
| `VIDEO_DIR` | `/app/videos` | Where MP4s are written/served |
| `MAX_STEPS` | `600` | Steps per demo episode (snappier demo) |
| `CORS_ORIGINS` | `https://your-app.vercel.app` | Allowed frontend origin(s) |
| `PORT` | `8000` | Provided by the host on most platforms |

### Render (Docker)

1. New → **Web Service** → connect the repo, **Root Directory** = `showcase/backend`.
2. **Runtime / Language: Docker** (Render auto-detects the `Dockerfile`). Do **not**
   pick the Python buildpack.
3. Add the environment variables above (at minimum `MODEL_URL` and `CORS_ORIGINS`).
4. Instance type: use at least the **512 MB+** tier — PyTorch + CarRacing inference
   is memory-heavy; the free tier may OOM.
5. Deploy. Note the URL, e.g. `https://racemind-api.onrender.com`.

### Railway / Fly.io

- **Railway:** new service from repo → root `showcase/backend` → **Dockerfile** → set env vars.
- **Fly.io:** `fly launch` in `showcase/backend` (detects the Dockerfile), set
  secrets with `fly secrets set CORS_ORIGINS=... MODEL_URL=...`, then `fly deploy`.

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

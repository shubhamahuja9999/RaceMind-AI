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
`best.zip` into `showcase/backend/`) or mount it and set `MODEL_PATH`.

```bash
cp ../data/checkpoints/ppo_1m/best.zip showcase/backend/best.zip
```

### Environment variables

| Variable | Example | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `/app/best.zip` | Path to the trained model |
| `VIDEO_DIR` | `/app/videos` | Where MP4s are written/served |
| `MAX_STEPS` | `600` | Steps per demo episode (snappier demo) |
| `CORS_ORIGINS` | `https://your-app.vercel.app` | Allowed frontend origin(s) |
| `PORT` | `8000` | Provided by the host on most platforms |

### Render

1. New → **Web Service** → point at the repo, root `showcase/backend`.
2. Runtime: **Docker** (uses the provided `Dockerfile`).
3. Set the environment variables above.
4. Deploy. Note the URL, e.g. `https://racemind-api.onrender.com`.

### Railway / Fly.io

- **Railway:** new service from repo → root `showcase/backend` → Dockerfile → set env vars.
- **Fly.io:** `fly launch` in `showcase/backend` (detects the Dockerfile), set
  secrets with `fly secrets set CORS_ORIGINS=...`, then `fly deploy`.

> First request runs an episode on CPU (~10–20 s) and caches the MP4; subsequent
> requests for the same `(policy, seed)` are instant.

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

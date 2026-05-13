# PaperMind — Deployment Guide (For First-Time Deployers)

This document explains how to take PaperMind out of your laptop and put it on the internet so other people can use it.

It is written assuming you have **never deployed anything before**. We cover what each thing is, why it exists, and what you actually have to do. Read it top to bottom once before doing anything.

---

## Part 1 — What "deployment" actually means

When you run `uvicorn api.main:app` on your laptop, three things are happening:

1. **A server process is running.** It listens on port 8000 and waits for HTTP requests.
2. **Your code has access to files** (`data/chroma_db/`, `data/papers/`, `.env`).
3. **You can reach it at `localhost`** — because "localhost" is your own machine.

"Deploying" means recreating those three things on a computer that:

- Is **always on** (your laptop closes; servers don't).
- Has a **public address** people can type into a browser (`papermind.com` instead of `localhost:8000`).
- Has the **same files and environment** your code expects (Python version, dependencies, the `.env` keys, the data folder).

That computer is called a **server**, and you rent one from a **cloud provider** (AWS, Google Cloud, Render, Railway, Fly.io, Hugging Face, etc.). You don't buy a physical box — you rent a slice of someone else's data centre, configured to look like a Linux machine.

### The two pieces of PaperMind

PaperMind has two halves that get deployed differently because they have very different needs:

| Piece | What it is | What it needs |
|---|---|---|
| **Frontend** (`frontend/`) | React app, compiled by Vite into static HTML/JS/CSS | A web server that just hands files to browsers. Cheap, fast, can be free. |
| **Backend** (`api/`, `ingestion/`) | Python FastAPI app, heavy ML dependencies (torch, sentence-transformers, ChromaDB) | Always-on Python process, ~2 GB RAM minimum, persistent disk for `data/`. |

You will deploy these to **two different places**, then wire them together. This is normal — almost every modern web app works this way.

---

## Part 2 — Key vocabulary

Before we touch anything, here are the words you will see everywhere:

- **Container / Docker image** — a "frozen" copy of your app + Python + dependencies + OS, in one file. You build it once, run it anywhere. Removes "works on my machine" problems. Almost every host wants either a Docker image or something they can turn into one.
- **Dockerfile** — a recipe for building that frozen copy. We'll write one.
- **Environment variables** — values like `GEMINI_API_KEY` that change between your laptop and production. You don't bake them into the image — you set them in the host's dashboard. That way the same image works in dev/staging/prod.
- **Static hosting** — for the frontend. Just files on a CDN. Free tier on Vercel, Netlify, Cloudflare Pages, GitHub Pages.
- **PaaS (Platform as a Service)** — hosts that take your repo or Docker image and run it for you: Render, Railway, Fly.io, Hugging Face Spaces. You don't manage Linux. You pay per month.
- **IaaS (Infrastructure as a Service)** — raw Linux VMs: AWS EC2, GCP Compute, DigitalOcean Droplets. Cheaper at scale, but **you** install Python, manage updates, deal with crashes at 3am. Skip this for your first deploy.
- **CORS** — Cross-Origin Resource Sharing. When your frontend (at `papermind.vercel.app`) calls your backend (at `papermind-api.render.com`), browsers block it by default for security. The backend has to send a header saying "I trust this origin". You already have `CORSMiddleware` set up — but it's set to `"*"` (allow everyone) which is fine for dev, **not** for prod.
- **Persistent disk / volume** — extra storage attached to a server that survives restarts. Without it, every redeploy wipes `data/papers.json`, all uploaded PDFs, and the ChromaDB index. PaperMind needs persistent disk.
- **Cold start** — when a server has been idle and the first request has to wake it up. For us, "wake up" means loading sentence-transformers (~90MB) and ChromaDB into RAM, which takes 10–30 seconds. Free tiers cold-start aggressively.

---

## Part 3 — PaperMind's deployment constraints (read this before picking a host)

Not every cloud host can run PaperMind. Here is what PaperMind specifically needs and why it rules some options out:

### 3.1 — Heavy ML dependencies

`requirements.txt` pulls in `sentence-transformers`, which pulls in `torch`. PyTorch CPU wheels are ~200MB. Plus `chromadb`, `tiktoken`, `pdfplumber`. The installed environment is ~2 GB.

**Consequence:** Free tiers with 512MB RAM (Heroku free, some Render free instances) **will not work**. You need at least **2 GB RAM**, ideally 4 GB so embeddings load with headroom.

### 3.2 — Persistent state on disk

Look at `api/storage.py:7-11` — it writes `data/papers.json` and PDFs to local disk. `ingestion/embedder.py:15` writes ChromaDB to `data/chroma_db/`.

**Consequence:** You cannot use a host that gives you only ephemeral filesystems (Vercel functions, Cloudflare Workers, Lambda). You need either:
- A host with attached persistent volumes (Render disk, Fly.io volumes, Railway volumes), **or**
- To refactor storage to use an external store (S3 for PDFs, Postgres for the registry, hosted vector DB like Pinecone for embeddings).

For your first deploy: **use a host with volumes.** Refactoring is a v2 problem.

### 3.3 — Long-running requests

`/upload` kicks off ingestion in a background task that can take 30–90 seconds. `/query` can take 20+ seconds for multi-hop or comparison queries. The endpoint timeout is set to 60s/120s in `api/main.py`.

**Consequence:** Hosts with a hard 30-second request timeout (some serverless platforms) will cut you off mid-answer. PaaS hosts running normal Python processes are fine.

### 3.4 — Models downloaded at runtime

First time `SentenceTransformer("all-MiniLM-L6-v2")` runs, it downloads the model from Hugging Face Hub (~90 MB) into a cache directory. If your container is read-only or the cache path isn't persisted, every cold start re-downloads.

**Consequence:** Bake the model into the Docker image (download during build), or use a host-provided persistent cache.

### 3.5 — Secrets in `.env`

`.env` currently contains live API keys for Gemini, Groq, OpenAI, Cerebras, Mistral. It's gitignored (good), but it lives on your disk. **Never commit it. Never paste it into a chat. Never bake it into a Docker image.** The host's dashboard has a "Secrets" or "Environment Variables" section — that's where it goes.

> **Action item before you deploy:** because you just shared the contents of `.env` with me in this chat, rotate every one of those keys (regenerate them in each provider's console) before going live. Treat any key that has been pasted anywhere outside your machine as burned.

---

## Part 4 — Architecture: where each piece will live

The end state looks like this:

```
                            ┌──────────────────────────┐
                            │   user's browser         │
                            │   types papermind.app    │
                            └────────────┬─────────────┘
                                         │
                       ┌─────────────────┴───────────────────┐
                       │                                     │
              ┌────────▼─────────┐                ┌──────────▼─────────┐
              │  FRONTEND HOST   │                │   BACKEND HOST     │
              │  (Vercel/        │   API calls    │   (Render/Fly/HF)  │
              │   Netlify/CF)    │ ─────────────► │                    │
              │                  │                │  FastAPI + Python  │
              │  Static files    │                │  + ML models       │
              │  built by Vite   │                │                    │
              │                  │                │  + persistent disk │
              │  FREE tier OK    │                │  ($5-25/month)     │
              └──────────────────┘                └──────────┬─────────┘
                                                             │
                                                  ┌──────────▼──────────┐
                                                  │ External LLM APIs   │
                                                  │ Gemini, Groq, etc.  │
                                                  │ (you pay per token) │
                                                  └─────────────────────┘
```

Two URLs, two deployments, talking over HTTPS.

---

## Part 5 — Recommended platforms (for someone deploying for the first time)

I'll recommend a specific stack, then list alternatives. Pick whichever you're comfortable with after reading.

### Recommended: **Render (backend) + Vercel (frontend)**

| Piece | Host | Why |
|---|---|---|
| Backend | **Render** ($7–25/mo) | Persistent disks, supports Dockerfile or native Python, simple dashboard, decent free tier for testing, no credit card surprises. Cold-starts on free tier; pay $7 to remove that. |
| Frontend | **Vercel** (free) | Designed for Vite/React. Deploys on every git push. HTTPS by default. Free forever for personal projects. |

### Alternatives worth considering

- **Hugging Face Spaces** (free, ML-friendly): great if you want a quick demo, comes with persistent storage and beefy machines. Tradeoff: domain is `huggingface.co/spaces/you/papermind`, not your own URL. Best for portfolio/demo use.
- **Fly.io** ($5+/mo): more powerful than Render, deploys with `flyctl deploy`, volumes are first-class. Steeper learning curve.
- **Railway** ($5+/mo): similar to Render, slightly nicer UX for Postgres etc. Volumes are newer/less battle-tested.
- **AWS / GCP / Azure**: do **not** start here. The learning curve is brutal and the bill is unpredictable. Move here in 6 months if PaperMind scales.

For the rest of this guide I assume **Render + Vercel**. The general flow translates to any other PaaS — only the dashboard names change.

---

## Part 6 — Pre-deployment checklist (do this before touching any host)

These are all things to fix in your code/config **first**, locally, so deployment doesn't fail halfway.

### 6.1 — Rotate exposed API keys

Go to each provider's dashboard and regenerate:
- Gemini (Google AI Studio) — both `GEMINI_API_KEY` and `GEMINI_API_KEY_2` and `VLM_GEMINI_API_KEY`
- Groq
- OpenAI
- Cerebras
- Mistral

Update your local `.env`. Confirm the app still works (`uvicorn api.main:app --reload`, upload a PDF, ask a question).

### 6.2 — Lock CORS to your frontend domain

Right now `api/main.py:32-38` has `allow_origins=["*"]`. That means any website can call your API. Change it to a config-driven list before going live:

```python
import os

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then in Render you'll set `ALLOWED_ORIGINS=https://papermind.vercel.app,https://yourdomain.com`.

### 6.3 — Pin Python version

Create a `runtime.txt` in the project root:

```
python-3.11
```

(Or whatever you're developing on — check with `python --version`.) Render reads this. Without it, you'll get whatever default the host picks, which might break a dependency.

### 6.4 — Add a `.dockerignore`

If you Dockerize (recommended), you don't want to copy your 500MB `venv/`, `__pycache__/`, `.git/`, test outputs, or `data/` into the image. Create `.dockerignore` next to the Dockerfile:

```
venv/
__pycache__/
*.pyc
.pytest_cache/
.git/
.env
data/
node_modules/
frontend/node_modules/
frontend/dist/
*.txt
!requirements.txt
output/
scratch/
test_*.py
*.md
```

### 6.5 — Make the frontend API URL configurable

Currently `frontend/src/api.js:1` uses `const BASE = '/api'` which relies on Vite's dev proxy (`vite.config.js`). In production the frontend is on a different domain than the backend, so this won't work.

Change to:

```js
const BASE = import.meta.env.VITE_API_URL || '/api'
```

Then in dev you keep using the Vite proxy (fast HMR, no CORS). In prod, you set `VITE_API_URL=https://papermind-api.onrender.com` in Vercel's env vars at build time.

### 6.6 — Write a `Dockerfile` for the backend

Create `Dockerfile` in project root:

```dockerfile
# Use a slim Python image — smaller, faster to pull.
FROM python:3.11-slim

# System deps that pdfplumber/torch sometimes need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first and install them.
# This is a Docker layer-caching trick: requirements rarely change,
# so this layer gets cached and rebuilds are fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models so they're baked into
# the image. Otherwise the first request after every deploy waits
# 30s for ~200MB of model weights to download from Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Now copy the rest of the app.
COPY api ./api
COPY ingestion ./ingestion

# Render injects $PORT. Bind to it.
ENV PORT=8000
EXPOSE 8000

# Use uvicorn directly — no auto-reload in prod.
# --workers 1 because chromadb and the in-memory model are not multi-worker safe
# and you have one machine anyway. Scale by getting a bigger box, not more workers.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
```

Test it locally before pushing:

```powershell
docker build -t papermind .
docker run --rm -p 8000:8000 --env-file .env papermind
# In another terminal:
curl http://localhost:8000/health
```

If `/health` returns `{"status":"ok"}`, you're good.

### 6.7 — Verify the frontend builds

```powershell
cd frontend
npm install
npm run build
```

If `frontend/dist/` is produced without errors, Vercel will be happy.

### 6.8 — Commit everything, push to GitHub

Both Render and Vercel deploy from a GitHub repo. If your project isn't on GitHub yet:

```powershell
# Create a new private repo on github.com first, then:
git remote add origin https://github.com/Yuvrajjys2512/papermind.git
git add Dockerfile .dockerignore runtime.txt DEPLOYMENT.md
git commit -m "deployment prep: dockerfile, ignore rules, runtime"
git push -u origin main
```

**Double-check** before pushing: `git status` should not show `.env` as tracked.

---

## Part 7 — Step-by-step: deploying the backend to Render

### 7.1 — Create a Render account

Go to https://render.com, sign in with GitHub. Authorize Render to access the PaperMind repo.

### 7.2 — Create a new Web Service

- Dashboard → **New +** → **Web Service**.
- Connect your GitHub repo.
- Render auto-detects the Dockerfile. Confirm it picked it.

Configure:

| Field | Value | Why |
|---|---|---|
| Name | `papermind-api` | Becomes your subdomain: `papermind-api.onrender.com`. |
| Region | Pick closest to your users | Latency. Singapore for India, Oregon for US west, Frankfurt for EU. |
| Branch | `main` | Render redeploys on every push to this branch. |
| Runtime | Docker | We have a Dockerfile. |
| Instance Type | **Starter ($7/mo) or Standard ($25/mo)** | Free tier has 512MB RAM and cold-starts after 15 min idle. PaperMind needs **at least 2 GB**, so Standard. You can start on Starter (512MB) only to test the deploy succeeds — the app will OOM-crash the first time you actually use it. |
| Auto-Deploy | Yes | Push to main → auto-redeploy. |

### 7.3 — Add a persistent disk

Render's free disks aren't actually persistent across redeploys. On paid plans:

- In the service settings → **Disks** → **Add Disk**.
- Mount path: `/app/data`
- Size: 5 GB to start (you can grow it later, not shrink).

This is critical. Without it, every redeploy wipes uploaded PDFs and ChromaDB. After the first deploy, verify the data folder shows up at `/app/data` inside the container.

### 7.4 — Set environment variables (your secrets)

In the service settings → **Environment** → **Add Environment Variable**, add **each** of:

```
GEMINI_API_KEY=<new rotated key>
GEMINI_API_KEY_2=<new rotated key>
VLM_GEMINI_API_KEY=<new rotated key>
GROQ_API_KEY=<new rotated key>
OPENAI_API_KEY=<new rotated key>
CEREBRAS_API_KEY=<new rotated key>
MISTRAL_API_KEY=<new rotated key>
ALLOWED_ORIGINS=https://papermind.vercel.app
```

(You won't know the Vercel URL yet on the first pass — leave it as `*` for now, fix it after step 8.)

### 7.5 — Deploy

Click **Create Web Service**. Render will:

1. Clone your repo.
2. Build the Docker image (~5–10 min the first time, mostly downloading torch).
3. Start the container.
4. Run health checks on `/health`.

Watch the build logs. If anything fails, the most common causes are:

- **OOM during build**: torch install gets killed. Upgrade instance type, or use `--no-cache-dir` (already in Dockerfile).
- **Model download fails**: Hugging Face rate-limited. Retry the deploy.
- **Port mismatch**: ensure your Dockerfile uses `${PORT}`, not hardcoded `8000`. Render injects its own `PORT`.

When it's live, you'll see "Your service is live at `https://papermind-api.onrender.com`". Open `https://papermind-api.onrender.com/health` in a browser. You should see `{"status":"ok"}`.

### 7.6 — Smoke-test from your laptop

```powershell
curl https://papermind-api.onrender.com/health
curl https://papermind-api.onrender.com/papers
```

The second one should return `[]` (no papers yet). If both work, the backend is up.

---

## Part 8 — Step-by-step: deploying the frontend to Vercel

### 8.1 — Create a Vercel account

https://vercel.com → sign in with GitHub. Authorize the repo.

### 8.2 — Import the project

- Dashboard → **Add New** → **Project**.
- Select the PaperMind repo.
- **Root Directory**: `frontend` (this is the key step — without it, Vercel tries to build the whole repo).
- Framework: Vercel will auto-detect Vite. Confirm.
- Build Command: `npm run build` (default).
- Output Directory: `dist` (default).

### 8.3 — Set the API URL env var

Before the first deploy, click **Environment Variables**:

```
VITE_API_URL=https://papermind-api.onrender.com
```

This is read at **build time** by Vite and baked into the JS bundle. If you change it later, you must trigger a redeploy.

### 8.4 — Deploy

Click **Deploy**. Vercel builds the static bundle (~30s) and gives you a URL like `https://papermind-abc123.vercel.app`.

Open it. Try uploading a PDF. If it works end-to-end, you're done. If you get a CORS error in the browser console:

### 8.5 — Fix CORS

Go back to Render → your service → Environment → update:

```
ALLOWED_ORIGINS=https://papermind-abc123.vercel.app
```

Render redeploys automatically (~2 min). Refresh the frontend. CORS error gone.

### 8.6 — Add a custom domain (optional)

In Vercel → Project → **Domains** → Add `papermind.yourdomain.com`. Vercel gives you DNS records to add at your registrar (Namecheap, Cloudflare, GoDaddy). After DNS propagates (~5 min to 24h), HTTPS is automatic. Don't forget to also add your custom domain to `ALLOWED_ORIGINS` on Render.

---

## Part 9 — After deployment: things to actually do

### 9.1 — Watch the logs for a day

In Render → your service → **Logs**. The first time real users hit it, you'll see things you didn't see in dev:

- LLM rate limits (Groq especially is aggressive on free tier).
- ChromaDB locking errors if two uploads happen at once.
- OOM kills if the model + a long PDF exceed your RAM tier.

The logger you wrote at `api/logger.py` will tag each query with a `request_id`. When something fails, search the logs by that ID.

### 9.2 — Set spend caps on the LLM providers

Each provider's dashboard has a usage page. Set hard limits:

- OpenAI: Settings → Limits → **Hard limit** (e.g., $10/month).
- Gemini: usually free tier covers a lot, but watch quota.
- Groq: free tier has rate limits, not cost — fine.

This protects you against a runaway loop or someone abusing your public endpoint.

### 9.3 — Add basic rate limiting

Right now anyone can spam `/query`. Cheapest fix: add `slowapi` to the backend, limit to e.g. 30 requests/minute per IP. Not urgent for a portfolio project, urgent before you put it on Hacker News.

### 9.4 — Plan for ChromaDB growth

5 GB of disk holds maybe 200–500 papers depending on length. Watch the **Disk** tab in Render. When you cross 70%, expand it (Render lets you grow disks without downtime; shrinking requires a migration).

### 9.5 — Back up `data/`

Render disks are durable but not backed up. Once a week, SSH into your Render shell (Settings → Shell) and:

```bash
tar czf /tmp/papermind-data-$(date +%F).tar.gz data/
# Then download via:
# scp, or upload to S3/Drive from inside the shell
```

For a v1, "back up before any risky deploy" is fine. Automate later.

---

## Part 10 — Common failure modes and how to debug them

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` works, `/upload` returns 500 | Model didn't download / no write perm on `data/` | Check the disk is mounted at `/app/data`. Check logs for `PermissionError`. |
| CORS error in browser only | `ALLOWED_ORIGINS` doesn't include your Vercel URL | Update env var, redeploy. |
| Frontend shows network error, backend logs are silent | `VITE_API_URL` is wrong or unset | Check Vercel env vars. Remember they're build-time — you must redeploy after changing. |
| App is slow for first request after idle | Render free-tier cold start | Upgrade to Starter ($7) or use UptimeRobot to ping `/health` every 10 min (cheap hack, free tier–compatible). |
| Build fails with "killed" mid-pip-install | OOM during torch install | Upgrade instance tier during build, or use a pre-built torch wheel index. |
| ChromaDB throws "readonly database" | Two processes (or two workers) touching ChromaDB at once | Keep `--workers 1`. ChromaDB is not multi-process safe. |
| First query after redeploy is 30s slow | Models re-downloading because they weren't baked in | Confirm the `RUN python -c ...` step in your Dockerfile ran during build. |
| 502 errors intermittently | The container is restarting (likely OOM) | Check the Events tab. Upgrade RAM. |

---

## Part 11 — What's next (v2 thoughts, not urgent)

Once the basic deploy is up, these are the natural improvements:

1. **External Postgres for the paper registry** instead of `data/papers.json`. Render has managed Postgres ($7/mo). This makes scaling to multiple backend instances possible.
2. **S3 (or Cloudflare R2 — cheaper) for PDFs** instead of local disk. Decouples storage from compute.
3. **Hosted vector DB** (Pinecone, Weaviate Cloud, Qdrant Cloud) instead of local ChromaDB. Same reason.
4. **CI/CD via GitHub Actions** — run your test suite on every PR before Render auto-deploys.
5. **Auth** — currently anyone with the URL can upload. Adding Clerk or Auth0 is half a day's work.
6. **Observability** — wire `api/logger.py` to ship logs to a service like Axiom or Better Stack. Local logs disappear on redeploy.

None of those are needed for a working v1. Get the basic deploy live first, then iterate.

---

## TL;DR — the minimum path

1. Rotate every API key you pasted into chat.
2. Add `Dockerfile`, `.dockerignore`, `runtime.txt`, make `VITE_API_URL` configurable, make `ALLOWED_ORIGINS` configurable.
3. Push to GitHub.
4. Render → New Web Service → connect repo → Docker → 2GB instance → add 5GB disk at `/app/data` → set all the env vars.
5. Vercel → New Project → root dir `frontend` → set `VITE_API_URL` → deploy.
6. Update `ALLOWED_ORIGINS` on Render with the Vercel URL.
7. Test end to end. Watch logs. Set spend caps.

Total time on a fresh weekend: **3–5 hours**, most of it waiting for builds.

# PaperMind — Deployment Guide

How to take PaperMind from your laptop to a real product at your own domain that can charge money — **on a free tier** to start.

This reflects the architecture *after* Launch Checklist §1–§4: Clerk auth, Neon Postgres, Cloudflare R2, Stripe billing, CORS lockdown, rate limiting, and a Docker image. If you read an older version of this file that talked about `data/papers.json` and "no auth," that's gone — this is current.

---

## Part 1 — The shape of it

PaperMind deploys as **two halves to two places**, wired together over HTTPS:

| Piece | What it is | Where it goes | Cost |
|---|---|---|---|
| **Frontend** (`frontend/`) | React/Vite static files | **Vercel** | Free + your domain (~$10/yr) |
| **Backend** (`api/`, `ingestion/`, `discovery/`) | FastAPI + PyTorch + 2 ML models | **Hugging Face Spaces** (Docker) | Free |

**Users only ever see your domain** (e.g. `papermind.app`). It serves the frontend; the frontend calls the backend's `*.hf.space` URL behind the scenes. That backend URL being a Hugging Face address is invisible to users — which is why a free backend host is perfectly compatible with a "real product on my own domain."

### Why this stack (and not Render)

The backend loads PyTorch + two models into RAM, so it needs **~2 GB RAM**. Render's free tier is 512 MB — it would crash on boot; Render only works here as a paid plan (~$7–25/mo). **Hugging Face Spaces free tier is 2 vCPU / 16 GB RAM** — the only free option that can actually run this backend.

### The one free-tier tradeoff

HF Spaces free has two catches, both handled:
- **It sleeps after 48 h idle** → the first request after a nap takes ~20–30 s to wake. (Mitigation: a free UptimeRobot ping every 10 min keeps it warm — see Part 7.)
- **Its disk is ephemeral** (wiped on rebuild/restart). Our ChromaDB index lives there — but **§4 added regenerate-on-startup**: on boot the backend rebuilds any missing index from the PDFs in R2 (the registry is in Neon, the PDFs in R2, so the index is fully regenerable). No paid disk needed. The cost is some free-tier LLM calls + a slower first request after a rebuild.

**Upgrade path when you have paying users:** add HF's persistent storage ($5/mo). The regeneration step then becomes a harmless no-op — no code change.

---

## Part 2 — What the backend depends on (already built)

These are live services from §1–§3. You already have accounts/keys for them in your local `.env`; deployment just means putting the same values into the host's secret store.

| Concern | Service | Env vars |
|---|---|---|
| Auth | Clerk | `CLERK_ISSUER` (backend), `VITE_CLERK_PUBLISHABLE_KEY` (frontend) |
| Registry DB | Neon Postgres | `DATABASE_URL` |
| PDF storage | Cloudflare R2 | `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` |
| Billing | Stripe | `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET` |
| LLM providers | Groq/Gemini/Mistral/Cerebras/OpenAI | `GROQ_API_KEY`, `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `VLM_GEMINI_API_KEY`, `MISTRAL_API_KEY`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY` |
| Frontend origin (CORS + Stripe redirects) | — | `ALLOWED_ORIGINS`, `PAPERMIND_FRONTEND_URL` |
| Frontend → backend URL | — | `VITE_API_URL` (frontend, build-time) |

Vector search (ChromaDB) stays local on the backend and regenerates from R2 — no service, no env var.

---

## Part 3 — 🔑 Rotate your keys first (do this before anything else)

Every key in `.env` (all the LLM keys, Clerk, Neon, R2, Stripe) was pasted into a chat at some point during development. **Treat them as burned.** Before going live, regenerate each one in its provider's dashboard and update your local `.env`:

- Gemini (×3: `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `VLM_GEMINI_API_KEY`), Groq, OpenAI, Cerebras, Mistral
- Clerk: rotate the secret/instance if exposed
- Neon: rotate the database password (new `DATABASE_URL`)
- R2: roll the access key pair
- Stripe: roll the secret key

Confirm the app still works locally (`uvicorn api.main:app --reload`, upload a PDF, ask a question) after rotating.

---

## Part 4 — Pre-deploy checklist (code side — already done in §4)

These were the code prerequisites; §4 implemented them, so they're listed here just so you know they're handled:

- ✅ **CORS** is env-driven (`ALLOWED_ORIGINS`), no longer `"*"`.
- ✅ **Frontend API URL** is configurable (`VITE_API_URL`, falls back to the Vite proxy in dev).
- ✅ **Dockerfile** + `.dockerignore` exist, copy `api`/`ingestion`/`discovery`, and **pre-bake the two models** (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) so there's no cold-start download.
- ✅ **Upload validation** (size cap + PDF magic-byte check) and **per-IP rate limiting** (120/min) are live.
- ✅ **ChromaDB regenerates** from R2 on startup.

Local Docker smoke test (optional but recommended):
```bash
docker build -t papermind .
docker run --rm -p 7860:7860 --env-file .env papermind
curl http://localhost:7860/health   # → {"status":"ok"}
```

---

## Part 5 — Deploy the backend to Hugging Face Spaces

1. **Create the Space:** huggingface.co → New Space → **SDK: Docker** (blank template). Name it e.g. `papermind-api`. Hardware: **CPU basic (free)**. Its URL will be `https://<you>-papermind-api.hf.space`.
2. **Push the code:** add the Space as a git remote and push (the Space builds from the root `Dockerfile`; `.dockerignore` keeps `frontend/`, `data/`, secrets, etc. out):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/papermind-api
   git push space main
   ```
   The Dockerfile already `EXPOSE`s 7860 (HF's default port), so no extra port config is needed.
3. **Set secrets:** Space → **Settings → Variables and secrets** → add **all** the backend env vars from the Part 2 table (LLM keys, `CLERK_ISSUER`, `DATABASE_URL`, `R2_*`, `STRIPE_*`). Set `ALLOWED_ORIGINS` and `PAPERMIND_FRONTEND_URL` after you know the frontend domain (Part 6) — leave them for now or set to your eventual domain.
4. **Watch the build** (~5–10 min the first time — torch + model bake). When live, open `https://<you>-papermind-api.hf.space/health` → `{"status":"ok"}`.

> The first real request after a rebuild or a 48 h nap will be slow (wake + Chroma regeneration). That's expected on the free tier.

---

## Part 6 — Deploy the frontend to Vercel + your domain

1. **Import:** vercel.com → Add New → Project → pick the repo. **Root Directory: `frontend`** (critical — otherwise it builds the whole repo). Framework auto-detects as Vite.
2. **Build-time env vars** (Project → Settings → Environment Variables):
   ```
   VITE_API_URL=https://<you>-papermind-api.hf.space
   VITE_CLERK_PUBLISHABLE_KEY=<your Clerk publishable key>
   ```
   These are baked in at build time — changing them later requires a redeploy.
3. **Deploy.** You get `https://papermind-xxxx.vercel.app`. Test it works end-to-end (sign in, upload, query).
4. **Custom domain:** Project → Domains → add `papermind.app` (buy it at any registrar — Namecheap/Cloudflare/Porkbun, ~$10/yr). Add the DNS records Vercel shows you. HTTPS is automatic once DNS propagates.

---

## Part 7 — Wire the three cross-service settings

After both halves are live and you know your domain, finish the connections:

1. **CORS + Stripe redirects (HF secrets):** set on the Space and let it restart:
   ```
   ALLOWED_ORIGINS=https://papermind.app,https://papermind-xxxx.vercel.app
   PAPERMIND_FRONTEND_URL=https://papermind.app
   ```
2. **Stripe webhook (production):** in the Stripe **live/test dashboard** → Developers → Webhooks → add endpoint `https://<you>-papermind-api.hf.space/billing/webhook`, subscribe to `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`. Copy that endpoint's **signing secret** into the Space's `STRIPE_WEBHOOK_SECRET`. (The local `stripe listen` secret is dev-only.)
3. **Clerk allowed origins:** in the Clerk dashboard, add your production domain so tokens are issued for it.
4. **Keep-warm (optional, free):** UptimeRobot → monitor `https://<you>-papermind-api.hf.space/health` every 10 min to blunt the 48 h sleep.

---

## Part 8 — After launch

- **Set spend caps** on each LLM provider (OpenAI hard limit, watch Gemini/Groq quotas). Protects against abuse loops. The §2 per-user quotas (3 papers / 20 queries/mo free) already bound this.
- **Watch the HF Space logs** for the first day — LLM rate limits, OOM, Chroma locking under concurrent uploads.
- **Upgrade trigger:** when papers/traffic grow, buy HF persistent storage ($5/mo) and the startup regeneration becomes a no-op; or move the backend to a paid always-on host (Render/Fly) to kill the cold start.

---

## Part 9 — Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| CORS error in browser only | `ALLOWED_ORIGINS` missing your domain | Add it to the Space secrets; restart. |
| Frontend network error, backend silent | `VITE_API_URL` wrong/unset | Fix in Vercel env, **redeploy** (build-time var). |
| First request after idle is 30 s | HF free cold start + regeneration | Expected; UptimeRobot ping mitigates. |
| Queries 404/fail right after a redeploy | Chroma still regenerating | Wait for the rebuild; check logs for `[regenerate]`. |
| Upgrade button does nothing / 502 | Stripe webhook not wired in prod | Register the webhook endpoint + set its signing secret on the Space. |
| `users.tier` never flips to pro | Webhook signature mismatch | The Space's `STRIPE_WEBHOOK_SECRET` must be the **dashboard endpoint's** secret, not `stripe listen`'s. |
| 401 on every authed request | Clerk domain not allowed / clock skew | Add the domain in Clerk; `api/auth.py` already allows 60 s leeway. |
| 429s under normal use | Rate limit too tight behind a shared IP | Raise the `120/minute` default in `api/main.py`. |

---

## Env var reference

**Backend (HF Space secrets):** `GROQ_API_KEY`, `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `VLM_GEMINI_API_KEY`, `MISTRAL_API_KEY`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `CLERK_ISSUER`, `DATABASE_URL`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `ALLOWED_ORIGINS`, `PAPERMIND_FRONTEND_URL`. Optional: `PAPERMIND_MAX_UPLOAD_MB` (25), `PAPERMIND_REGENERATE_ON_STARTUP` (1), `PAPERMIND_FREE_MAX_PAPERS` (3), `PAPERMIND_FREE_MAX_QUERIES_PER_MONTH` (20).

**Frontend (Vercel, build-time):** `VITE_API_URL`, `VITE_CLERK_PUBLISHABLE_KEY`.

---

## TL;DR

1. Rotate every API key.
2. (Code prereqs already done in §4: Dockerfile, CORS, validation, rate limiting, regeneration.)
3. HF Space (Docker) → push repo → set backend secrets → `/health` ok.
4. Vercel → root `frontend` → set `VITE_API_URL` + `VITE_CLERK_PUBLISHABLE_KEY` → deploy → add custom domain.
5. Set `ALLOWED_ORIGINS`/`PAPERMIND_FRONTEND_URL` to the domain; register the Stripe webhook in the dashboard; add the domain in Clerk.
6. Test end to end; set LLM spend caps; add an UptimeRobot ping.

Cost: **$0 + ~$10/yr for the domain.** Upgrade to $5/mo HF storage (or a paid always-on backend) once it's earning.

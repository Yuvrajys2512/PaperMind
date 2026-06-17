# PaperMind — Launch Checklist (Demo → Paid Freemium Product)

What stands between the current codebase and a real product at a real URL that people pay for.

The core pipeline is genuinely differentiated, and [DEPLOYMENT.md](DEPLOYMENT.md) already covers the hosting mechanics. But there is a real gap between **"demo on a URL"** (a weekend) and **"freemium product people pay for"** (several focused weeks). The gap is almost entirely in three areas: **multi-tenancy**, **LLM economics**, and **billing**.

---

## 1. Multi-tenancy — the hard blocker

Right now PaperMind is a single-user app. `api/storage.py` writes everything to one shared `data/papers.json` and one `data/papers/` folder, and ChromaDB has one shared index. Deployed as-is, every visitor would see every other visitor's uploaded papers — and could delete them.

- [x] **Auth** — don't build it yourself; use Clerk, Supabase Auth, or Auth0. Email + Google sign-in is enough.
- [x] **Per-user data scoping** — every paper record, chunk, and Chroma entry needs a `user_id`, and every query/list/delete endpoint must filter by it.
- [x] **Real storage instead of local disk**:
  - [x] `data/papers.json` → **Postgres** (Neon / Supabase free tier is fine)
  - [x] PDFs → **object storage** (S3 or Cloudflare R2)
  - [x] ChromaDB → either keep it with per-user metadata filtering on a persistent volume, or move to a hosted vector DB *(turned out to need neither — already one collection per paper, so registry-level ownership checks are sufficient; kept local since it's fully regenerable from the R2-stored PDF)*

> DEPLOYMENT.md calls the storage refactor "a v2 problem" — that's correct for a demo. For a **paid** product it's a v1 problem: ephemeral-disk data loss is unacceptable when people pay.

---

## 2. LLM economics — the freemium model lives or dies here

The provider chain (Groq → Gemini → Mistral → Cerebras) currently runs entirely on **free-tier keys under a personal account**. That breaks immediately with real users:

- Free quotas are shared across *all* users — ten active users will exhaust them in a day (we already hit Gemini's free-quota-0 problem on 2.0-flash).
- Most free tiers **prohibit commercial use** in their ToS.

To do:

- [ ] **Paid API keys** for at least the primary provider. *(Manual action, not code — every provider key in `ingestion/llm_client.py` is read purely from env vars, so upgrading billing on Groq/Gemini/etc. needs zero code changes whenever it happens.)*
- [x] **Per-request cost tracking** — tokens in/out, logged per user per query. A multi-hop query with planning + generation + sentence-by-sentence evidence grading + faithfulness eval is *many* LLM calls. **Measure what one query actually costs before pricing anything.** *(`ingestion/llm_client.py` now captures `usage.prompt_tokens`/`completion_tokens` per call and prices them against a public-list-price table; `api/usage.py`'s `usage_events` table logs tokens/cost per user for both queries and paper uploads, via `record_usage()` called from `api/main.py` and `api/ingestion_runner.py`. `GET /usage` surfaces the running totals.)*
- [x] **Per-user quotas enforced in middleware** — e.g. free = 3 papers + 20 queries/month. This is the same machinery the paid tier needs. *(Implemented as FastAPI dependencies — `enforce_paper_quota`/`enforce_query_quota` in `api/usage.py`, matching the existing `get_current_user_id` dependency pattern rather than ASGI middleware. Limits are env-configurable (`PAPERMIND_FREE_MAX_PAPERS`, `PAPERMIND_FREE_MAX_QUERIES_PER_MONTH`); a `users.tier` column exists for §3/Stripe to flip later.)*
- [ ] Consider **disabling the expensive stages on the free tier** (evidence grading, retry engine) and making "full verified answers" the paid hook. *(Deferred — there's no paid tier yet to contrast against, so the trade-off isn't meaningful until §3 lands and real usage data exists.)*

---

## 3. Billing

- [ ] **Stripe Checkout + customer portal** (the standard path).
- [ ] A `subscriptions` table + webhook handler for subscription created/cancelled.
- [ ] **Entitlement check** in the same middleware that enforces quotas.
- [ ] **Decide the free/paid split early.** The natural split for this product:
  - **Free** — limited papers/queries, basic answers
  - **Paid** — unlimited papers, evidence-graded answers, multi-hop/comparison queries

A few days of work once auth exists.

---

## 4. Deployment & hardening

(DEPLOYMENT.md covers most of the mechanics.)

- [ ] Dockerize the backend; host on Render / Railway / Fly with ≥2 GB RAM.
- [ ] Frontend on Vercel / Cloudflare Pages; buy a domain.
- [ ] **Lock down CORS** from `"*"` to the real frontend origin (already flagged in DEPLOYMENT.md).
- [ ] **Upload validation** — max file size, PDF-only enforcement *server-side*, per-IP rate limiting on unauthenticated routes. An open `/upload` endpoint is an invitation to fill the disk.
- [ ] **Pre-download the embedding/reranker models into the Docker image** so cold starts don't take 30s.
- [ ] Move ingestion to a proper background queue *eventually* — in-process background tasks are fine at launch.

---

## 5. Product & legal minimum

- [ ] **Sentry** (error tracking) and **PostHog** (analytics) — both free tier, ~an hour each. Can't run a product blind.
- [ ] **Landing page** with a demo GIF.
- [ ] **Preloaded sample papers** so a new user gets value in 30 seconds without uploading anything.
- [ ] **Terms of Service + Privacy Policy** — users upload PDFs that are often copyrighted; the ToS must put upload-rights responsibility on the user.

---

## Suggested order

1. **Postgres + object storage refactor** (replaces `storage.py`)
2. **Auth + per-user scoping**
3. **Deploy behind a real URL, free tier only** — invite-only, no payments yet
4. **Cost instrumentation + quotas** — watch real usage for a couple of weeks
5. **Stripe + paid tier** — set prices from the *measured* costs

Steps 1–3 are the bulk of the engineering (~2–3 weeks focused). Steps 4–5 are smaller but should **follow real usage data, not precede it**.

> **Marketing note:** the QASPER eval harness is an asset here — *"answers benchmarked on QASPER"* is a credibility line none of the wrapper competitors have.

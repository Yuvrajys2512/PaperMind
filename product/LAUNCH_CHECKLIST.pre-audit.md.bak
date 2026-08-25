# PaperMind — Launch Checklist (Demo → Paid Freemium Product)

What stands between the current codebase and a real product at a real URL that people pay for.

The core pipeline is genuinely differentiated, and [DEPLOYMENT.md](DEPLOYMENT.md) already covers the hosting mechanics. But there is a real gap between **"demo on a URL"** (a weekend) and **"freemium product people pay for"** (several focused weeks). The gap is almost entirely in three areas: **multi-tenancy**, **LLM economics**, and **billing**.

---

## ▶ START HERE — pick up next session (as of 2026-06-18)

**The whole build is done.** Launch Checklist §1–§5 are code-complete and verified. The 3 sample papers (Attention, BERT, RAG) are seeded live and were end-to-end verified (read-only + quota-exempt + queryable with cited answers). **Everything left is manual deploy/ops — there is no more feature code to write before launch.**

### 0. Commit the working tree first
The last session's work is **uncommitted**. Suggested split (matches the repo's commit style; no AI attribution):
```
add public landing page + ToS/Privacy pages at #/terms,#/privacy
add preloaded sample papers (shared read-only demo set); fix duplicate chunk-ID ingestion crash
```
Notable change to keep in the commit: **`ingestion/embedder.py`** — chunk IDs were a bare `md5(text)`, so two identical chunks in one paper crashed the *entire* ingestion (`DuplicateIDError`). Now prefixed with the chunk's ordinal position. This is a real fix for **all** uploads, not just demo seeding.

### 1. Then the deploy (the actual launch path) — follow [DEPLOYMENT.md](DEPLOYMENT.md)
Do them in this order:

1. **(Optional but recommended) Local Docker smoke test** — `docker build -t papermind .` → `docker run --rm -p 7860:7860 --env-file .env papermind` → `curl localhost:7860/health`. Catches a deploy-breaker before pushing to HF. (Claude can run this if Docker Desktop is up.)
2. **Rotate every API key** (DEPLOYMENT.md Part 3) — they were pasted into chats during dev; treat as burned.
3. **Backend → Hugging Face Spaces** (Docker) — push repo, set the backend secrets, confirm `/health`. (Part 5)
4. **Frontend → Vercel** — root dir `frontend`, set `VITE_API_URL` + `VITE_CLERK_PUBLISHABLE_KEY`, deploy, add custom domain. (Part 6)
5. **Wire the cross-service settings** — `ALLOWED_ORIGINS` / `PAPERMIND_FRONTEND_URL` on HF; register the **production Stripe webhook** + its signing secret; add the domain in Clerk; UptimeRobot keep-warm ping. (Part 7)

> **DEPLOYMENT.md env reference was corrected this session** to match the actual code (it predated §5). It now lists `GROQ_API_KEY_2`, `SENTRY_DSN`/`SENTRY_ENVIRONMENT`, `SEMANTIC_SCHOLAR_API_KEY`, `PAPERMIND_DEMO_USER_ID`, and the frontend `VITE_SENTRY_*`/`VITE_POSTHOG_*`; and dropped `OPENAI_API_KEY` / `VLM_GEMINI_API_KEY` / `GEMINI_API_KEY_2`, which the code never reads. Trust the doc's env tables now.

### 2. The remaining manual setup tasks (can happen alongside / after deploy)
- [ ] **Stripe test-mode e2e** — create the $9/mo Price, `stripe listen`, complete Checkout with `4242…`. Needs your test keys. (See LAUNCH_DOCUMENT.md Part 3.)
- [ ] **Create Sentry + PostHog projects** and add the DSN/keys (HF secret `SENTRY_DSN`; Vercel `VITE_SENTRY_DSN`/`VITE_POSTHOG_KEY`/`VITE_POSTHOG_HOST`). Until then observability is a no-op (fail-open by design).
- [ ] **Fill the legal placeholders** — greppable `[[OPERATOR]]` / `[[JURISDICTION]]` / `[[CONTACT_EMAIL]]` in `frontend/src/legal/content.js`, then have a lawyer review the ToS/Privacy.
- [ ] **Add `frontend/public/demo.gif`** — the landing page shows a labelled placeholder until it exists (no code change needed).

### Other open track (not launch-blocking)
- **QASPER eval / workshop-paper** — living roadmap in `research/to_do.md`. Pending: wire `generator.py` `PAPERMIND_GEN_MODEL`, a clean grader re-run, and the broken Cerebras provider (`NotFoundError`).

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

- [x] **Stripe Checkout + customer portal** (the standard path). *(`api/billing.py`: `POST /billing/checkout` creates a subscription Checkout Session; `POST /billing/portal` opens the hosted customer portal. Frontend `BillingPage.jsx` + a "Plan" nav link drive both.)*
- [x] A `subscriptions` table + webhook handler for subscription created/cancelled. *(`subscriptions` table on Neon maps `user_id ↔ stripe_customer_id`; `POST /billing/webhook` verifies the Stripe signature and handles `checkout.session.completed` / `customer.subscription.updated` / `customer.subscription.deleted`.)*
- [x] **Entitlement check** in the same machinery that enforces quotas. *(No new middleware — entitlement **is** the tier: the webhook flips `users.tier` to `pro`/`free`, and the existing `enforce_paper_quota`/`enforce_query_quota` dependencies already read `tier`. `pro` is unlimited via `TIER_LIMITS`.)*
- [x] **Decide the free/paid split early.** *(Decided: **Free** = 3 papers + 20 queries/mo; **Pro = $9/mo** = unlimited papers + queries. Feature-level stage-gating (evidence grading / multi-hop) stays deferred per §2 — no current user is degraded.)*

> Code-complete and mechanically verified (imports, schema creation on Neon, frontend build/lint). The remaining manual step is the **live Stripe test-mode e2e** (create the $9/mo Price, run `stripe listen`, complete Checkout with test card `4242…`) — it needs the account owner's test keys; see `LAUNCH_DOCUMENT.md` Part 3 for the exact steps.

---

## 4. Deployment & hardening

(DEPLOYMENT.md rewritten for the post-§1–§3 architecture + the free HF Spaces + Vercel + custom-domain path.)

- [x] **Dockerize the backend.** *(Root `Dockerfile` + `.dockerignore`: `python:3.12-slim`, copies `api`/`ingestion`/`discovery`, single worker, binds `${PORT:-7860}`. Target is **HF Spaces free (16 GB RAM)** — Render's free 512 MB can't run torch; Render/Fly remain the paid always-on option.)*
- [ ] **Frontend on Vercel; buy a domain.** *(Code is ready — `VITE_API_URL` makes the API base configurable. Actual Vercel project + domain purchase are manual deploy-time steps; see DEPLOYMENT.md Parts 5–7.)*
- [x] **Lock down CORS** from `"*"` to the real frontend origin. *(Env-driven `ALLOWED_ORIGINS` in `api/main.py`, default `localhost:5173`.)*
- [x] **Upload validation** — max file size + server-side PDF magic-byte enforcement + per-IP rate limiting. *(`api/uploads.py` + chunked validating copy in `/upload` and `discovery/fetcher.py`; `slowapi` global 120/min in `api/main.py`. Note: nearly every route is now Clerk-gated + §2-quota-bounded, so the unauthenticated surface is just `/health` + `/billing/webhook`.)*
- [x] **Pre-download the embedding/reranker models into the Docker image.** *(Dockerfile bakes `BAAI/bge-small-en-v1.5` + `cross-encoder/ms-marco-MiniLM-L-6-v2` — the models actually loaded; the old guide named the wrong one.)*
- [ ] Move ingestion to a proper background queue *eventually* — in-process background tasks are fine at launch. *(Deferred by design.)*

> **New for the free-tier path:** ChromaDB now **regenerates from R2 on startup** (`regenerate_missing_collections`, daemon thread in `api/main.py`), so an ephemeral-disk host needs no paid persistent volume. Code-complete and mechanically verified (imports, rate-limit 429s, regeneration no-op, frontend build). Remaining work is the **manual deploy** (HF Space + Vercel + domain + key rotation) per DEPLOYMENT.md.

---

## 5. Product & legal minimum

- [x] **Sentry** (error tracking) and **PostHog** (analytics) — both free tier. *(Sentry on backend (`api/main.py`, guarded by `SENTRY_DSN`) + frontend (`main.jsx` + `ErrorBoundary`). PostHog frontend-only via `frontend/src/analytics.js` — autocapture + identify by Clerk ID + funnel events `paper_uploaded`/`query_asked`/`plan_viewed`/`upgrade_clicked`. All **fail-open**: no-op without keys. Code-complete + verified; creating the Sentry/PostHog projects and adding the DSN/key env vars is the manual deploy-time step.)*
- [x] **Landing page** with a demo GIF. *(New public `frontend/src/pages/LandingPage.jsx` — signed-out visitors now land here (hero + features + Free/Pro pricing + footer) instead of a bare Clerk box; CTAs open Clerk via `SignInButton`/`SignUpButton mode="modal"`. The demo GIF is the one manual asset: drop `frontend/public/demo.gif` and it renders automatically — until then a labelled placeholder shows in its place.)*
- [x] **Preloaded sample papers** so a new user gets value in 30 seconds without uploading anything. *(Built as the decided **shared demo set**: papers owned by a fixed system `user_id` (`api/storage.DEMO_USER_ID`, env `PAPERMIND_DEMO_USER_ID`). New `list_demo_papers()` + `get_readable_paper()` in `storage.py`; `/papers` returns own papers then the demo set tagged `is_demo`; all **read** endpoints (status/query/stream/pdf/glossary/recommendations) use `get_readable_paper` while **delete** stays `get_owned_paper`, so the set is read-only to everyone. Quota dependencies count only `list_papers(user_id)` (own), so demo papers are quota-exempt by construction. Demo papers get a normal Chroma collection, so the startup R2-regeneration rebuilds them like any paper. Frontend `LibraryPage.jsx` shows a "Sample" badge and hides Delete for `is_demo`. **One manual step ([[feedback-flag-manual-steps]]):** run `python scripts/seed_demo_papers.py <pdf>…` once with the chosen PDFs — it runs the real ingestion pipeline (live LLM calls), so it's owner-run, not CI.)*
- [x] **Terms of Service + Privacy Policy** — users upload PDFs that are often copyrighted; the ToS must put upload-rights responsibility on the user. *(`frontend/src/legal/content.js` holds both docs; `frontend/src/pages/LegalPage.jsx` renders them at the auth-agnostic hash routes `#/terms` / `#/privacy`, linked from the landing + app footers. ToS §3 puts upload-rights responsibility squarely on the user; the Privacy Policy discloses every processor incl. that document text is sent to the LLM providers. **Manual before live:** fill the `[[OPERATOR]]` / `[[JURISDICTION]]` / `[[CONTACT_EMAIL]]` placeholders and have it reviewed by a lawyer.)*

---

## Suggested order

1. **Postgres + object storage refactor** (replaces `storage.py`)
2. **Auth + per-user scoping**
3. **Deploy behind a real URL, free tier only** — invite-only, no payments yet
4. **Cost instrumentation + quotas** — watch real usage for a couple of weeks
5. **Stripe + paid tier** — set prices from the *measured* costs

Steps 1–3 are the bulk of the engineering (~2–3 weeks focused). Steps 4–5 are smaller but should **follow real usage data, not precede it**.

> **Marketing note:** the QASPER eval harness is an asset here — *"answers benchmarked on QASPER"* is a credibility line none of the wrapper competitors have.

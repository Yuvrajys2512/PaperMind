# PaperMind — Final Day (Launch / Deploy)

**Goal of the day:** take PaperMind from "code-complete" to "live at a real URL that a stranger can sign up for and pay $9/mo on."

There is **no feature code left to write.** §1–§5 of the Launch Checklist are done and verified. Everything below is manual ops: accounts, keys, deploy, wiring. Work top to bottom — each step has a clear "done when."

Reference docs: `product/DEPLOYMENT.md` (mechanics), `product/LAUNCH_DOCUMENT.md` (the why), `product/LAUNCH_CHECKLIST.md` (status).

---

## 0. Pre-flight — clean the working tree (15 min)

The tree is dirty (landing/legal/sample-paper work + eval changes are uncommitted). Commit before you deploy so the deploy is reproducible.

- [ ] Review `git status` / `git diff`.
- [ ] Commit in two logical chunks (no AI attribution, matches repo style):
  - `add public landing page + ToS/Privacy pages at #/terms,#/privacy`
  - `add preloaded sample papers (shared read-only demo set); fix duplicate chunk-ID ingestion crash`
- [ ] Keep the eval/research changes (`eval/`, `research/to_do.md`, `ingestion/*`) in a **separate** commit — they're not launch-blocking and shouldn't be tangled with the launch commit.

**Done when:** `git status` is clean and you can name what's in each commit.

---

## 1. Local Docker smoke test (30 min) — optional but do it

Catches a deploy-breaker before you waste time pushing to HF.

- [ ] Docker Desktop running.
- [ ] `docker build -t papermind .`
- [ ] `docker run --rm -p 7860:7860 --env-file .env papermind`
- [ ] `curl localhost:7860/health` → 200.

**Done when:** the container boots and `/health` answers. (I can run this step for you if Docker Desktop is up.)

---

## 2. Rotate every API key (45 min) — DO NOT SKIP

Every key was pasted into chat during dev. Treat all of them as **burned**. Regenerate, don't reuse.

Rotate (and have the new value ready to paste into HF/Vercel secrets):
- [ ] `GROQ_API_KEY`, `GROQ_API_KEY_2`
- [ ] `GEMINI_API_KEY`
- [ ] `MISTRAL_API_KEY`
- [ ] `CEREBRAS_API_KEY` (broken anyway, but rotate)
- [ ] `CLERK_SECRET_KEY` + publishable key
- [ ] `DATABASE_URL` (Neon — rotate the role password)
- [ ] `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`
- [ ] `SEMANTIC_SCHOLAR_API_KEY`
- [ ] Stripe keys (handled in step 6)

**Done when:** the old keys are revoked in each provider dashboard and you hold a fresh set.

---

## 3. Backend → Hugging Face Spaces (Docker) (45 min)

HF free tier = 2 vCPU / 16 GB RAM (Render free 512 MB can't boot torch — don't use it).

- [ ] Create a new **Docker** Space.
- [ ] Push the repo to it.
- [ ] Set **all backend secrets** in the Space settings (use DEPLOYMENT.md's env table — it was corrected to match the real code): Clerk, Neon, R2, all provider keys, `SENTRY_DSN`/`SENTRY_ENVIRONMENT`, `PAPERMIND_DEMO_USER_ID`.
- [ ] Wait for build, hit `https://<space>.hf.space/health` → 200.

**Done when:** `/health` answers on the public Space URL. (Note its URL — the frontend needs it.)

---

## 4. Frontend → Vercel + domain (45 min)

- [ ] New Vercel project, **root dir = `frontend`**.
- [ ] Env vars: `VITE_API_URL` = the HF Space URL, `VITE_CLERK_PUBLISHABLE_KEY`, plus `VITE_SENTRY_DSN`/`VITE_POSTHOG_KEY`/`VITE_POSTHOG_HOST` (after step 7).
- [ ] Deploy.
- [ ] Buy + attach the custom domain.

**Done when:** the domain loads the landing page over HTTPS.

---

## 5. Wire the cross-service settings (30 min)

These are the "it works in isolation but the two halves can't talk" settings.

- [ ] On HF: set `ALLOWED_ORIGINS` = your domain, `PAPERMIND_FRONTEND_URL` = your domain.
- [ ] In Clerk: add the production domain to allowed origins.
- [ ] UptimeRobot (or similar): a keep-warm ping to `/health` every ~30 min so HF doesn't idle-sleep mid-demo.

**Done when:** signing in on the live domain works end-to-end (no CORS/401 errors in the console).

---

## 6. Stripe — go live on billing (45 min)

You can do this in **test mode** first to verify, then flip to live.

- [ ] In the Stripe **test** dashboard: create a Product with a **$9/mo recurring Price** → put its id in `STRIPE_PRICE_ID`.
- [ ] Put the test secret key in `STRIPE_SECRET_KEY`.
- [ ] Register the **production webhook**: endpoint `https://<space>.hf.space/billing/webhook`, copy its signing secret into `STRIPE_WEBHOOK_SECRET`.
- [ ] On the live site: **Plan → Upgrade**, pay with test card `4242 4242 4242 4242`.
- [ ] Confirm: `checkout.session.completed` fires → a `subscriptions` row appears → `users.tier` flips to `pro` → `GET /usage` shows unlimited caps → a 4th paper / 21st query is no longer blocked.
- [ ] Cancel via the portal → tier back to `free`, caps re-enforced.
- [ ] When satisfied, swap test keys for **live** keys.
- [ ] Clean up any synthetic test rows from Neon.

**Done when:** a real upgrade flips you to Pro and a cancel flips you back.

---

## 7. Observability + final assets (30 min)

- [ ] Create a **Sentry** project → `SENTRY_DSN` (HF) + `VITE_SENTRY_DSN` (Vercel).
- [ ] Create a **PostHog** project → `VITE_POSTHOG_KEY` (+ `VITE_POSTHOG_HOST`) (Vercel).
- [ ] Redeploy frontend so the analytics keys take effect.
- [ ] Trigger a test error and confirm it lands in Sentry; do a signup and confirm `paper_uploaded`/`query_asked` events land in PostHog.

**Done when:** an error shows in Sentry and a funnel event shows in PostHog.

---

## 8. Legal + demo polish (before sharing the link publicly)

- [ ] Fill the greppable placeholders in `frontend/src/legal/content.js`: `[[OPERATOR]]`, `[[JURISDICTION]]`, `[[CONTACT_EMAIL]]`.
- [ ] Have a lawyer (or at minimum a careful read) review ToS/Privacy — you're hosting user-uploaded copyrighted PDFs and sending text to third-party LLMs; the ToS must keep upload-rights responsibility on the user.
- [ ] Add `frontend/public/demo.gif` (landing page shows a labelled placeholder until then — no code change needed).
- [ ] Run `python scripts/seed_demo_papers.py <pdf>...` **once** with the chosen sample PDFs (real ingestion, live LLM calls — owner-run, not CI). Confirm the 3 sample papers appear with a "Sample" badge and are queryable.

**Done when:** legal pages name the real entity, the landing GIF renders, and a brand-new account sees queryable sample papers on first login.

---

## ⚠️ The one thing that's a real prerequisite, not just polish

**Free-tier provider keys cannot legally/practically carry paid traffic.** Most free LLM tiers prohibit commercial use and share quotas across *all* your users — a handful of active users will exhaust them in a day. Upgrading to **paid keys is zero code** (everything reads from env), but you should not charge anyone until at least your primary provider (Groq) is on a paid plan. Put a card on the Groq account before you flip Stripe to live mode.

---

## Definition of "launched"

A person who is not you can:
1. open the domain, see the landing page,
2. sign up (Clerk),
3. query a sample paper and get a cited answer in 30 seconds,
4. upload their own paper,
5. hit the free cap and upgrade to Pro with a real card,
6. and you can see the error/usage telemetry for all of it.

When all six are true on the live domain, you're live.

---

## NOT part of today (separate track — don't let it bleed in)

The QASPER eval / workshop-paper work (`research/`, `eval/`) is **not** launch-blocking. Keep it out of today entirely. See the research verdict below / in chat for what that track is actually worth.

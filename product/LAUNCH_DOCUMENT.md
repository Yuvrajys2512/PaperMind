# PaperMind — Build Log (Launch Checklist §1–§5)

This is a narrative record of how PaperMind went from a single-user demo toward a real product — what we built, in what order, why each decision was made, and how we verified it actually worked. It covers the work behind every checked box in `LAUNCH_CHECKLIST.md` §1 (multi-tenancy) and §2 (LLM economics) so far. Read this top to bottom and you should understand the whole thing without having to reconstruct it from diffs.

**Part 1 (§1 — multi-tenancy)** covers auth, per-user data scoping, and the move to real persistent storage. **Part 2 (§2 — LLM economics)** picks up from there: per-request cost tracking and per-user quotas.

---

# Part 1 — Multi-Tenancy (§1)

## 1. The problem we were solving

Before this work, `api/storage.py` wrote every paper to one shared `data/papers.json` file and one shared `data/papers/` folder, and there was zero authentication on any route. That's fine for a demo running on your own laptop. It is **not** fine the moment a second person can reach the URL: every visitor would see every other visitor's uploaded papers, and could delete them. That's the "hard blocker" the checklist names — you cannot charge money for a product where customers can see and destroy each other's data.

The fix has three independent layers, and we built them in this order:

1. **Auth** — establish *who* is making each request.
2. **Per-user data scoping** — use that identity to make sure each user only ever sees their own data.
3. **Real persistent storage** — move off local disk so a server restart, redeploy, or disk failure doesn't lose everything.

Each layer depends on the one before it — scoping is meaningless without an identity to scope by, and it doesn't matter whether storage is durable if anyone can read anyone else's data anyway.

---

## 2. Layer 1 — Auth (Clerk)

### Why Clerk

The checklist suggested Clerk, Supabase Auth, or Auth0. We picked **Clerk** because:
- It ships polished, ready-made React UI components (`<SignIn/>`, `<UserButton/>`) — no custom login form to build or secure.
- Supabase Auth only made sense if we were *also* using Supabase for the database, which we weren't (we ended up on Neon — see §4).
- Auth0 is more enterprise-oriented and has more configuration overhead than a solo project needs at this stage.

### How it actually works

**Frontend** (`frontend/src/main.jsx`, `frontend/src/App.jsx`):
The whole app is wrapped in `<ClerkProvider publishableKey={...}>`. Inside that, `<SignedOut>` renders Clerk's built-in `<SignIn/>` component; `<SignedIn>` renders the real app plus a `<UserButton/>` for sign-out. Clerk's React SDK also sets a global `window.Clerk` object — this matters because `frontend/src/api.js` is a plain module of `fetch()` wrapper functions, not a React component, so it can't use the `useAuth()` hook. Instead, every API call does:

```js
async function authHeaders() {
  const token = await window.Clerk?.session?.getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}
```

and merges that into its request headers. This is the standard documented pattern for getting a Clerk token from non-component code.

**Backend** (`api/auth.py`):
Clerk issues short-lived signed JWTs (60-second validity, by design — forces frequent silent refresh). Rather than pull in Clerk's backend SDK (heavier dependency, less common, less battle-tested for FastAPI specifically), we verify tokens manually — the approach Clerk itself documents for arbitrary backends:

1. Fetch Clerk's public signing keys from `{CLERK_ISSUER}/.well-known/jwks.json`, cached in memory for an hour.
2. On each request, read the `Authorization: Bearer <token>` header, look up the matching key by the token's `kid`, and verify the signature + claims with `PyJWT`.
3. Return the verified `sub` claim (Clerk's user ID) as `user_id`.
4. This is wired in as a FastAPI dependency — `user_id: str = Depends(get_current_user_id)` — added to every route except `/health`.

### The bug we hit, and what it taught us

After wiring everything up, every authenticated request failed with `401`, even though the frontend was correctly attaching a real, well-formed token. We debugged it methodically rather than guessing:

1. Confirmed in the browser console that `window.Clerk.session.getToken()` really did return a token.
2. Decoded the token's header and payload in the console — `iss` matched our configured `CLERK_ISSUER` exactly, and the `kid` matched a real key in Clerk's JWKS endpoint. So the token itself was fine.
3. Added temporary exception logging to the backend and had the request retried — the log showed the real Python exception: `jwt.exceptions.ExpiredSignatureError: Signature has expired`.
4. Compared the local machine's clock against Clerk's server clock (via the `Date` response header) — the local machine was running **~63 seconds ahead**. Since Clerk's tokens are only valid for 60 seconds, that skew alone was enough to make every token look expired the instant it arrived.

**Fix**: added `leeway=60` to the JWT verification call, so it tolerates normal clock drift. This is a permanent, sensible addition — any real server can drift a little, and a tolerance equal to one token's lifetime isn't a meaningful security loosening. (Separately: if your machine's clock is regularly off by a minute, it's worth a `Settings → Time & Language → Date & Time → Sync now` — clock drift can bite other things too, like TLS.)

**The lesson worth keeping**: when something fails mysteriously, narrow it down by actually looking at evidence at each layer (token contents, then exact exception, then root cause) rather than guessing fixes. Each step here ruled something out concretely before moving to the next.

---

## 3. Layer 2 — Per-user data scoping

Once we know *who* is asking, every route needs to actually use that identity. The mechanism is one function in `api/storage.py`:

```python
def get_owned_paper(paper_id, user_id):
    paper = get_paper(paper_id)
    if not paper or paper.get("user_id") != user_id:
        return None
    return paper
```

Every route that touches a specific paper (`/status/{id}`, `DELETE /papers/{id}`, `/papers/{id}/pdf`, `/glossary`, `/recommendations`, `/query`, `/query/stream`) calls this instead of a bare lookup. **A paper that exists but belongs to someone else returns the exact same 404 as a paper that doesn't exist at all** — this is deliberate. If a mismatched-owner request returned a *different* error (e.g. 403), an attacker could enumerate which paper IDs exist just by watching which error code comes back. `/papers` (the list endpoint) filters by `user_id` directly at the query level instead.

**A genuinely useful discovery from this step**: we initially expected to also need to add per-user filtering inside ChromaDB (the vector database), because the checklist assumed one shared vector index. Reading the actual retrieval code (`ingestion/retriever.py`) showed PaperMind already gives **every paper its own Chroma collection**, named by `paper_id`. There's no shared index to leak across — a user can only ever reach a paper's collection if they first pass the `get_owned_paper` check to get its ID. So that whole piece of the checklist's suggested work turned out to be unnecessary. This is a good example of why it's worth reading the actual code before assuming a generic checklist applies literally — the real architecture was already better-isolated than the checklist assumed.

---

## 4. Layer 3 — Real persistent storage (Neon Postgres + Cloudflare R2)

### Why this was needed at all

`data/papers.json` and `data/papers/*.pdf` lived on local disk. Even with per-user scoping in place, that's still a single point of failure: one disk corruption, host migration, or accidental deletion and every paying customer's data is gone, with no backup. `DEPLOYMENT.md` calls this a "v2 problem" for a free demo — correct for a demo, wrong for something people pay for.

### Why Neon (Postgres) and Cloudflare R2 (object storage)

- **Neon** over Supabase Postgres or Render's managed Postgres: generous free tier, dead-simple pooled connection string, and it's not tied to whichever host we eventually deploy the backend to — portable if we move later. Supabase would have made more sense only if we were *also* using Supabase Auth, which we weren't (we'd already picked Clerk).
- **Cloudflare R2** over AWS S3: R2 speaks the same S3 API (so the same `boto3` client code works unchanged), but has **no egress fees** and a more generous free tier — meaningfully cheaper for a budget-conscious early-stage product, with effectively the same tooling.
- **ChromaDB stayed exactly where it was** — local disk. Two reasons: it's already isolated per-paper (see §3), so it never had the multi-tenancy problem the other two stores had; and it's **fully regenerable** — if the disk holding it were ever lost, every paper's embeddings could simply be recomputed by re-running ingestion against the PDF, which now lives safely in R2. Paying for a hosted vector database to protect data that can be rebuilt from a copy you already have elsewhere isn't worth the cost or complexity at this stage.

### The design problem this created, and how we solved it

The PDF-parsing code (`ingestion/ingest_document.py`, via `pdfplumber`) needs a literal file on local disk — it can't read directly from an object store. So local disk didn't disappear; its *role* changed: from **persistent store** to **transient scratch space**, used only for the few seconds it takes to parse a file.

The resulting flow, used identically everywhere a PDF enters the system (direct upload *and* the "import from arXiv/Semantic Scholar" discovery flow):

1. Get the PDF's bytes onto a local temp file (from the upload stream, or downloaded from a URL).
2. Upload that temp file to R2 — **this is now the only permanent copy**.
3. Delete the temp file.
4. In a background task: download the PDF from R2 into a *fresh* temp file, run the existing (unchanged) ingestion pipeline against it, then delete that temp file too.

Because both entry points (direct upload and discovery-import) needed the exact same "download from R2 → ingest → clean up" logic, we factored it into one shared function, `api/ingestion_runner.py::run_ingestion_from_storage`, instead of duplicating it. (It couldn't live in `api/main.py` itself, because `discovery/router.py` is imported *by* `api/main.py` — putting it there would create a circular import.)

**Serving the PDF back to the viewer** (`GET /papers/{id}/pdf`) had a real design choice: redirect the browser to a temporary signed R2 URL, or have the backend fetch the bytes from R2 and stream them through itself. We chose **streaming through the backend**, specifically so that `get_owned_paper`'s ownership check remains the *only* gate a request has to pass to read a PDF. A signed URL, once generated, is a bearer credential usable by anyone who later obtains it (browser history, logs, a forwarded link) — completely independent of whether the original user is still authorized. For an app where every PDF view already goes through our own auth, there's no upside to that indirection. We stream rather than fully load the file into memory first, so a large PDF doesn't get buffered whole in RAM on a server running a single worker process.

**Deleting a paper** also got a deliberate ordering fix: delete the R2 object *before* the Postgres row, not after. If the PDF delete fails partway through, you're left with "the row still exists, the file is just still there too" — recoverable, retry-able. Deleting the row first and then failing to delete the PDF would leave an orphaned file in R2 with nothing pointing to it anymore, silently costing money forever with no way to find it again.

**Connection handling**: Postgres access goes through a small connection pool (`psycopg_pool.ConnectionPool`) kept open for the life of the process, rather than opening a brand-new connection on every single request. Neon's databases scale their compute down to zero when idle, and reconnecting from scratch on every request would mean re-paying that cold-start cost (TLS handshake, possible compute wake-up) far more often than necessary.

---

## 5. Tech stack — summary (§1)

| Concern | Tool | Why this one |
|---|---|---|
| Authentication | **Clerk** (`@clerk/clerk-react` + manual JWT/JWKS verification in `api/auth.py`) | Ready-made UI, no Supabase dependency elsewhere, simpler backend integration than Auth0 |
| Paper registry (was `data/papers.json`) | **Neon** (serverless Postgres, via `psycopg` + `psycopg_pool`) | Generous free tier, simple pooled connection string, host-agnostic |
| PDF storage (was `data/papers/`) | **Cloudflare R2** (via `boto3`, S3-compatible API) | No egress fees, same client code as S3, cheaper for an early-stage product |
| Vector search | **ChromaDB**, unchanged, still local disk | Already isolated per-paper; fully regenerable from R2 — no durability gap to fix |

New environment variables this introduced (all in the root `.env`): `CLERK_ISSUER`, `DATABASE_URL`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`. The frontend additionally needs `frontend/.env.local` → `VITE_CLERK_PUBLISHABLE_KEY`.

New files: `api/auth.py` (token verification), `api/ingestion_runner.py` (shared ingest-from-storage logic). Rewritten: `api/storage.py` (now a thin client over Postgres + R2 instead of JSON-file + local-disk I/O).

---

## 6. How we actually worked through this (process)

For each of the three layers, the process was the same shape, deliberately:

1. **Decide what's actually unclear, and ask.** Provider choices (Clerk vs. Supabase vs. Auth0; Neon vs. Supabase vs. Render Postgres; R2 vs. S3) are genuine judgment calls with real trade-offs — those got asked explicitly rather than silently assumed.
2. **Plan before touching code.** Each layer went through an explicit planning pass: read the actual current code first (not just the checklist's description of it), identify every call site that would be affected, write a concrete plan naming exact files and functions, and get it approved before writing anything. For the storage migration specifically, that plan was also run past a second independent review pass before finalizing — it caught several real issues that weren't obvious from the first draft (a duplicate-parsing edge case, a delete-ordering bug, the cold-start cost of not pooling Postgres connections, a preexisting bug in the discovery-import flow that the migration's review surfaced incidentally).
3. **Implement with a visible task list**, marking each piece off as it's verified — partly for transparency, partly so nothing silently gets skipped in a multi-file change.
4. **Verify mechanically before verifying by hand.** Every change was compile-checked and import-checked (catching wiring mistakes immediately, for free) before ever touching a browser. The fail-loud-on-missing-config pattern (`api/auth.py` and `api/storage.py` both raise immediately if their required environment variables are missing, rather than failing mysteriously deep inside a background task) was a deliberate design choice made *during* planning, not bolted on after a confusing failure.
5. **Verify end-to-end for real**, in the actual browser, only after the mechanical checks passed — and when something failed anyway (the clock-skew bug), debug it as a chain of concrete evidence rather than trial-and-error.

---

# Part 2 — LLM Economics (§2)

## 7. The problem, and the scope decisions

§2 of the checklist has four open items: paid API keys, per-request cost tracking, per-user quotas, and (optionally) disabling expensive pipeline stages on the free tier. The provider chain (`ingestion/llm_client.py`, Groq → Gemini → Mistral → Cerebras) was already running entirely on personal free-tier keys, with no visibility into what a query actually costs and no limit on how many a single user could run.

Three scoping decisions were made explicitly before writing any code, rather than assumed:

- **Paid API keys is not a code task.** Every provider key is already read purely from `os.getenv(...)` in `llm_client.py` — upgrading to a paid plan on Groq/Gemini/etc. is something only the account owner can do (it needs real billing credentials), and it needs zero code changes whenever it happens. Left as a manual to-do.
- **Stage-gating (disabling evidence grading/retry for free users) was deferred.** The checklist itself hedges this with "Consider..." — and there's no paid tier yet (§3/Stripe isn't built) to make that trade-off meaningful. Building it now would mean degrading the product for *every* current user with no upgrade path to offer them. Revisit once §3 exists and there's real usage data.
- **Cost tracking and quotas were built now**, using the checklist's own suggested numbers (3 papers + 20 queries/month) as the default, made tunable via environment variables rather than hardcoded — because the checklist explicitly says these limits should be set from *measured* usage, not guessed, and that measurement only starts once tracking exists.

---

## 8. Cost tracking — what it actually required

The provider chain already had a thread-local stats mechanism (`reset_stats()` / `get_stats()` in `llm_client.py`) tracking call count and which provider answered — reset at the start of each query in `ingestion/pipeline.py`, read at the end to attach `llm_calls`/`providers_used` to the response. It just discarded the one field that mattered for cost: every OpenAI-compatible response includes a `usage` object with `prompt_tokens`/`completion_tokens`, and the code never looked at it.

**What changed:**
- Added a `_PRICING` table to `llm_client.py`, keyed by `(provider, model)`, with current public list prices **looked up live** (not recalled from training data, since these change): Groq `llama-3.3-70b-versatile` $0.59/$0.79 per 1M tokens (in/out), Gemini `gemini-2.5-flash-lite` $0.10/$0.40, Mistral `mistral-small-latest` $0.20/$0.60, Cerebras `gpt-oss-120b` $0.35/$0.75.
- Important framing: PaperMind's keys are still free-tier today. This table doesn't bill anyone — it answers "what would this have cost on paid keys," which is precisely the number the checklist says to gather *before* setting any price.
- Extended the existing thread-local stats dict with `tokens_in`/`tokens_out`/`cost_usd`, populated right where `call_count` was already being incremented in `chat_completion()` — reusing the mechanism already in place rather than building a second one alongside it. An unpriced `(provider, model)` pair costs `$0` instead of raising, so adding a new model later just under-counts until it's priced, rather than crashing live traffic.
- A question the checklist doesn't address came up during planning: **uploading a paper also burns real LLM calls** (section/structure detection during ingestion, in `ingestion/section_detector.py` and friends) — not just answering questions. This was flagged explicitly rather than silently expanding scope, and the answer was yes, track it too: `api/ingestion_runner.py` now calls `reset_stats()` before ingesting and logs a `kind="upload"` usage event afterward, attributed to the paper's owning `user_id` (already on the row — no signature changes needed to thread it through).
- New module `api/usage.py` owns a `usage_events` table (one row per query or upload: tokens, cost, call count, timestamp) and a minimal `users` table (just a `tier` column, defaulting to `'free'`) — the latter exists now specifically so §3's eventual Stripe webhook has a column to flip, instead of needing its own schema migration when that day comes.
- A `GET /usage` endpoint surfaces the running totals (tier, papers used/limit, queries used/limit this month) for the current user. Not explicitly requested, but cheap given the data was already being tracked, and a natural complement to quotas.

---

## 9. Per-user quotas — what it actually required

The checklist says "enforced in middleware." We implemented it as FastAPI `Depends()` dependencies instead of ASGI-level middleware, because that's the pattern already used for auth in this exact codebase (`get_current_user_id`) — middleware has no clean way to know who the authenticated user is without re-implementing JWT verification at a different layer. `enforce_paper_quota` and `enforce_query_quota` (`api/usage.py`) each take over the same dependency slot the relevant endpoints already used: `/upload` now depends on `enforce_paper_quota`, `/query` and `/query/stream` on `enforce_query_quota` — swapping an existing wiring pattern rather than adding a new one.

Design choices worth keeping in mind:
- **Limits**: free = 3 papers + 20 queries/month, read from `PAPERMIND_FREE_MAX_PAPERS` / `PAPERMIND_FREE_MAX_QUERIES_PER_MONTH` so they can be retuned from real usage data without a deploy.
- **Fail-open on unknown tiers.** A tier with no entry in `TIER_LIMITS` (e.g. a future `"pro"` tier that Stripe has already assigned to a `users` row before the code defining its limits has shipped) is treated as *unlimited*, not blocked. That ordering race is real once §3 exists, and the failure mode of "a paying customer gets locked out" is much worse than "a brand-new tier briefly has no cap."
- **Paper count reuses `list_papers()`** — the exact same count already rendered on the Library page — rather than inventing a second definition of "how many papers does this user have."
- Off-topic / pre-LLM-guard query rejections still count against the monthly query quota (the user asked a question; that consumes one of their 20 "asks," even if it cost $0 in tokens) — kept this simple rather than special-casing it.

---

## 10. A gap caught before this shipped

The first implementation pass only touched the backend. Checking `frontend/src/api.js` against it surfaced a real problem: `uploadPaper`, `queryPaper`, and `comparePapers` all discarded the backend's actual error body and threw a hardcoded generic message — `Error('Upload failed')`, `Error('Query failed')` — on *any* non-OK response. A 429 with a specific "you've used all 3 of your free papers" message would have rendered to the user as a meaningless, indistinguishable failure. Two other functions already in the same file (`rewriteText`, `importPaper`) had the correct pattern — parse the JSON body, throw `err.detail` if present. That pattern was applied to the other three call sites plus the SSE `streamQuery` path.

One more layer of the same bug existed in `frontend/src/pages/UploadPage.jsx`: its catch block didn't even bind the caught error (`catch { setError('Upload failed. Is the server running?') }`), so fixing `api.js` alone wouldn't have been enough — the real message would have been thrown away a second time, on the way into `setError`. Fixed to `catch (err) { setError(err.message || '...') }`.

This is the kind of gap that's easy to miss because the backend behaves correctly in isolation (curl/Postman would show the real `detail` field) — it only shows up once you trace the exact path a real error takes through the actual frontend code that's already there.

---

## 11. Verification

- Confirmed `api/usage.py`'s `_ensure_schema()` creates `users` and `usage_events` on the real Neon database with no errors, by importing the module directly.
- Exercised `get_user_tier`, `record_usage`, `count_queries_this_month`, and `get_usage_summary` end-to-end against a throwaway test user ID — then deleted those rows immediately afterward so no synthetic data was left sitting in the production database.
- Made one real (free-tier) call through `chat_completion` and checked the cost by hand rather than trusting the output blindly: 38 input tokens and 3 output tokens against Groq's $0.59 / $0.79 per-million pricing comes out to exactly **$0.0000248** — and that's exactly what the code reported, confirming the formula is wired correctly end to end, not just "returning a number that looks plausible."
- Ran the frontend's ESLint and a full production `vite build` after the `api.js` / `UploadPage.jsx` changes — both clean, no new warnings.

---

## 12. Tech stack / files — summary (§2)

| Concern | Tool / mechanism | Why |
|---|---|---|
| Token + cost capture | `response.usage` (OpenAI-compatible field, already present, previously discarded) | No new dependency — every provider already returns this |
| Cost pricing | Hardcoded `_PRICING` table in `llm_client.py`, sourced from live provider pricing pages | Keys are free-tier; this projects paid-tier cost for pricing decisions later |
| Usage + tier storage | New tables on the existing Neon Postgres (`users`, `usage_events`) | Reuses the connection pool and DB already in place from §1 — no new infra |
| Quota enforcement | FastAPI `Depends()` dependencies (`enforce_paper_quota`, `enforce_query_quota`) | Matches the existing auth dependency pattern, not new ASGI middleware |

New file: `api/usage.py`. Modified: `ingestion/llm_client.py` (token/cost capture), `ingestion/pipeline.py` (surfaces `tokens_in`/`tokens_out`/`cost_usd` on every result), `api/main.py` (quota dependencies wired onto `/upload`, `/query`, `/query/stream`; new `GET /usage`), `api/ingestion_runner.py` (upload-time cost logging), `frontend/src/api.js` and `frontend/src/pages/UploadPage.jsx` (real error messages reach the user).

New environment variables (optional, both have defaults): `PAPERMIND_FREE_MAX_PAPERS` (default `3`), `PAPERMIND_FREE_MAX_QUERIES_PER_MONTH` (default `20`).

---

## 13. What's done, what isn't

**§1 — done and verified** (multi-tenancy, in full): auth, per-user scoping, Postgres + R2 storage. A real upload → background ingest → question-answer → PDF-preview cycle has been run end-to-end against the live Clerk, Neon, and R2 services, with no errors in the backend log.

**§2 — partially done:**
- Done and verified: per-request cost tracking (queries *and* uploads) and per-user quotas (3 papers / 20 queries/month, env-configurable), including a `GET /usage` endpoint and legible error messages on the frontend when a limit is hit.
- Not done: **paid API keys** (manual action for whoever owns the provider accounts — no code blocker) and **disabling expensive stages on the free tier** (deliberately deferred — there's no paid tier yet to contrast against, and the checklist itself only says "consider").

**§3 (Stripe billing)**: code-complete and mechanically verified — see Part 3 below. Live test-mode e2e is the one remaining manual step.

**§4 (deployment & hardening)**: code-complete and mechanically verified — see Part 4 below. The live deploy (HF Space + Vercel + domain) is manual.

**§5 (product & legal minimum)**: nearly done — observability (Sentry + PostHog), the public landing page, and the ToS/Privacy Policy are all built and verified, see Part 5. Only **preloaded sample papers** remain (design settled, implementation deferred).

---

# Part 3 — Stripe Billing (§3)

## 14. The problem, and why it was small

§2 ended by deliberately leaving two things in place that turn out to be most of §3's foundation: a `users.tier` column (default `'free'`) and a quota system that reads that column and **fails open** on any tier it doesn't recognise. So §3 wasn't "build billing from scratch" — it was "let Stripe move a user between tiers, and make `pro` mean unlimited." Everything that *enforces* the difference (the quota dependencies on `/upload`, `/query`, `/query/stream`) already existed and needed no changes at all.

Four product decisions were made with the user up front rather than assumed:
- **Pro = lift quotas only.** `pro` → unlimited papers + queries. The §2 decision to *not* gate features (evidence grading, multi-hop) by tier stays in force — gating now would degrade the product for every existing free user with nothing to upgrade *to* yet. Revisit once there's real paid usage.
- **$9/mo, single recurring price**, referenced by `STRIPE_PRICE_ID` env var so annual/other prices can be added later without code.
- **Test-mode keys** available → verification is via the Stripe CLI + test card.
- **Full upgrade UX**, not backend-only — otherwise there's no way for a user to actually pay.

## 15. The entitlement model — why there's no new "entitlement middleware"

The checklist says "entitlement check in the same middleware that enforces quotas." Taken literally that sounds like a new gate. But the quota dependencies from §2 (`enforce_paper_quota` / `enforce_query_quota` in `api/usage.py`) *already* resolve `users.tier` → `TIER_LIMITS` on every protected request. A `pro` user whose limits are unlimited passes them for free. So **the tier lookup already is the entitlement check** — the only thing §3 adds is the thing that *sets* the tier (Stripe), plus making `pro` resolve to unlimited.

Making `pro` unlimited was a one-line change: `"pro": None` in `TIER_LIMITS`. `None` rides the *exact same* code path (`_limits_for` → `None` ⇒ no caps) that §2 already used for unknown tiers, so there was no new branch to enforce or test. This is why §3 touched the pipeline and the quota code essentially not at all.

## 16. `api/billing.py` — the new module

Built to mirror `api/usage.py`'s shape exactly (reuse the `_pool` from `api/storage.py`, an `_ensure_schema()` run at import, an `APIRouter` included from `main.py` next to `discovery_router`), so it reads like the code already there.

- **`subscriptions` table** keyed by `user_id`, storing `stripe_customer_id` (UNIQUE), `stripe_subscription_id`, `status`, `current_period_end`. Two reasons for the customer-id mapping: a returning user reuses one Stripe customer instead of spawning duplicates, and webhook events — which identify the account only by `customer` — can be mapped back to *our* `user_id`.
- **Table ownership stays clean**: `usage.py` owns `users`, `billing.py` owns `subscriptions`. Billing flips the tier through a new `usage.set_user_tier(user_id, tier)` helper rather than writing the `users` table directly, so all `users` writes stay in one module.
- **Fail-loud config**, matching `auth.py`/`storage.py`: the module raises at import if `STRIPE_SECRET_KEY` / `STRIPE_PRICE_ID` / `STRIPE_WEBHOOK_SECRET` are missing, rather than failing deep inside a request later.
- **Three endpoints**: `POST /billing/checkout` (auth'd; get-or-create customer → subscription Checkout Session, stamping `client_reference_id=user_id` *and* `subscription_data.metadata.user_id` so the user is recoverable from either the session or the subscription), `POST /billing/portal` (auth'd; opens the hosted portal for the user's customer), and `POST /billing/webhook`.

## 17. The webhook — the one genuinely security-sensitive piece

The webhook is the only route in the whole app that is **not** behind Clerk auth — Stripe calls it, not a logged-in browser. Its authenticity comes entirely from the Stripe signature over the **raw request body**, so it reads `await request.body()` (the exact bytes) and verifies with `stripe.Webhook.construct_event(...)`; any failure (bad or missing signature, malformed payload) returns **400**, never 200 or 500. It handles three events:
- `checkout.session.completed` → record the subscription, `set_user_tier(user_id, "pro")`.
- `customer.subscription.updated` → tier follows status: `pro` while `active`/`trialing`, else `free`.
- `customer.subscription.deleted` → `set_user_tier(user_id, "free")`.

Every verified event returns `{"received": true}` (even ones we don't act on) so Stripe stops retrying. Webhook-driven tier is treated as the source of truth — acceptable for launch because Stripe retries failed deliveries and the customer portal reconciles any user-initiated change.

## 18. Frontend — making it actually usable

- `frontend/src/api.js`: added `startCheckout()`, `openBillingPortal()`, and `getUsage()` (no wrapper for the existing `GET /usage` existed yet). Also added a small `httpError(res, fallback)` helper that **attaches the HTTP status to the thrown Error** — so the UI can tell a quota `429` apart from any other failure without brittle string-matching. Refactored `uploadPaper` / `queryPaper` / `streamQuery` onto it (they previously discarded the status).
- New `frontend/src/pages/BillingPage.jsx`: reads `getUsage()`, shows the current tier + papers/queries meters (a `null` limit renders as "unlimited"), and a single CTA — "Upgrade to Pro — $9/mo" (→ Checkout) when free, "Manage billing" (→ portal) when pro — each redirecting to the Stripe-hosted page.
- Wiring in `App.jsx`: a new `billing` page in the existing page-state router, a "Plan" link in `UploadPage`'s top nav, and **`?billing=success` return handling** — after Stripe redirects back, the app lands on the billing page and strips the query param so a refresh doesn't re-trigger it.
- **429 upsell**: `UploadPage`'s error box now shows an "Upgrade" button (instead of "Try again") when the upload failed specifically because of a quota cap. For `ChatPage`, the existing toast already surfaces the backend's specific message ("Free tier limit of 20 queries/month reached…"); threading a CTA deeper into that component tree was judged disproportionate for v1.

## 19. Verification

- **Mechanical (done):** `stripe` installed (15.2.1, satisfies `>=11`). `import api.billing, api.usage` is clean and creates the `subscriptions` table on the real Neon DB. `import api.main` is clean and exposes exactly `/billing/checkout`, `/billing/portal`, `/billing/webhook`. Frontend `eslint src` is clean and `vite build` succeeds.
- **Live test-mode e2e (manual, pending the account owner's keys):**
  1. In the Stripe **test** dashboard, create a Product with a $9/mo recurring Price → put its id in `STRIPE_PRICE_ID`. Put the test secret key in `STRIPE_SECRET_KEY`.
  2. `stripe listen --forward-to localhost:8000/billing/webhook` → copy the printed `whsec_…` into `STRIPE_WEBHOOK_SECRET`.
  3. Start backend + frontend, sign in, **Plan → Upgrade**, pay with test card `4242 4242 4242 4242`. Confirm `checkout.session.completed` in the `stripe listen` log, a `subscriptions` row, `users.tier` flipped to `pro`, and `GET /usage` showing `tier: pro` with unlimited (null) caps.
  4. Confirm a 4th paper / 21st query is no longer blocked.
  5. Cancel via the portal → `customer.subscription.deleted` → tier back to `free`, caps re-enforced.
  6. A POST to `/billing/webhook` with a bad signature returns 400.
- Clean up any synthetic `subscriptions`/`users` test rows from Neon afterward (same discipline as §2).

## 20. Tech stack / files — summary (§3)

| Concern | Tool / mechanism | Why |
|---|---|---|
| Checkout + portal + webhooks | **Stripe** (`stripe` Python SDK) + Stripe-hosted Checkout/portal | Standard path; no card data ever touches our server |
| Subscription state | New `subscriptions` table on the existing Neon Postgres | Reuses the §1 connection pool — no new infra |
| Entitlement | Existing `users.tier` + the §2 quota dependencies | The tier check already gates quotas; `pro` = unlimited via one line in `TIER_LIMITS` |
| Upgrade UX | New `BillingPage.jsx` + "Plan" nav + 429 upsell | Makes paying reachable from the product, not just the API |

New files: `api/billing.py`, `frontend/src/pages/BillingPage.jsx`. Modified: `api/usage.py` (`"pro"` tier + `set_user_tier`), `api/main.py` (router mount), `requirements.txt` (`stripe>=11`), `frontend/src/api.js` (billing fns + status-carrying errors), `frontend/src/App.jsx` (route + return handling), `frontend/src/pages/UploadPage.jsx` ("Plan" nav + 429 upsell). New env vars: `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `PAPERMIND_FRONTEND_URL` (default `http://localhost:5173`).

---

# Part 4 — Deployment & Hardening (§4)

## 21. The problem, and the decisions that shaped it

§1–§3 made PaperMind correct and monetizable; §4 makes it *deployable as a real product* and closes the obvious holes in a public surface. Two constraints from the user framed everything: **free-tier now**, but a **real paid product on its own domain**.

Those sound contradictory for an app that loads PyTorch + two models (≈2 GB RAM), so the decisions were made explicitly rather than assumed:

- **The two halves split across two hosts, and the domain is the *frontend's*.** Users only ever see the domain (Vercel, free, custom domain). The backend's URL is invisible behind it. So a free backend host is fully compatible with "my own domain / charges money."
- **Backend → Hugging Face Spaces free, not Render free.** A quick check of current specs (not memory) settled it: Render's free tier is 512 MB RAM — it can't even boot torch. HF Spaces free is 2 vCPU / **16 GB RAM**. It's the only free option that runs this backend. Its real catches — 48 h idle-sleep and an ephemeral disk — are livable and handled (below).
- **ChromaDB durability via regenerate-on-startup, not a paid disk.** This is the interesting one (§24).
- **Rate limiting added now; job-queue deferred.** The checklist itself says in-process ingestion is fine at launch.

The code is **host-agnostic** — only `DEPLOYMENT.md` names HF/Vercel. Nothing here locks PaperMind to a host.

## 22. CORS lockdown + rate limiting

`api/main.py` had `allow_origins=["*"]` — fine for dev, wrong for a credentialed public API (and actually invalid combined with `allow_credentials=True`). Replaced with an env-driven `ALLOWED_ORIGINS` list (default `localhost:5173`), set to the real domain at deploy.

Rate limiting uses **slowapi** with a global `120/minute` per-IP default via `SlowAPIMiddleware`. We chose the middleware + `default_limits` form deliberately over the per-route `@limiter.limit` decorator: slowapi's decorator requires a `request: Request` parameter in the endpoint signature, and our query routes already bind a Pydantic model *named* `request` — the decorator would clash. The middleware applies the cap globally with no signature changes. The trade-off noted in code: the Stripe webhook and `/health` pings ride the same bucket, which is fine at launch volume (a path exemption is a later tweak if needed).

## 23. Upload validation

A `.pdf` extension proves nothing — it's trivially spoofable, and the import flow pulls from an *arbitrary URL*. New `api/uploads.py` centralizes two cheap, real checks: the bytes must start with the `%PDF-` magic number, and the file must stay under `PAPERMIND_MAX_UPLOAD_MB` (default 25). Both entry points enforce them while streaming to a temp file: `/upload` (a chunked validating copy that replaced `shutil.copyfileobj`, returning 400 for non-PDF and 413 for oversize) and `discovery/fetcher.py`'s URL download. A nice side effect in `/upload`: the registry row is now created *after* validation, so a rejected upload no longer leaves an orphaned `processing` row.

## 24. ChromaDB regenerate-on-startup — the piece worth understanding

The free HF Spaces disk is **ephemeral**: wiped on every rebuild/restart. ChromaDB lives there (§1 deliberately kept it local since it's regenerable). PDFs are in R2 and the registry in Neon — so the index can always be rebuilt. §4 makes that automatic.

- `regenerate_missing_collections()` (in `api/ingestion_runner.py`) lists all `ready` papers (`list_ready_paper_ids()` added to `api/storage.py`), checks each against the live Chroma collections, and re-ingests only the missing ones via the existing `run_ingestion_from_storage`. It is **idempotent** — papers that already have a collection are skipped, so on a warm or (future) persistent disk it's a near-instant no-op.
- A `@app.on_event("startup")` hook fires it in a **daemon thread**, so `/health` answers immediately (HF's health check passes) while papers rebuild in the background. Toggle with `PAPERMIND_REGENERATE_ON_STARTUP=0`.
- **Cost honesty:** re-ingestion calls LLMs (section detection). Since that's a server-side rebuild, not a user action, the usage event is logged as `kind="regenerate"` (a `kind` arg threaded through `run_ingestion_from_storage`) so the §2 cost dashboard never conflates it with a real user upload. It doesn't touch quotas, which are count-based.
- While here, the collection-name logic — duplicated in `ingestion/retriever.py` and the delete path in `api/main.py` — was centralized into `collection_name()`/`collection_exists()` in `retriever.py` and reused everywhere.

The escape hatch: when there are paying users, HF's $5/mo persistent storage makes the disk durable and the whole regeneration step a silent no-op — no code change.

## 25. The Dockerfile (and a bug in the old guide)

The previous `DEPLOYMENT.md` carried a Dockerfile that would have **shipped broken**: it copied only `api` and `ingestion` (missing `discovery`, which `api.main` imports), and pre-downloaded `all-MiniLM-L6-v2` — *not* a model PaperMind loads. Reading `ingestion/models.py` and `ingestion/reranker.py` showed the real hot-path models are `BAAI/bge-small-en-v1.5` (embeddings, used everywhere incl. the evaluator — whose "all-MiniLM" docstring is just stale) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker). The new root `Dockerfile` copies all three packages, bakes the two correct models so the first post-deploy request doesn't stall on a download, runs a single worker (Chroma/in-mem models aren't multi-worker safe), and binds `${PORT:-7860}` (7860 = HF default, `$PORT` keeps it portable to Render/Fly). A `.dockerignore` keeps `venv/`, `data/`, `.env`, `frontend/`, and dev-only dirs out of the image.

## 26. Frontend configurable API base

`frontend/src/api.js` hard-coded `const BASE = '/api'` (the Vite dev proxy). In prod the frontend and backend are different origins, so it's now `import.meta.env.VITE_API_URL || '/api'` — dev keeps the proxy, prod sets `VITE_API_URL` to the Space URL at Vercel build time.

## 27. Docs reconciliation

`DEPLOYMENT.md` predated §1–§3 and was actively misleading (local-disk storage, "no auth," the broken Dockerfile). It was rewritten around the real architecture and the free HF Spaces + Vercel + custom-domain path: corrected Dockerfile/models, the full env-var reference (Clerk/Neon/R2/Stripe/CORS), the key-rotation action (every key was pasted in chat), the production Stripe-webhook wiring (dashboard endpoint, not `stripe listen`), and a failure-mode table.

## 28. Verification

- **Imports:** `api.main` imports clean with the slowapi limiter, both middlewares (CORS + SlowAPI), and the startup hook wired.
- **Rate limiting (behavioral):** 135 `TestClient` GETs to `/health` returned **exactly 120×200 then 15×429** — the cap enforces precisely.
- **Regeneration (safe dry-run):** against the warm local `data/chroma_db`, the listing + existence logic reported "1 ready paper, 0 to regenerate" — confirming the idempotent no-op before trusting it to rebuild on a fresh disk. (A full rebuild-from-empty costs LLM calls, so it's validated on the real HF deploy.)
- **Frontend:** `eslint src` + `vite build` clean after the `api.js` change.
- **Upload validation:** logic is import-verified; full 400/413 behavior needs an authed running server, so it folds into the deploy-time smoke test.

## 29. Tech stack / files — summary (§4)

| Concern | Mechanism | Why |
|---|---|---|
| Public-surface limits | `slowapi` global 120/min + env-driven CORS | Cheap abuse insurance; CORS correct for a credentialed API |
| Upload safety | `api/uploads.py` magic-byte + size cap, both entry points | Extension/URL content can't be trusted |
| Free-tier durability | Regenerate Chroma from R2 on startup (daemon thread) | No paid disk; index is regenerable by design |
| Image | `Dockerfile` (3.12-slim, models baked, 3 packages, `$PORT`) | Correct + fast cold start; host-portable |

New files: `api/uploads.py`, `Dockerfile`, `.dockerignore`. Modified: `api/main.py` (CORS env, slowapi, startup regeneration, upload validation, delete dedup), `api/storage.py` (`list_ready_paper_ids`), `api/ingestion_runner.py` (`regenerate_missing_collections` + `kind` arg), `ingestion/retriever.py` (`collection_name`/`collection_exists`), `discovery/fetcher.py` (download validation), `frontend/src/api.js` (configurable base), `requirements.txt` (`slowapi`), and a rewritten `product/DEPLOYMENT.md`. New env vars: `ALLOWED_ORIGINS`, `PAPERMIND_MAX_UPLOAD_MB`, `PAPERMIND_REGENERATE_ON_STARTUP`, plus the frontend's `VITE_API_URL`.

---

# Part 5 — Product & Legal Minimum (§5)

## 30. Observability — Sentry + PostHog (done)

§5's first slice is "can't run a product blind." Two services, one principle.

**The principle: fail-open, not fail-loud.** Auth/storage/billing (§1–§3) deliberately *raise* on missing config — you can't run the product without them. Observability is the opposite: it must be a complete **no-op when its keys are absent**, so local dev and CI run untouched. Every init here is guarded by the presence of its env var.

**Sentry (errors), both sides:**
- Backend (`api/main.py`): `sentry_sdk.init(...)` only when `SENTRY_DSN` is set, placed before the `FastAPI()` instance so the SDK's automatic FastAPI/Starlette integration wraps the app — unhandled route exceptions are captured with zero per-route code. `send_default_pii=False`, `traces_sample_rate=0.1`.
- Frontend (`frontend/src/main.jsx`): `Sentry.init(...)` only when `VITE_SENTRY_DSN` is set, and `<App/>` wrapped in a `<Sentry.ErrorBoundary>` with a minimal fallback.

**PostHog (product analytics), frontend-only** — by decision, since §2 already records server-truth (cost/tokens/quota) in Postgres, so a second server-side analytics path would just duplicate it:
- New `frontend/src/analytics.js` is a thin wrapper: `initAnalytics()` inits the `posthog-js` singleton only when `VITE_POSTHOG_KEY` is set, and `track`/`identify`/`resetAnalytics` are no-ops otherwise. Centralizing the guard keeps every call site a clean one-liner.
- `App.jsx` identifies the signed-in user pseudonymously by **Clerk ID only** (a `<PostHogIdentify/>` using `useUser()`), and `<PostHogReset/>` clears identity on sign-out so events aren't misattributed on a shared device.
- Four funnel events at existing handlers: `paper_uploaded` (UploadPage), `query_asked` (ChatPage, with a `compare` flag), `plan_viewed` + `upgrade_clicked` (BillingPage). Enough to analyze the upload→query→upgrade conversion from day one without over-instrumenting.

**Verification:** backend imports clean with no DSN (`is_initialized()` → False) and initializes without raising when a dummy DSN is set; frontend `eslint` + `vite build` pass with **no** analytics keys present, which is itself the proof that the guarded no-op path holds. Confirming events actually land in the Sentry/PostHog dashboards needs real keys and is a deploy-time step.

**Manual (yours, at deploy):** create a Sentry project (→ `SENTRY_DSN` on the HF Space + `VITE_SENTRY_DSN` on Vercel) and a PostHog project (→ `VITE_POSTHOG_KEY` + optional `VITE_POSTHOG_HOST` on Vercel; `frontend/.env.local` to test locally).

New files: `frontend/src/analytics.js`. Modified: `api/main.py` (guarded Sentry init), `requirements.txt` (`sentry-sdk[fastapi]`), `frontend/src/main.jsx` (Sentry + analytics init + ErrorBoundary), `frontend/src/App.jsx` (identify/reset), `frontend/src/pages/{UploadPage,ChatPage,BillingPage}.jsx` (funnel events), `frontend/package.json` (`@sentry/react`, `posthog-js`). New env vars: `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, and frontend `VITE_SENTRY_DSN`/`VITE_POSTHOG_KEY`/`VITE_POSTHOG_HOST`.

## 31. Landing page + legal pages (done)

The second §5 slice was the public front door and the legal minimum. Two product decisions were made with the user up front: signed-out visitors should hit a **landing page first** (CTA → Clerk), not the bare sign-in box; and preloaded sample papers should be a **shared, system-owned demo set** (read-only, quota-exempt, ingested once) rather than seeded per user — but that one was **deferred** (only its design was settled), so this slice shipped the landing page and the legal pages.

**Routing, without a router.** PaperMind has no `react-router` — navigation is plain page-state in `App.jsx`. Rather than pull one in for two static pages, the legal docs use a **hash route** layer (`useHashRoute()` + a `hashchange` listener): `#/terms` and `#/privacy` render a standalone `LegalPage` **regardless of auth state**, and short-circuit before the `SignedIn`/`SignedOut` split. Hash routing needs no Vercel rewrite config (it never hits the server) and the URLs are directly shareable — which matters because footers, and potentially Stripe/Clerk, need reachable legal links. The signed-out branch, previously just a centered `<SignIn/>`, now renders `LandingPage`; `SignIn` is no longer imported there because the landing page opens auth through Clerk's `SignInButton`/`SignUpButton mode="modal"` instead.

**The landing page** (`frontend/src/pages/LandingPage.jsx`) reuses UploadPage's exact visual language (the `#22d3ee` accent, dot-grid backdrop, glass cards) so it reads as the same product: hero with a QASPER-benchmark eyebrow and modal CTAs, a four-card feature row (exact-section citations / evidence-graded / multi-hop / benchmarked), a Free-vs-Pro pricing block that mirrors the real $9/mo split, and a footer linking the legal pages. The **demo GIF** is the one asset left to the owner: `DemoMedia` tries `${BASE_URL}demo.gif` and falls back to a labelled placeholder on error, so the page looks intentional whether or not the GIF exists yet — drop `frontend/public/demo.gif` and it appears with no code change. `upgrade_clicked` is fired from the landing CTAs so the existing PostHog funnel now starts one step earlier, at the marketing page.

**The legal docs** (`frontend/src/legal/content.js`, rendered through `react-markdown`) are written against the *real* architecture, not boilerplate. The **ToS** makes the upload-rights point the checklist calls out — §3 states the user represents they have the rights to every document they upload and that copyright responsibility is theirs — and also covers the AI-not-advice disclaimer, the Free/Pro/Stripe billing terms, acceptable use, and liability. The **Privacy Policy** discloses every processor in a table (Clerk, Neon, R2, Stripe, the four LLM providers, Sentry, PostHog, hosting) and, importantly, states plainly that **document text and queries are sent to third-party LLM providers** to generate answers. Three tokens — `[[OPERATOR]]`, `[[JURISDICTION]]`, `[[CONTACT_EMAIL]]` — are left as greppable placeholders so the docs can't silently ship with the wrong entity on them.

**Verification:** `eslint src` clean and `vite build` succeeds (only the pre-existing bundle-size advisory, nothing new). Live confirmation — the GIF rendering, the Clerk modals opening, the funnel events landing — is a deploy-time step. **Manual before live:** fill the three legal placeholders + get the docs reviewed, and add `frontend/public/demo.gif`.

New files: `frontend/src/pages/LandingPage.jsx`, `frontend/src/pages/LegalPage.jsx`, `frontend/src/legal/content.js`. Modified: `frontend/src/App.jsx` (hash routing + landing in the signed-out branch), `frontend/src/pages/UploadPage.jsx` (Terms/Privacy footer links).

## 32. Still to come in §5
**Preloaded sample papers** — design settled (shared system-owned, quota-exempt, ingest-once demo set; see §31), implementation deferred to a follow-up. That's the last open §5 item.

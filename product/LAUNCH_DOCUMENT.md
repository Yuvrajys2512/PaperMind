# PaperMind — Multi-Tenancy Build Log (Launch Checklist §1)

This is a narrative record of how PaperMind went from a single-user demo to a properly multi-tenant app — what we built, in what order, why each decision was made, and how we verified it actually worked. It covers exactly the work behind the three checked boxes in `LAUNCH_CHECKLIST.md` §1. Read this top to bottom and you should understand the whole thing without having to reconstruct it from diffs.

---

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

## 5. Tech stack — summary

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

## 7. What's done, what isn't

**Done and verified** (§1 of `LAUNCH_CHECKLIST.md`, in full): auth, per-user scoping, Postgres + R2 storage. A real upload → background ingest → question-answer → PDF-preview cycle has been run end-to-end against the live Clerk, Neon, and R2 services, with no errors in the backend log.

**Not started**: §2 (LLM economics — paid keys, per-request cost tracking, per-user quotas), §3 (Stripe billing), §4 (deployment hardening — most of the mechanics are already written up in `DEPLOYMENT.md`), §5 (Sentry/PostHog, landing page, ToS/Privacy Policy).

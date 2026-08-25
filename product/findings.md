# PaperMind — Pre-Launch Findings (Docker Smoke Test + Code Audit)

_Date: 2026-06-19 — audit of the §1–§5 "code-complete" claims in [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md) plus a local Docker smoke test._

> ⚠️ **Superseded in part (2026-08-25).** A fuller audit is in [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md).
> Two claims below are now **stale**: Finding #1 (Stripe vars crash at import) was **fixed** — `api/billing.py`
> soft-disables billing instead of raising; and the praise for the webhook "ACKing every verified event" is
> **wrong** — it also ACKs events whose handler crashed, which can silently lose a payment (checklist item 2.1).
> Findings #2, #3 and #4 are still open and carried forward as checklist items 1.5, 2.3, 1.6 / 4.x.

**Bottom line:** the engineering is done and the deploy artifact provably works (built and ran it locally). Nothing on the critical path needs more feature code. What's left is manual ops gated by external approvals (Stripe live-mode verification, key rotation, HF + Vercel deploy, legal review) — **days to "live and taking payments," not weeks.**

---

## ✅ Smoke test — PASSED

- **Image builds clean** — `docker build -t papermind:smoketest .`, exit 0, ~6 min, **9.92 GB** (torch + the two baked models). Within HF Spaces' 16 GB tier; far too big for any 512 MB free host, exactly as the checklist warned.
- **Container boots and serves `/health` → `{"status":"ok"}`** with no tracebacks. Logs show `Application startup complete`.
- **Caveat:** boot required injecting **dummy Stripe values** — which surfaced Finding #1 below.

Repro:
```bash
docker build -t papermind:smoketest .
docker run -d --name papermind_smoke -p 7860:7860 --env-file .env \
  -e STRIPE_SECRET_KEY=sk_test_dummy -e STRIPE_PRICE_ID=price_dummy \
  -e STRIPE_WEBHOOK_SECRET=whsec_dummy -e PAPERMIND_REGENERATE_ON_STARTUP=0 \
  papermind:smoketest
curl localhost:7860/health   # -> {"status":"ok"}
```

---

## 🔍 Code audit

### Verified genuinely done (the "code-complete" claims hold up)
- **Per-user data scoping** (`api/storage.py`) — every read goes through `get_readable_paper` (owner *or* demo), every write/delete through `get_owned_paper`. The demo set is genuinely read-only and quota-exempt by construction (`list_papers` excludes `DEMO_USER_ID`).
- **Quotas + entitlement** (`api/usage.py`) — `enforce_paper_quota` / `enforce_query_quota` wired as deps on `/upload`, `/query`, `/query/stream`. `pro` = unlimited via `TIER_LIMITS`; unknown tier fails open to unlimited (matches the comment).
- **Stripe webhook** (`api/billing.py`) — verifies the signature over the raw body, flips `users.tier` on `checkout.session.completed` / `subscription.updated` / `subscription.deleted`, ACKs every verified event so Stripe stops retrying.
- **Auth** (`api/auth.py`) — proper Clerk JWKS verification with key-rotation refresh.
- **Secret hygiene** — `.env` is gitignored and **not** committed. No leaked secrets in the repo.
- **Frontend wiring** — `VITE_API_URL`, `VITE_CLERK_PUBLISHABLE_KEY`, Sentry/PostHog all read from `import.meta.env`; demo.gif loads from `BASE_URL`.

### Issues, ranked by how much they'll bite

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | **Backend won't boot without all 3 Stripe vars.** `api/billing.py:40-43` raises `RuntimeError` at *import* time; they're empty in `.env`. `main.py` imports the billing router, so on HF Spaces missing Stripe secrets = the whole API 500s on startup. | 🔴 Deploy-blocker | Set `STRIPE_SECRET_KEY` / `STRIPE_PRICE_ID` / `STRIPE_WEBHOOK_SECRET` before first boot (needed for billing anyway). |
| 2 | **Unpinned dependencies** (`requirements.txt` is all `>=`, no upper bounds). Today's build ≠ next month's build; a breaking major could fail the build or silently change behavior. | 🟠 "Breaks later" | `pip freeze` the working set into a pinned lockfile and build from that. |
| 3 | **Stripe `current_period_end` may be null** — `_period_end_from` reads `subscription["current_period_end"]`, which newer Stripe API versions moved off the subscription object onto the items. | 🟡 Cosmetic | Non-blocking (not used for entitlement, which is `status`-based); fix if you want the column populated. |
| 4 | **Known manual items still pending** — legal placeholders unfilled (6× `[[OPERATOR]]` / `[[JURISDICTION]]` / `[[CONTACT_EMAIL]]` in `frontend/src/legal/content.js`); `SENTRY_DSN`, Stripe vars, and `PAPERMIND_DEMO_USER_ID` empty/missing in `.env`. | 🟡 Known manual | Already on the manual launch list. |

---

## `.env` key presence (values not shown)

| Key | Status |
|-----|--------|
| DATABASE_URL | SET |
| R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME | SET |
| CLERK_ISSUER | SET |
| GROQ_API_KEY / GEMINI_API_KEY | SET |
| STRIPE_SECRET_KEY / STRIPE_PRICE_ID / STRIPE_WEBHOOK_SECRET | **EMPTY** |
| SENTRY_DSN | EMPTY |
| PAPERMIND_DEMO_USER_ID | MISSING (defaults to `__papermind_demo__`) |

---

## How much work is left

**Engineering is done and verified.** Remaining:

- **~1 hr of recommended code fixes** — pin dependencies (#2); confirm/handle the Stripe-boot coupling (#1).
- **Manual ops** — key rotation, backend → HF Spaces, frontend → Vercel + domain, Stripe live-mode verification, Clerk prod instance, Sentry/PostHog projects, legal review. Gated by external approvals, not code.

**Biggest unknown:** Stripe live-mode business verification (legal entity / bank / tax), which is outside our control and can take hours-to-days.

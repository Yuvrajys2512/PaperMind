# PaperMind — LLM / API Layer

_Last updated: 2026-06-20_

This doc explains the LLM aspect of PaperMind: how many providers there are, how
the chain works, each provider's limits, and what that means for the customer
free tier and aggregate capacity.

There are **two completely different "limits" in play** — keep them separate:

- **Layer 1 — upstream LLM suppliers** (Groq, Gemini, Mistral, Cerebras): the keys
  *we* run on, with *their* rate limits. This is our supply.
- **Layer 2 — customer-facing free tier**: the per-user caps *we* impose in
  `api/usage.py`. This is what each customer gets.

---

## Layer 1 — The provider chain (`ingestion/llm_client.py`)

A single `chat_completion()` function fronts **5 provider slots** as a
**rotation-with-fallback** chain: try #1, and on a rate-limit / quota / 404 error
fall through to #2, #3… instantly. Every client is constructed with
`max_retries=0` so the OpenAI SDK doesn't waste ~2s retrying a 429 — *we* own the
fallback loop.

Priority order (fastest / most reliable first):

| # | Provider | Model | Speed | Free-tier limit (supplier's cap on us) |
|---|----------|-------|-------|----------------------------------------|
| 1 | **Groq-1** | llama-3.3-70b-versatile | ~150ms (LPU) | ~100k tokens/day, per-minute RPM/TPM |
| 2 | **Groq-2** | llama-3.3-70b-versatile | ~150ms | second key purely for overflow |
| 3 | **Gemini** | gemini-2.5-flash-lite | ~1s | free daily quota (real, unlike 2.0-flash which 429s at limit:0) |
| 4 | **Mistral** | mistral-small-latest | reliable | ~1B tokens/month |
| 5 | **Cerebras** | gpt-oss-120b | last resort | returns null content intermittently → kept last |

Each provider is only added to the chain if its API key is present in `.env`
(`GROQ_API_KEY`, `GROQ_API_KEY_2`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
`CEREBRAS_API_KEY`). If none are set, the module raises at import.

### How the fallback behaves

- A **per-minute** rate limit (Groq's "try again in 2s") is treated as
  *retryable*. If a whole pass exhausts every provider AND at least one was a
  per-minute limit, the client sleeps on an escalating backoff `(5s, 15s, 30s)`
  and retries the whole chain.
- A **daily-quota / auth / 404** error is *not* retryable — it skips that
  provider and fails fast (waiting can't help; a daily quota resets in 24h).
- **Null content** (Cerebras' intermittent failure mode) → skip to next provider.
- An unexpected error (network, bad auth not matching skip keywords) is re-raised
  immediately rather than silently swallowed.

### Pinning (experiments)

`chat_completion(..., pin=(provider_name, model))` forces a call onto one
provider+model with **no fallback**, so the model is held fixed. Used by the eval
harness (e.g. weak-vs-strong generator) where the producing model must be
controlled.

### Cost tracking

Every successful call records `tokens_in/out` and a projected USD cost via the
`_PRICING` table. Because we run on free keys today, this is a **projection** —
"what this *would* cost on paid keys" — exactly the input needed to price a paid
tier.

Per-1M-token list prices baked in (verified Jun 2026):

| Provider / model | input | output |
|---|---|---|
| Groq llama-3.3-70b | $0.59 | $0.79 |
| Gemini 2.5-flash-lite | $0.10 | $0.40 |
| Mistral small | $0.20 | $0.60 |
| Cerebras gpt-oss-120b | $0.35 | $0.75 |

Unpriced `(provider, model)` pairs cost $0 rather than crashing — a newly added
model just under-counts until priced. Pinned by `tests/test_llm_cost.py`.

Stats are **per-thread** (`threading.local`) because FastAPI runs the sync
pipeline in a thread-pool executor; the pipeline calls `reset_stats()` at the
start of a request and `get_stats()` at the end to surface
`llm_calls / providers_used / cost` on the response.

---

## Layer 2 — What customers get (`api/usage.py`)

Independent of the providers above. Each **user** on the free tier gets:

```
free tier:
  max_papers            = 3      (env: PAPERMIND_FREE_MAX_PAPERS)
  max_queries_per_month = 20     (env: PAPERMIND_FREE_MAX_QUERIES_PER_MONTH)
```

- Enforced by two FastAPI dependencies: `enforce_paper_quota` (429s on a 4th
  upload) and `enforce_query_quota` (429s past 20 queries/month, counted from
  `date_trunc('month')`).
- Both limits are **env-overridable** — retune without a code change.
- **Pro tier = unlimited** (`TIER_LIMITS["pro"] = None`). Stripe's webhook
  (`api/billing.py`) flips `users.tier` to `pro` on an active subscription; the
  quota code reads `users.tier` directly, so there's no separate entitlement
  gate. Pro is **$9/mo**.
- **Fails open:** any unknown/unmapped tier rides the `None` (unlimited) path, so
  a paying user is never wrongly locked out.

Usage is logged to the `usage_events` table (`kind`, `llm_calls`, `tokens_in/out`,
`cost_usd`). `record_usage` never raises — a logging failure must not take down
the request it describes.

---

## The connection between the layers (the thing to watch)

The per-user caps exist primarily to **protect the shared upstream quota**. All
users draw from the *same* Groq → Gemini → Mistral → Cerebras pool, and each
query fires **multiple** LLM calls (guard, planner, HyDE, multi-hop, generation).

So "how much do customers get" has two answers:

1. **Per customer (the contract):** 3 papers + 20 queries/month free; unlimited at
   $9/mo.
2. **In aggregate (the physics):** bounded by the free upstream quotas below.

---

## Aggregate capacity estimate

> **Data caveat:** `usage_events` currently holds only seed rows — **zero real
> query events** — so a direct from-the-table estimate isn't yet possible. The
> numbers below are derived from QASPER eval runs (same pipeline) for the
> measured call count, plus a token model for per-call size. They firm up once
> real queries land in `usage_events`.

> **Live aggregates:** `GET /admin/usage` returns these numbers computed straight
> from `usage_events` — users by tier, per-query call/token/cost averages, and
> the capacity projection. It's gated to the Clerk user ids in
> `PAPERMIND_ADMIN_USER_IDS` (empty by default → locked to nobody). The
> `capacity.tokens_source` field reports `"measured"` once real query events
> exist, or `"modeled"` (~6k tokens/query) until then. The one-off
> `scripts/manual_checks/usage_capacity.py` computes the same thing from the CLI.

**LLM calls per query — measured across 68 eval queries:** mean **4.4**, median
**5**, p90 **5**, max **5**. Matches the pipeline stages (guard `max_tokens=5` →
planner 400 → HyDE 180 → multi-hop 400 → generator 2048, fed `llm_k` reranked
chunks of 512 tokens; `llm_k` default 5, range 3–7).

**Tokens per query — modeled** (eval didn't log tokens):

| Call group | input | output |
|---|---|---|
| guard + planner + HyDE + multi-hop | ~1,200 | ~1,000 |
| generator (5 chunks × 512 + prompt) | ~3,200 | ~600 |
| **per query total** | | **≈ 6,000 tokens** (range 5k–8k) |

**Capacity, by binding upstream quota (@ ~6k tokens/query):**

| Provider | Free quota | Queries | Role |
|---|---|---|---|
| **Groq ×2** | 200k tok/day | **~33/day** (~1k/mo) | fast path, exhausts first |
| **Gemini** flash-lite | ~1k req/day (~4 calls/query) | **~250/day** (~7.5k/mo) | second wave |
| **Mistral** small | 1B tok/month | **~165,000/month** | the real backstop |

**Headline:** Groq's fast LPU path is *tiny* in token terms (~16 queries/day/key),
so most traffic spends on Gemini → Mistral. **Mistral's 1B/month carries the
system: ~150k–200k queries/month aggregate** before the free stack runs dry.

**Translated to free users (20 queries/user/month):**

- **Groq alone:** ~50 fully-active free users
- **Whole free stack (Mistral-bound):** **~8,000+ free users/month** in theory

---

## Takeaways

- The free tier is safe to offer: the 20-query/user cap plus graceful
  Groq → Gemini → Mistral degradation means a user spike *slows* responses (falls
  off the LPU) rather than erroring.
- The scarce resource is the **fast-path budget (Groq)**, not total capacity. To
  keep most queries sub-second, paid Groq is the first upgrade — `_PRICING`
  projects the cost (~6k tok/query ≈ **$0.004/query** at Groq 70B list rates).
- Estimates rest on a measured call count (4.4) but a *modeled* token figure; they
  sharpen the moment real queries populate `usage_events`.

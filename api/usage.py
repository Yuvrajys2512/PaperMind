"""
api/usage.py

Per-user tier lookup, usage logging, and quota enforcement.

Public API
----------
enforce_paper_quota(user_id) -> str   FastAPI dependency; 429s over the free cap
enforce_query_quota(user_id) -> str   FastAPI dependency; 429s over the free cap
enforce_audit_quota(user_id) -> str   FastAPI dependency; 429s over the free cap
                                       (deep paper analyses: claim audit + review)
record_usage(...)                     log one LLM-costing event (query/audit/upload)
get_usage_summary(user_id) -> dict    tier + current usage vs. limits
get_aggregate_usage() -> dict         all-user totals + capacity projection (admin)
"""

import os

from fastapi import Depends, HTTPException

from api.auth import get_current_user_id
from api.storage import _pool, list_papers

# Free upstream quotas (tokens), used only for the capacity projection in
# get_aggregate_usage. Mirrors ingestion/llm_client.py's provider chain and the
# numbers in product/llm_api.md. Env-overridable so the projection tracks reality
# if a paid key lifts a ceiling.
_GROQ_TOKENS_PER_DAY = int(os.getenv("PAPERMIND_GROQ_TOKENS_PER_DAY", "100000"))
_GROQ_KEYS = int(os.getenv("PAPERMIND_GROQ_KEYS", "2"))
_MISTRAL_TOKENS_PER_MONTH = int(os.getenv("PAPERMIND_MISTRAL_TOKENS_PER_MONTH", "1000000000"))
# Fallback per-query token estimate used until real query events accumulate.
# Derived from the eval-based model in product/llm_api.md (~6k tokens/query).
_MODELED_TOKENS_PER_QUERY = int(os.getenv("PAPERMIND_MODELED_TOKENS_PER_QUERY", "6000"))

TIER_LIMITS = {
    "free": {
        "max_papers": int(os.getenv("PAPERMIND_FREE_MAX_PAPERS", "3")),
        "max_queries_per_month": int(os.getenv("PAPERMIND_FREE_MAX_QUERIES_PER_MONTH", "20")),
        # Deep paper analyses (claim audit + weakness review) share a dedicated
        # monthly cap so they never compete with the Q&A query budget. Each is
        # the heaviest LLM operation in the app, but results are R2-cached — only
        # the first run and forced re-runs count.
        "max_audits_per_month": int(os.getenv("PAPERMIND_FREE_MAX_AUDITS_PER_MONTH", "10")),
    },
    # "pro" (set by §3/Stripe's webhook) is explicitly unlimited. `None` rides
    # the same code path as an unknown tier in _limits_for, so a `users.tier`
    # value that Stripe flips before this dict is ever extended still fails
    # open (unlimited) rather than wrongly locking out a paying user.
    "pro": None,
}


def _ensure_schema():
    with _pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                tier TEXT NOT NULL DEFAULT 'free',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                req_id TEXT,
                paper_id TEXT,
                llm_calls INT NOT NULL DEFAULT 0,
                tokens_in INT NOT NULL DEFAULT 0,
                tokens_out INT NOT NULL DEFAULT 0,
                cost_usd NUMERIC(10,6) NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_user_created "
            "ON usage_events (user_id, created_at)"
        )


_ensure_schema()


def _limits_for(tier: str) -> dict | None:
    """None means unlimited (unknown/paid tier not yet in TIER_LIMITS)."""
    return TIER_LIMITS.get(tier)


def get_user_tier(user_id: str) -> str:
    """Lazily creates the user's row (default tier 'free') and returns the tier."""
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING",
            (user_id,),
        )
        cur = conn.execute("SELECT tier FROM users WHERE user_id = %s", (user_id,))
        return cur.fetchone()[0]


def set_user_tier(user_id: str, tier: str) -> None:
    """Upserts the user's row and sets their tier. This module owns the `users`
    table, so billing (§3) flips tiers through here rather than touching it."""
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, tier) VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET tier = EXCLUDED.tier
            """,
            (user_id, tier),
        )


def _count_events_this_month(user_id: str, kind: str) -> int:
    with _pool.connection() as conn:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM usage_events
            WHERE user_id = %s AND kind = %s
              AND created_at >= date_trunc('month', now())
            """,
            (user_id, kind),
        )
        return cur.fetchone()[0]


def count_queries_this_month(user_id: str) -> int:
    return _count_events_this_month(user_id, "query")


def count_audits_this_month(user_id: str) -> int:
    """Deep paper analyses (claim audit + weakness review) logged as kind='audit'."""
    return _count_events_this_month(user_id, "audit")


def record_usage(
    user_id: str,
    kind: str,
    req_id: str = None,
    paper_id: str = None,
    llm_calls: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Logs one LLM-costing event. Never raises — a logging failure must not
    take down the request it's describing (same posture as api/logger.py)."""
    try:
        with _pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_events
                    (user_id, kind, req_id, paper_id, llm_calls, tokens_in, tokens_out, cost_usd)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, kind, req_id, paper_id, llm_calls, tokens_in, tokens_out, cost_usd),
            )
    except Exception as exc:
        print(f"[usage] failed to record usage event: {exc}")


def enforce_paper_quota(user_id: str = Depends(get_current_user_id)) -> str:
    limits = _limits_for(get_user_tier(user_id))
    if limits is None:
        return user_id

    # list_papers(user_id) deliberately counts BOTH reference papers and drafts
    # (one shared pool) and excludes the demo set. That is correct but invisible:
    # the user sees a Library of their papers *plus* the samples, and their
    # drafts live on a different page entirely — so "limit of 3 reached" while
    # the sidebar reads 5 looks like a bug. The message therefore spells out
    # both halves rather than leaving the user to guess which items count.
    own = list_papers(user_id)
    if len(own) >= limits["max_papers"]:
        drafts = sum(1 for p in own if p.get("paper_type") == "draft")
        papers = len(own) - drafts

        def _n(count: int, word: str) -> str:
            return f"{count} {word}" if count == 1 else f"{count} {word}s"

        breakdown = _n(papers, "paper")
        if drafts:
            breakdown += f" and {_n(drafts, 'draft')}"
        raise HTTPException(
            status_code=429,
            detail=(
                f"Free tier limit of {limits['max_papers']} reached — you have {breakdown}. "
                "Drafts count toward the same limit; the shared sample papers do not. "
                "Delete a paper or a draft to upload a new one."
            ),
        )
    return user_id


def enforce_query_quota(user_id: str = Depends(get_current_user_id)) -> str:
    limits = _limits_for(get_user_tier(user_id))
    if limits is not None and count_queries_this_month(user_id) >= limits["max_queries_per_month"]:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Free tier limit of {limits['max_queries_per_month']} queries/month reached. "
                "Try again next month."
            ),
        )
    return user_id


def enforce_audit_quota(user_id: str = Depends(get_current_user_id)) -> str:
    """Gate for the deep paper analyses (claim audit + weakness review). Separate
    from the Q&A query budget so a user running audits never burns query credit."""
    limits = _limits_for(get_user_tier(user_id))
    if limits is not None and count_audits_this_month(user_id) >= limits["max_audits_per_month"]:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Free tier limit of {limits['max_audits_per_month']} paper analyses/month "
                "reached (claim audits and weakness reviews share this limit). "
                "Try again next month."
            ),
        )
    return user_id


def get_usage_summary(user_id: str) -> dict:
    tier = get_user_tier(user_id)
    limits = _limits_for(tier) or {}
    own = list_papers(user_id)
    drafts_used = sum(1 for p in own if p.get("paper_type") == "draft")
    return {
        "tier": tier,
        # papers_used is the quota number: the user's own papers AND drafts,
        # excluding the shared demo set. `drafts_used` breaks out the part the
        # Library page never shows, so the UI can explain the total instead of
        # contradicting it.
        "papers_used": len(own),
        "drafts_used": drafts_used,
        "papers_limit": limits.get("max_papers"),
        "queries_used": count_queries_this_month(user_id),
        "queries_limit": limits.get("max_queries_per_month"),
        "audits_used": count_audits_this_month(user_id),
        "audits_limit": limits.get("max_audits_per_month"),
    }


def get_aggregate_usage() -> dict:
    """All-user aggregates plus an upstream-capacity projection. Admin-only.

    Reports real per-query stats from usage_events once they exist, and projects
    how many queries the free provider quotas can sustain. While usage_events has
    no query rows yet, the projection falls back to the modeled ~6k tokens/query
    (see product/llm_api.md) and flags `tokens_source` accordingly."""
    with _pool.connection() as conn:
        users_by_tier = dict(
            conn.execute("SELECT tier, COUNT(*) FROM users GROUP BY tier").fetchall()
        )
        events_by_kind = dict(
            conn.execute("SELECT kind, COUNT(*) FROM usage_events GROUP BY kind").fetchall()
        )
        q = conn.execute(
            """
            SELECT
                COUNT(*)                                                          AS n,
                AVG(llm_calls)                                                    AS avg_calls,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY llm_calls)            AS p90_calls,
                AVG(tokens_in + tokens_out)                                       AS avg_tokens,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tokens_in+tokens_out) AS p90_tokens,
                AVG(cost_usd)                                                     AS avg_cost,
                SUM(tokens_in + tokens_out)                                       AS sum_tokens,
                SUM(cost_usd)                                                     AS sum_cost,
                COUNT(DISTINCT user_id)                                           AS distinct_users
            FROM usage_events
            WHERE kind = 'query'
            """
        ).fetchone()

    n = q[0] or 0
    avg_calls, p90_calls, avg_tokens, p90_tokens, avg_cost, sum_tokens, sum_cost, distinct_q_users = (
        float(q[1]) if q[1] is not None else 0.0,
        float(q[2]) if q[2] is not None else 0.0,
        float(q[3]) if q[3] is not None else 0.0,
        float(q[4]) if q[4] is not None else 0.0,
        float(q[5]) if q[5] is not None else 0.0,
        int(q[6]) if q[6] is not None else 0,
        float(q[7]) if q[7] is not None else 0.0,
        int(q[8]) if q[8] is not None else 0,
    )

    # Use measured avg tokens/query if we have real data; else the modeled value.
    tokens_per_query = avg_tokens if (n and avg_tokens) else float(_MODELED_TOKENS_PER_QUERY)
    tokens_source = "measured" if (n and avg_tokens) else "modeled"

    groq_daily = (_GROQ_TOKENS_PER_DAY * _GROQ_KEYS) / tokens_per_query
    mistral_monthly = _MISTRAL_TOKENS_PER_MONTH / tokens_per_query
    free_cap = TIER_LIMITS["free"]["max_queries_per_month"]

    return {
        "users": {"total": sum(users_by_tier.values()), "by_tier": users_by_tier},
        "events_by_kind": events_by_kind,
        "per_query": {
            "n": n,
            "avg_llm_calls": round(avg_calls, 2),
            "p90_llm_calls": round(p90_calls, 1),
            "avg_tokens": round(avg_tokens),
            "p90_tokens": round(p90_tokens),
            "avg_cost_usd": round(avg_cost, 6),
            "distinct_users": distinct_q_users,
        },
        "lifetime": {
            "query_tokens": sum_tokens,
            "projected_cost_usd": round(sum_cost, 4),
        },
        "capacity": {
            "tokens_per_query": round(tokens_per_query),
            "tokens_source": tokens_source,
            "groq_queries_per_day": round(groq_daily),
            "groq_queries_per_month": round(groq_daily * 30),
            "mistral_queries_per_month": round(mistral_monthly),
            "free_users_supportable": round(mistral_monthly / free_cap) if free_cap else None,
        },
    }

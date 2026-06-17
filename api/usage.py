"""
api/usage.py

Per-user tier lookup, usage logging, and quota enforcement.

Public API
----------
enforce_paper_quota(user_id) -> str   FastAPI dependency; 429s over the free cap
enforce_query_quota(user_id) -> str   FastAPI dependency; 429s over the free cap
record_usage(...)                     log one LLM-costing event (query or upload)
get_usage_summary(user_id) -> dict    tier + current usage vs. limits
"""

import os

from fastapi import Depends, HTTPException

from api.auth import get_current_user_id
from api.storage import _pool, list_papers

TIER_LIMITS = {
    "free": {
        "max_papers": int(os.getenv("PAPERMIND_FREE_MAX_PAPERS", "3")),
        "max_queries_per_month": int(os.getenv("PAPERMIND_FREE_MAX_QUERIES_PER_MONTH", "20")),
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


def count_queries_this_month(user_id: str) -> int:
    with _pool.connection() as conn:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM usage_events
            WHERE user_id = %s AND kind = 'query'
              AND created_at >= date_trunc('month', now())
            """,
            (user_id,),
        )
        return cur.fetchone()[0]


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
    if limits is not None and len(list_papers(user_id)) >= limits["max_papers"]:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Free tier limit of {limits['max_papers']} papers reached. "
                "Delete a paper to upload a new one."
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


def get_usage_summary(user_id: str) -> dict:
    tier = get_user_tier(user_id)
    limits = _limits_for(tier) or {}
    return {
        "tier": tier,
        "papers_used": len(list_papers(user_id)),
        "papers_limit": limits.get("max_papers"),
        "queries_used": count_queries_this_month(user_id),
        "queries_limit": limits.get("max_queries_per_month"),
    }

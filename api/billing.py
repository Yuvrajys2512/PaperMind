"""
api/billing.py

Stripe billing: Checkout, the customer portal, and the webhook that flips a
user between the 'free' and 'pro' tiers.

Tier is the entitlement. The quota dependencies in api/usage.py already read
`users.tier` — a 'pro' user has no caps (see TIER_LIMITS). So §3 only has to
move users between tiers via Stripe; there is no separate entitlement gate.

This module owns the `subscriptions` table (user_id ↔ Stripe customer/sub).
It flips `users.tier` through usage.set_user_tier rather than writing `users`
directly, since usage.py owns that table.

Public API
----------
router                          APIRouter mounted under /billing in api/main.py
  POST /billing/checkout        authed; returns a Stripe Checkout URL
  POST /billing/portal          authed; returns a Stripe customer-portal URL
  POST /billing/webhook         Stripe-signed; flips tiers on subscription events
"""

import os
from datetime import datetime, timezone

import stripe
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import get_current_user_id
from api.storage import _pool
from api.usage import set_user_tier
from api.logger import log_operation

load_dotenv()

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Billing is guarded by its keys, exactly like the observability integrations
# (Sentry/PostHog). With all three Stripe secrets present it runs fully; with any
# missing it stays dormant — the app still boots and every /billing route returns
# 503 "billing not configured" instead of crashing at import. This keeps local
# dev and pre-Stripe deploys runnable. Unlike auth/storage (which fail loud),
# billing is optional at launch.
BILLING_ENABLED = all([STRIPE_SECRET_KEY, STRIPE_PRICE_ID, STRIPE_WEBHOOK_SECRET])

# Where Stripe sends the user back after Checkout / the portal.
FRONTEND_URL = os.getenv("PAPERMIND_FRONTEND_URL", "http://localhost:5173").rstrip("/")

if BILLING_ENABLED:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    print("[billing] Stripe keys not set — billing disabled; /billing routes return 503.")

# Subscription statuses that should grant the 'pro' tier. Anything else
# (canceled, unpaid, incomplete_expired, …) drops the user back to 'free'.
_ACTIVE_STATUSES = {"active", "trialing"}

router = APIRouter(prefix="/billing", tags=["billing"])


def require_billing():
    """Dependency: 503 every billing route when Stripe isn't configured, rather
    than crashing the whole app at import when keys are absent."""
    if not BILLING_ENABLED:
        raise HTTPException(status_code=503, detail="Billing is not configured on this server.")


def _ensure_schema():
    with _pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id                TEXT PRIMARY KEY,
                stripe_customer_id     TEXT UNIQUE,
                stripe_subscription_id TEXT,
                status                 TEXT,
                current_period_end     TIMESTAMPTZ,
                updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscriptions_customer "
            "ON subscriptions (stripe_customer_id)"
        )


if BILLING_ENABLED:
    _ensure_schema()


# ── Mapping: user_id ↔ Stripe customer ───────────────────────────────────────

def _get_customer_id(user_id: str) -> str | None:
    with _pool.connection() as conn:
        cur = conn.execute(
            "SELECT stripe_customer_id FROM subscriptions WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _user_id_for_customer(customer_id: str) -> str | None:
    with _pool.connection() as conn:
        cur = conn.execute(
            "SELECT user_id FROM subscriptions WHERE stripe_customer_id = %s",
            (customer_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _get_or_create_customer(user_id: str) -> str:
    """Returns the user's Stripe customer id, creating (and recording) one on
    first use so a returning user never spawns a second Stripe customer."""
    existing = _get_customer_id(user_id)
    if existing:
        return existing
    customer = stripe.Customer.create(metadata={"user_id": user_id})
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO subscriptions (user_id, stripe_customer_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET stripe_customer_id = EXCLUDED.stripe_customer_id
            """,
            (user_id, customer["id"]),
        )
    return customer["id"]


def _record_subscription(
    user_id: str,
    customer_id: str,
    subscription_id: str | None,
    status: str | None,
    period_end: datetime | None,
) -> None:
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO subscriptions
                (user_id, stripe_customer_id, stripe_subscription_id, status,
                 current_period_end, updated_at)
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                stripe_customer_id     = EXCLUDED.stripe_customer_id,
                stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                status                 = EXCLUDED.status,
                current_period_end     = EXCLUDED.current_period_end,
                updated_at             = now()
            """,
            (user_id, customer_id, subscription_id, status, period_end),
        )


def _period_end_from(subscription: dict) -> datetime | None:
    ts = subscription.get("current_period_end")
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/checkout")
def create_checkout(user_id: str = Depends(get_current_user_id), _=Depends(require_billing)):
    try:
        customer_id = _get_or_create_customer(user_id)
    except Exception as exc:
        log_operation(
            "create_stripe_customer",
            "error",
            user_id=user_id,
            error=exc,
        )
        raise HTTPException(status_code=502, detail="Could not create billing account.")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/?billing=success",
            cancel_url=f"{FRONTEND_URL}/?billing=cancel",
            client_reference_id=user_id,
            subscription_data={"metadata": {"user_id": user_id}},
        )
        log_operation(
            "create_checkout_session",
            "success",
            user_id=user_id,
            context={"customer_id": customer_id},
        )
    except Exception as exc:
        log_operation(
            "create_checkout_session",
            "error",
            user_id=user_id,
            error=exc,
            context={"customer_id": customer_id},
        )
        raise HTTPException(status_code=502, detail="Could not start checkout.")
    return {"url": session["url"]}


@router.post("/portal")
def create_portal(user_id: str = Depends(get_current_user_id), _=Depends(require_billing)):
    customer_id = _get_customer_id(user_id)
    if not customer_id:
        raise HTTPException(status_code=400, detail="No billing account yet — subscribe first.")
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{FRONTEND_URL}/?billing=portal",
        )
        log_operation(
            "create_portal_session",
            "success",
            user_id=user_id,
            context={"customer_id": customer_id},
        )
    except Exception as exc:
        log_operation(
            "create_portal_session",
            "error",
            user_id=user_id,
            error=exc,
            context={"customer_id": customer_id},
        )
        raise HTTPException(status_code=502, detail="Could not open billing portal.")
    return {"url": session["url"]}


@router.post("/webhook")
async def stripe_webhook(request: Request, _=Depends(require_billing)):
    # No Clerk auth — Stripe calls this. Authenticity comes from the signature
    # over the raw request body, so we must use the bytes exactly as received.
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        log_operation(
            "stripe_webhook",
            "error",
            error=exc,
            context={"reason": "invalid_signature"},
        )
        raise HTTPException(status_code=400, detail="Invalid Stripe signature or payload.")

    event_type = event["type"]
    obj = event["data"]["object"]

    try:
        if event_type == "checkout.session.completed":
            # First subscription: the session carries our user_id and the new
            # customer/subscription ids.
            user_id = obj.get("client_reference_id")
            customer_id = obj.get("customer")
            if user_id and customer_id:
                _record_subscription(
                    user_id, customer_id, obj.get("subscription"), "active", None
                )
                set_user_tier(user_id, "pro")
                log_operation(
                    "stripe_checkout_completed",
                    "success",
                    user_id=user_id,
                    context={"customer_id": customer_id, "event_id": event.get("id")},
                )

        elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
            customer_id = obj.get("customer")
            # Prefer the user_id we stamped into subscription metadata; fall back to
            # our customer→user mapping for events that predate it.
            user_id = (obj.get("metadata") or {}).get("user_id") or _user_id_for_customer(customer_id)
            if user_id and customer_id:
                status = "canceled" if event_type.endswith("deleted") else obj.get("status")
                _record_subscription(
                    user_id, customer_id, obj.get("id"), status, _period_end_from(obj)
                )
                new_tier = "pro" if status in _ACTIVE_STATUSES else "free"
                set_user_tier(user_id, new_tier)
                log_operation(
                    "stripe_subscription_updated",
                    "success",
                    user_id=user_id,
                    context={
                        "event_type": event_type,
                        "subscription_status": status,
                        "tier": new_tier,
                        "event_id": event.get("id"),
                    },
                )
    except Exception as exc:
        log_operation(
            "stripe_webhook",
            "error",
            error=exc,
            context={"event_type": event_type, "event_id": event.get("id")},
        )

    # Acknowledge every verified event (handled or not) so Stripe stops retrying.
    return {"received": True}

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

# Pinned explicitly so a future `stripe` SDK upgrade can't silently move us
# onto a newer Stripe API version with a different response shape (this bit
# us once already: `current_period_end` moved off the Subscription object
# and onto its items in 2025-03-31+ — see `_period_end_from`).
STRIPE_API_VERSION = "2026-05-27.dahlia"

if BILLING_ENABLED:
    stripe.api_key = STRIPE_SECRET_KEY
    stripe.api_version = STRIPE_API_VERSION
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stripe_events (
                event_id    TEXT PRIMARY KEY,
                received_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


if BILLING_ENABLED:
    # Registered rather than run eagerly — the shared _pool (from api.storage)
    # is lazy, so this only actually runs on the first real connection, not
    # merely when api.billing is imported (Launch Checklist 2.12).
    _pool.on_first_open(_ensure_schema)


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
    event_created_at: datetime,
) -> None:
    """`updated_at` is stamped with the *Stripe event's* `created` time, not
    wall-clock `now()` — that's what makes it usable as an ordering guard
    (see `_current_subscription_updated_at`). Two events processed in the
    same second would otherwise both get `now()` and be indistinguishable."""
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO subscriptions
                (user_id, stripe_customer_id, stripe_subscription_id, status,
                 current_period_end, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                stripe_customer_id     = EXCLUDED.stripe_customer_id,
                stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                status                 = EXCLUDED.status,
                current_period_end     = EXCLUDED.current_period_end,
                updated_at             = EXCLUDED.updated_at
            """,
            (user_id, customer_id, subscription_id, status, period_end, event_created_at),
        )


def _period_end_from(subscription: dict) -> datetime | None:
    """As of Stripe API 2025-03-31+ (which STRIPE_API_VERSION pins us to),
    `current_period_end` no longer lives on the Subscription object — it
    moved onto each subscription item. Fall back to the old top-level field
    in case a webhook ever arrives from an account still on an older
    version."""
    items = ((subscription.get("items") or {}).get("data")) or []
    ts = items[0].get("current_period_end") if items else None
    if ts is None:
        ts = subscription.get("current_period_end")
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None


def _mark_event_processed(event_id: str) -> bool:
    """Returns True the first time event_id is seen (and records it), False
    on a duplicate delivery. Stripe retries deliveries it didn't get a 2xx
    for, and can also just send the same event twice — without this, a
    retried `checkout.session.completed` would re-run `set_user_tier`
    harmlessly, but a retried `customer.subscription.deleted` racing a newer
    `customer.subscription.updated` could flip a live subscriber back to
    `free`."""
    with _pool.connection() as conn:
        cur = conn.execute(
            "INSERT INTO stripe_events (event_id) VALUES (%s) ON CONFLICT DO NOTHING "
            "RETURNING event_id",
            (event_id,),
        )
        return cur.fetchone() is not None


def _current_subscription_updated_at(user_id: str) -> datetime | None:
    with _pool.connection() as conn:
        cur = conn.execute(
            "SELECT updated_at FROM subscriptions WHERE user_id = %s", (user_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None


def delete_customer_for_user(user_id: str) -> None:
    """Cancels any active Stripe subscription and forgets the local mapping for
    user_id. Called by the Clerk account-deletion webhook (api/main.py) so a
    deleted account can't keep being billed. No-op if billing is disabled or
    the user never subscribed."""
    if not BILLING_ENABLED:
        return
    customer_id = _get_customer_id(user_id)
    if not customer_id:
        return
    try:
        subs = stripe.Subscription.list(customer=customer_id, status="active", limit=10)
        for sub in subs.get("data", []):
            stripe.Subscription.cancel(sub["id"])
    except Exception as exc:
        # Best-effort: a Stripe outage must not block deleting the user's data
        # (the legal requirement). Logged loudly since an uncancelled
        # subscription means we keep charging someone who deleted their
        # account — needs manual follow-up if this ever fires.
        log_operation(
            "cancel_stripe_subscription_on_account_delete",
            "error",
            user_id=user_id,
            error=exc,
            context={"customer_id": customer_id},
        )
    with _pool.connection() as conn:
        conn.execute("DELETE FROM subscriptions WHERE user_id = %s", (user_id,))


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
    event_id = event.get("id")
    obj = event["data"]["object"]

    if event_id and not _mark_event_processed(event_id):
        log_operation(
            "stripe_webhook_duplicate",
            "success",
            context={"event_type": event_type, "event_id": event_id},
        )
        return {"received": True}

    event_created_at = datetime.fromtimestamp(event["created"], tz=timezone.utc)

    try:
        if event_type == "checkout.session.completed":
            # First subscription: the session carries our user_id and the new
            # customer/subscription ids.
            user_id = obj.get("client_reference_id")
            customer_id = obj.get("customer")
            if user_id and customer_id:
                _record_subscription(
                    user_id, customer_id, obj.get("subscription"), "active", None,
                    event_created_at,
                )
                set_user_tier(user_id, "pro")
                log_operation(
                    "stripe_checkout_completed",
                    "success",
                    user_id=user_id,
                    context={"customer_id": customer_id, "event_id": event_id},
                )

        elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
            customer_id = obj.get("customer")
            # Prefer the user_id we stamped into subscription metadata; fall back to
            # our customer→user mapping for events that predate it.
            user_id = (obj.get("metadata") or {}).get("user_id") or _user_id_for_customer(customer_id)
            if user_id and customer_id:
                # Ordering guard: subscription.updated/deleted events can arrive
                # out of order (Stripe doesn't guarantee delivery order). If we've
                # already applied a *later* event for this user, applying this
                # older one would flip the tier back to a stale value.
                last_applied = _current_subscription_updated_at(user_id)
                if last_applied and event_created_at <= last_applied:
                    log_operation(
                        "stripe_webhook_stale_event",
                        "success",
                        user_id=user_id,
                        context={
                            "event_type": event_type,
                            "event_id": event_id,
                            "event_created_at": event_created_at.isoformat(),
                            "last_applied_at": last_applied.isoformat(),
                        },
                    )
                    return {"received": True}

                status = "canceled" if event_type.endswith("deleted") else obj.get("status")
                _record_subscription(
                    user_id, customer_id, obj.get("id"), status, _period_end_from(obj),
                    event_created_at,
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
                        "event_id": event_id,
                    },
                )
    except Exception as exc:
        log_operation(
            "stripe_webhook",
            "error",
            error=exc,
            context={"event_type": event_type, "event_id": event.get("id")},
        )
        # A handler we implement raised — surface a 500 so Stripe retries the
        # delivery instead of believing it succeeded. Silently swallowing this
        # (as before) meant a transient Neon/DB failure could drop a payment:
        # the user's card is charged, `checkout.session.completed` is ACKed,
        # and `set_user_tier` never runs, so they stay on `free` forever.
        raise HTTPException(status_code=500, detail="Webhook handler failed.")

    # Acknowledge every verified event we don't implement a handler for, and
    # every one we do implement that ran to completion without raising.
    return {"received": True}

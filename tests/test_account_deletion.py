"""
Regression tests for the Clerk account-deletion cascade (Launch Checklist 1.8).

Why these tests are AST-based
------------------------------
Same reasoning as tests/test_stream_accounting.py: `api.main`, `api.usage` and
`api.billing` all connect to live Postgres/R2/Stripe at import time, and this
suite must never touch production (see 2.12 — it already happened once).
Parsing the source lets us pin structural properties (which tables get
deleted, whether the handler fails loud vs. swallows errors) without
importing anything.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "api" / "main.py"
USAGE = ROOT / "api" / "usage.py"
BILLING = ROOT / "api" / "billing.py"


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _called_names(node) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _string_constants(node) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


# ── api/main.py: POST /webhooks/clerk ────────────────────────────────────────

@pytest.fixture(scope="module")
def clerk_webhook_fn():
    fn = _find_function(_tree(MAIN), "clerk_webhook")
    assert fn is not None, "clerk_webhook handler is gone from api/main.py"
    return fn


def test_clerk_webhook_verifies_signature(clerk_webhook_fn):
    assert "verify" in _called_names(clerk_webhook_fn), (
        "clerk_webhook must verify the Svix signature (Webhook(...).verify(...)) "
        "before trusting the payload — otherwise anyone can POST a fake "
        "user.deleted event and wipe an arbitrary account"
    )


def test_clerk_webhook_does_not_swallow_deletion_errors(clerk_webhook_fn):
    """Unlike the Stripe webhook (2.1's whole point is that ACK-on-failure loses
    data silently), this handler must let an exception during the delete
    cascade propagate to a non-2xx so Svix retries — the alternative is
    silently keeping a deleted user's data forever."""
    for child in ast.walk(clerk_webhook_fn):
        if isinstance(child, ast.Try):
            # A try/except here would be the exact bug this test guards
            # against, *unless* it's scoped to the initial signature
            # verification (which legitimately turns a bad payload into a
            # 400, not a swallowed cascade failure).
            handled = _called_names(child)
            assert "verify" in handled or "WebhookVerificationError" in [
                h.id if isinstance(h, ast.Name) else getattr(h, "attr", "")
                for handler in child.handlers
                for h in ([handler.type] if handler.type else [])
            ], (
                "clerk_webhook has a try/except that isn't the signature check — "
                "this risks swallowing a delete-cascade failure and returning 200, "
                "which is the exact bug 2.1 flags for the Stripe webhook"
            )


def test_clerk_webhook_cascades_billing_and_usage(clerk_webhook_fn):
    called = _called_names(clerk_webhook_fn)
    for expected in ("delete_customer_for_user", "delete_user_usage", "_delete_paper_cascade"):
        assert expected in called, f"clerk_webhook no longer calls {expected} — cascade is incomplete"


def test_clerk_webhook_only_acts_on_user_deleted(clerk_webhook_fn):
    literals = _string_constants(clerk_webhook_fn)
    assert "user.deleted" in literals, "clerk_webhook must gate on the user.deleted event type"


# ── api/usage.py: delete_user_usage ──────────────────────────────────────────

def test_delete_user_usage_clears_both_tables():
    fn = _find_function(_tree(USAGE), "delete_user_usage")
    assert fn is not None, "delete_user_usage is gone from api/usage.py"
    sql = " ".join(_string_constants(fn)).lower()
    assert "delete from usage_events" in sql, "must delete the user's usage_events rows"
    assert "delete from users" in sql, "must delete the user's users row"


# ── api/billing.py: delete_customer_for_user ─────────────────────────────────

def test_delete_customer_for_user_removes_subscription_row():
    fn = _find_function(_tree(BILLING), "delete_customer_for_user")
    assert fn is not None, "delete_customer_for_user is gone from api/billing.py"
    sql = " ".join(_string_constants(fn)).lower()
    assert "delete from subscriptions" in sql

    called = _called_names(fn)
    assert "cancel" in called, "must cancel any active Stripe subscription, not just forget it locally"


def test_delete_customer_for_user_survives_a_stripe_outage():
    """A Stripe API failure must not block deleting the user's local data — the
    legal requirement — so the cancel call needs its own try/except that
    doesn't re-raise."""
    fn = _find_function(_tree(BILLING), "delete_customer_for_user")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    assert tries, "delete_customer_for_user must guard the Stripe call with try/except"
    for t in tries:
        for handler in t.handlers:
            reraises = any(isinstance(s, ast.Raise) and s.exc is None for s in ast.walk(handler))
            assert not reraises, "the Stripe-cancel except block must not bare-reraise"

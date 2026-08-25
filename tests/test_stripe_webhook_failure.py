"""
Regression tests for Launch Checklist 2.1 — Stripe webhook silently losing a
payment on a handler crash.

Why AST-based
-------------
Same reasoning as tests/test_account_deletion.py: `api.billing` connects to
live Stripe/Postgres at import time (see 2.12), so this suite parses the
source instead of importing it.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BILLING = ROOT / "api" / "billing.py"


def _tree() -> ast.AST:
    return ast.parse(BILLING.read_text(encoding="utf-8"), filename=str(BILLING))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _stripe_webhook_fn():
    fn = _find_function(_tree(), "stripe_webhook")
    assert fn is not None, "stripe_webhook handler is gone from api/billing.py"
    return fn


def test_handler_failure_is_not_swallowed_into_a_200():
    """The whole point of 2.1: a crash in checkout.session.completed /
    customer.subscription.* handling must not result in `{"received": True}`
    — that's the bug where a paid user stays on `free` forever because
    Stripe never retries a delivery it thinks succeeded."""
    fn = _stripe_webhook_fn()

    handler_try = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            called = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Attribute):
                        called.add(func.attr)
                    elif isinstance(func, ast.Name):
                        called.add(func.id)
            if "set_user_tier" in called:
                handler_try = node
                break

    assert handler_try is not None, (
        "expected a try/except wrapping the event-type dispatch that calls "
        "set_user_tier"
    )

    for handler in handler_try.handlers:
        raises_http_exception = False
        for stmt in handler.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name) and func.id == "HTTPException":
                        raises_http_exception = True
                if isinstance(child, ast.Raise):
                    raises_http_exception = True

        assert raises_http_exception, (
            "the except block around the webhook dispatch must re-raise "
            "(e.g. HTTPException(500, ...)) instead of logging-and-returning "
            "200 — otherwise Stripe stops retrying a failed tier flip and "
            "the payment is silently lost"
        )


def test_unknown_event_types_still_ack_200():
    """Only re-raise for event types we actually handle; unrecognized Stripe
    event types must still ACK 200 (they're not failures, just events we
    don't act on)."""
    fn = _stripe_webhook_fn()
    source = ast.dump(fn)
    assert "received" in source, "the 200 ack path ({'received': True}) must still exist"

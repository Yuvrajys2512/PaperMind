"""
Regression tests for Launch Checklist 2.2 — Stripe webhook has no idempotency
or ordering guard.

Why AST-based
-------------
Same reasoning as tests/test_stripe_webhook_failure.py: `api.billing`
connects to live Stripe/Postgres at import time (see 2.12).
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


def _string_constants(node) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


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


def test_dedupe_table_exists():
    fn = _find_function(_tree(), "_ensure_schema")
    assert fn is not None
    sql = " ".join(_string_constants(fn)).lower()
    assert "stripe_events" in sql, "expected a stripe_events dedupe table in _ensure_schema"
    assert "event_id" in sql


def test_mark_event_processed_uses_insert_or_skip():
    fn = _find_function(_tree(), "_mark_event_processed")
    assert fn is not None, "_mark_event_processed is gone from api/billing.py"
    sql = " ".join(_string_constants(fn)).lower()
    assert "insert into stripe_events" in sql
    assert "on conflict" in sql, "must insert-or-skip, not insert-or-error, on a duplicate event_id"


def test_webhook_checks_dedupe_before_dispatch():
    fn = _find_function(_tree(), "stripe_webhook")
    assert fn is not None
    called = _called_names(fn)
    assert "_mark_event_processed" in called, (
        "stripe_webhook must call _mark_event_processed so a Stripe retry "
        "doesn't re-run set_user_tier and doesn't re-trigger side effects"
    )


def test_webhook_has_an_ordering_guard():
    fn = _find_function(_tree(), "stripe_webhook")
    assert fn is not None
    called = _called_names(fn)
    assert "_current_subscription_updated_at" in called, (
        "stripe_webhook must compare the incoming event's created time against "
        "the stored subscriptions.updated_at before applying a "
        "subscription.updated/deleted event, or an out-of-order delivery can "
        "flip a live subscriber's tier back to a stale value"
    )


def test_record_subscription_stamps_event_created_time_not_wallclock():
    """updated_at must come from the Stripe event's `created` field, not
    now() — otherwise two events processed moments apart are
    indistinguishable and the ordering guard is meaningless."""
    fn = _find_function(_tree(), "_record_subscription")
    assert fn is not None
    sql_strings = [s for s in _string_constants(fn) if "insert into" in s.lower()]
    assert sql_strings, "expected an INSERT statement in _record_subscription"
    sql = " ".join(sql_strings).lower()
    assert "now()" not in sql, (
        "_record_subscription must stamp updated_at with the event's own "
        "created time (passed in), not now() — now() breaks the ordering "
        "guard in _current_subscription_updated_at"
    )

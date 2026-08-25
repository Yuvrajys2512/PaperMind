"""
Regression tests for Launch Checklist 2.3 — Stripe API version unpinned.

Why AST-based
-------------
`api.billing` imports `api.storage`, which connects to live Postgres at
import time (see 2.12), so this suite parses the source instead of
importing it.
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


def _module_assigns(tree) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_stripe_api_version_is_pinned_as_a_module_constant():
    tree = _tree()
    assert "STRIPE_API_VERSION" in _module_assigns(tree), (
        "expected an explicit STRIPE_API_VERSION module constant so a "
        "stripe SDK upgrade can't silently move the API version"
    )
    source = BILLING.read_text(encoding="utf-8")
    assert "stripe.api_version = STRIPE_API_VERSION" in source or (
        "stripe.api_version" in source and "STRIPE_API_VERSION" in source
    ), "STRIPE_API_VERSION must actually be assigned onto stripe.api_version"


def test_period_end_reads_from_subscription_items():
    fn = _find_function(_tree(), "_period_end_from")
    assert fn is not None, "_period_end_from is gone from api/billing.py"
    source = ast.unparse(fn)
    assert '"items"' in source or "'items'" in source, (
        "_period_end_from must read current_period_end from "
        "subscription['items']['data'][0] — recent Stripe API versions moved "
        "it off the top-level Subscription object"
    )
    assert "current_period_end" in source

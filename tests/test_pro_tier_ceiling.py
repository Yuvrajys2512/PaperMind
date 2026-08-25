"""
Regression tests for Launch Checklist 2.10 — `pro` tier is literally
unlimited.

AST-based: `api.usage` connects to live Postgres at import time (`_pool`
from api.storage, and `_ensure_schema()` runs at module load — see 2.12).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USAGE = ROOT / "api" / "usage.py"


def _tree() -> ast.AST:
    return ast.parse(USAGE.read_text(encoding="utf-8"), filename=str(USAGE))


def _find_dict_assign(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def test_pro_tier_has_real_caps_not_none():
    tier_limits = _find_dict_assign(_tree(), "TIER_LIMITS")
    assert tier_limits is not None, "TIER_LIMITS is gone from api/usage.py"
    assert isinstance(tier_limits, ast.Dict)

    pro_value = None
    for key, value in zip(tier_limits.keys, tier_limits.values):
        if isinstance(key, ast.Constant) and key.value == "pro":
            pro_value = value
    assert pro_value is not None, "no 'pro' entry in TIER_LIMITS"
    assert isinstance(pro_value, ast.Dict), (
        "TIER_LIMITS['pro'] must be a real limits dict, not None — an "
        "unlimited pro tier lets one subscriber exhaust the shared free "
        "Groq/Gemini/Mistral quotas for every other user"
    )

    pro_keys = {k.value for k in pro_value.keys if isinstance(k, ast.Constant)}
    for expected in ("max_papers", "max_queries_per_month", "max_audits_per_month"):
        assert expected in pro_keys, f"TIER_LIMITS['pro'] is missing {expected}"


def test_quota_error_messages_are_not_hardcoded_to_free():
    """The three quota-enforcing functions must name the user's actual tier
    in the 429 message, now that 'pro' can also hit a ceiling — a pro user
    reading 'Free tier limit reached' would be confused and think their
    subscription didn't take."""
    source = USAGE.read_text(encoding="utf-8")
    assert 'f"Free tier limit' not in source, (
        "quota error messages must not hardcode 'Free tier' now that pro "
        "carries a real limit too — use the actual tier name"
    )
    assert "tier.capitalize()" in source

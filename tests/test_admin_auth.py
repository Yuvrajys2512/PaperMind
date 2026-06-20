"""
Unit tests for api.auth.require_admin — the gate on admin-only endpoints
(e.g. GET /admin/usage).

This is a security boundary: it must deny everyone unless their Clerk user id is
explicitly in PAPERMIND_ADMIN_USER_IDS, and it must be locked to *nobody* when
the allow-list is empty (the default), so the admin surface can never ship open.
"""

import pytest
from fastapi import HTTPException

from api import auth


def test_listed_admin_is_allowed(monkeypatch):
    monkeypatch.setattr(auth, "_ADMIN_USER_IDS", {"user_admin"})
    assert auth.require_admin("user_admin") == "user_admin"


def test_non_admin_is_forbidden(monkeypatch):
    monkeypatch.setattr(auth, "_ADMIN_USER_IDS", {"user_admin"})
    with pytest.raises(HTTPException) as exc:
        auth.require_admin("user_someone_else")
    assert exc.value.status_code == 403


def test_empty_allowlist_locks_everyone_out(monkeypatch):
    # The default (no PAPERMIND_ADMIN_USER_IDS set) must deny even a real user.
    monkeypatch.setattr(auth, "_ADMIN_USER_IDS", set())
    with pytest.raises(HTTPException) as exc:
        auth.require_admin("any_user")
    assert exc.value.status_code == 403

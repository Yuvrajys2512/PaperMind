"""
Regression tests for the conftest.py network guard (Launch Checklist 2.12).

Verifies the guard conftest.py's pytest_configure installs is actually
active for this test run — i.e. these tests ARE running under the same
session where the patch applies, so a real connection attempt is provably
blocked rather than merely "not attempted."
"""

import pytest
from botocore.client import BaseClient
from psycopg_pool.pool import ConnectionPool


def test_postgres_open_is_blocked():
    pool = ConnectionPool.__new__(ConnectionPool)
    with pytest.raises(RuntimeError, match="Blocked a real Postgres connection"):
        ConnectionPool._open(pool)


def test_s3_api_call_is_blocked():
    client = BaseClient.__new__(BaseClient)
    with pytest.raises(RuntimeError, match="Blocked a real R2/S3 API call"):
        BaseClient._make_api_call(client, "GetObject", {})


def test_guard_is_not_a_monkeypatch_fixture_someone_could_forget():
    """The guard must be installed at the class level in pytest_configure,
    not via a fixture a test file could simply not request."""
    assert ConnectionPool._open.__name__ == "_blocked_pg_open"
    assert BaseClient._make_api_call.__name__ == "_blocked_s3_call"

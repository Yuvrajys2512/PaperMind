# conftest.py — pytest configuration

import os

# ingestion.llm_client raises at import time if no provider API key is present.
# Tests that exercise modules importing it (e.g. off_topic_guard) mock the actual
# network call, so a dummy key is enough to let the import succeed in CI / any
# environment without real credentials. setdefault never overrides a real key.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")


# ── Launch Checklist 2.12: block real Postgres/R2 calls for the whole suite ──
#
# This already happened once: a stubbing bug in a test let create_paper_record
# and upload_pdf reach the real services, writing 4 rows to the live papers
# table and 4 objects to live R2. api/storage.py's connection pool is lazy
# (Launch Checklist 2.12) so merely *importing* api.storage/api.usage/
# api.billing no longer opens a connection — but a test that calls a real
# storage function without fully stubbing it still can. This is the backstop:
# patch the two actual network chokepoints (psycopg_pool's connect and
# botocore's API-call dispatch) to raise instead of connecting, for every test
# in this suite. No test here is meant to touch a live service, so this should
# never fire — if it does, that's a stubbing bug to fix, not a guard to bypass.
#
# PAPERMIND_ALLOW_LIVE_TESTS=1 opts out, for a deliberate live integration
# test run outside the normal `pytest -q` workflow.
def pytest_configure(config):
    if os.environ.get("PAPERMIND_ALLOW_LIVE_TESTS") == "1":
        return

    from psycopg_pool.pool import ConnectionPool
    from botocore.client import BaseClient

    def _blocked_pg_open(self, *args, **kwargs):
        raise RuntimeError(
            "Blocked a real Postgres connection attempt during the test suite "
            "(Launch Checklist 2.12) — stub the call you're testing instead of "
            "letting it reach api.storage's real Neon pool. Set "
            "PAPERMIND_ALLOW_LIVE_TESTS=1 to bypass this deliberately."
        )

    def _blocked_s3_call(self, operation_name, *args, **kwargs):
        raise RuntimeError(
            f"Blocked a real R2/S3 API call ({operation_name}) during the test "
            "suite (Launch Checklist 2.12) — stub the call you're testing "
            "instead of letting it reach the real boto3 client. Set "
            "PAPERMIND_ALLOW_LIVE_TESTS=1 to bypass this deliberately."
        )

    ConnectionPool._open = _blocked_pg_open
    BaseClient._make_api_call = _blocked_s3_call

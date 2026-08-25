"""
Minimal integration test for structured logging.

Run with: python -m pytest tests/test_logging.py -v
Or manually: python -m tests.test_logging
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.logger import log_operation, generate_request_id, log_query


def test_generate_request_id():
    """Request ID generation is deterministic format."""
    req_id = generate_request_id()
    assert len(req_id) == 8
    assert all(c in "0123456789abcdef" for c in req_id)
    print(f"OK: Generated request ID {req_id}")


def test_log_operation_success():
    """Success operations are logged to disk."""
    req_id = generate_request_id()
    log_operation(
        "test_operation",
        "success",
        req_id=req_id,
        user_id="test_user_123",
        duration_ms=150,
        context={"test_key": "test_value"},
    )

    # Verify log file was written (operations.log exists and has content)
    log_file = Path("logs/operations.log")
    assert log_file.exists(), "logs/operations.log was not created"
    assert log_file.stat().st_size > 0, "logs/operations.log is empty"

    # Read last line and verify JSON
    with open(log_file) as f:
        last_line = f.readlines()[-1]

    record = json.loads(last_line)
    assert record["operation"] == "test_operation"
    assert record["status"] == "success"
    assert record["req_id"] == req_id
    assert record["user_id"] == "test_user_123"
    assert record["test_key"] == "test_value"
    print(f"OK: Logged success operation to {log_file}")


def test_log_operation_error():
    """Error operations are logged with exception details."""
    req_id = generate_request_id()
    test_exc = ValueError("Test error message")

    log_operation(
        "test_error_op",
        "error",
        req_id=req_id,
        user_id="test_user_456",
        error=test_exc,
        context={"paper_id": "paper_xyz"},
        duration_ms=500,
    )

    # Verify error was logged
    log_file = Path("logs/operations.log")
    with open(log_file) as f:
        last_line = f.readlines()[-1]

    record = json.loads(last_line)
    assert record["operation"] == "test_error_op"
    assert record["status"] == "error"
    assert "Test error message" in record.get("error", "")
    print("OK: Logged error operation with exception context")


def test_log_query():
    """Query logging remains functional."""
    req_id = generate_request_id()
    log_query(
        req_id=req_id,
        paper_id="paper_test_123",
        question="What is this paper about?",
        duration_ms=2500,
        confidence=0.95,
        attempts=1,
        passed=True,
        llm_calls=3,
        providers=["gemini", "groq"],
    )

    # Verify query log exists
    query_log = Path("logs/queries.jsonl")
    assert query_log.exists(), "logs/queries.jsonl was not created"

    with open(query_log) as f:
        last_line = f.readlines()[-1]

    record = json.loads(last_line)
    assert record["req_id"] == req_id
    assert record["paper_id"] == "paper_test_123"
    assert record["passed"] is True
    assert record["llm_calls"] == 3
    print("OK: Query logging functional")


if __name__ == "__main__":
    print("Running logging integration tests...\n")
    test_generate_request_id()
    test_log_operation_success()
    test_log_operation_error()
    test_log_query()
    print("\nAll tests passed!")


# ── Sentry reporting ─────────────────────────────────────────────────────────
# log_operation's Sentry branch is wrapped in a broad `except Exception` so a
# reporting failure can never take down a request. That safety net also hid a
# real bug for two months: the code called `sentry_sdk.with_scope(...)`, which
# does not exist in sentry-sdk 2.x, so EVERY capture raised AttributeError and
# was swallowed — Sentry looked configured and reported nothing.
#
# These tests pin the behaviour the safety net can otherwise hide: with a DSN
# set, an error must actually reach capture_exception/capture_message.

def _sentry_probe(monkeypatch):
    """Point log_operation's Sentry calls at recording stubs. Returns the list
    that captures land in."""
    from api import logger as logger_mod

    captured = []

    class _Scope:
        def set_context(self, *_args, **_kwargs):
            pass

        def set_tag(self, *_args, **_kwargs):
            pass

    class _NewScope:
        def __enter__(self):
            return _Scope()

        def __exit__(self, *_exc):
            return False

    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.invalid/1")
    monkeypatch.setattr(logger_mod.sentry_sdk, "new_scope", lambda: _NewScope())
    monkeypatch.setattr(
        logger_mod.sentry_sdk, "capture_exception",
        lambda exc: captured.append(("exception", exc)),
    )
    monkeypatch.setattr(
        logger_mod.sentry_sdk, "capture_message",
        lambda msg, level=None: captured.append(("message", msg)),
    )
    return captured


def test_error_with_exception_reaches_sentry(monkeypatch):
    captured = _sentry_probe(monkeypatch)
    boom = ValueError("boom")

    log_operation("test_op", "error", req_id="abc123", error=boom)

    assert captured == [("exception", boom)], (
        "an Exception passed to log_operation must reach capture_exception"
    )


def test_error_with_string_reaches_sentry(monkeypatch):
    captured = _sentry_probe(monkeypatch)

    log_operation("test_op", "error", error="timeout")

    assert captured == [("message", "timeout")], (
        "a string error must reach capture_message"
    )


def test_success_is_not_reported_to_sentry(monkeypatch):
    captured = _sentry_probe(monkeypatch)

    log_operation("test_op", "success", duration_ms=5)

    assert captured == [], "successful operations must not be sent to Sentry"


def test_no_dsn_means_no_capture(monkeypatch):
    captured = _sentry_probe(monkeypatch)
    monkeypatch.delenv("SENTRY_DSN", raising=False)

    log_operation("test_op", "error", error=ValueError("boom"))

    assert captured == [], "without SENTRY_DSN, nothing should be captured"


def test_sentry_failure_never_propagates(monkeypatch):
    """The safety net still has to work — a broken Sentry must not 500 a request."""
    from api import logger as logger_mod

    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.invalid/1")

    def _explode():
        raise RuntimeError("sentry is down")

    monkeypatch.setattr(logger_mod.sentry_sdk, "new_scope", _explode)

    log_operation("test_op", "error", error=ValueError("boom"))  # must not raise

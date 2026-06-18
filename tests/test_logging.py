"""
Minimal integration test for structured logging.

Run with: python -m pytest tests/test_logging.py -v
Or manually: python -m tests.test_logging
"""

import json
import tempfile
import os
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
    print(f"OK: Logged error operation with exception context")


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
    print(f"OK: Query logging functional")


if __name__ == "__main__":
    print("Running logging integration tests...\n")
    test_generate_request_id()
    test_log_operation_success()
    test_log_operation_error()
    test_log_query()
    print("\nAll tests passed!")

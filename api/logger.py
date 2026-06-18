"""
api/logger.py — Structured error & operation logging with Sentry integration.

Public API
----------
generate_request_id() -> str
log_query(req_id, paper_id, question, duration_ms, confidence, attempts,
          passed, llm_calls=0, providers=None) -> None
log_operation(operation: str, status: str, req_id=None, user_id=None,
              duration_ms=None, error=None, context=None) -> None
"""

import json
import uuid
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path

import sentry_sdk

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_QUERIES_JSONL = _LOG_DIR / "queries.jsonl"
_OPERATIONS_LOG = _LOG_DIR / "operations.log"

# Structured operation logger — write JSON lines to disk.
_op_logger = logging.getLogger("papermind.operations")
_op_logger.setLevel(logging.DEBUG)
if not _op_logger.handlers:
    handler = logging.handlers.RotatingFileHandler(
        _OPERATIONS_LOG,
        maxBytes=50 * 1024 * 1024,  # 50 MB per file
        backupCount=5,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _op_logger.addHandler(handler)


def generate_request_id() -> str:
    return uuid.uuid4().hex[:8]


def log_query(
    req_id: str,
    paper_id: str,
    question: str,
    duration_ms: int,
    confidence: float,
    attempts: int,
    passed: bool,
    llm_calls: int = 0,
    providers: list[str] | None = None,
) -> None:
    providers = providers or []
    trunc_q = question[:80].replace("\n", " ")
    status = "PASS" if passed else "FAIL"

    print(
        f"[{req_id}] {status} "
        f"paper={paper_id[:8]} "
        f'query="{trunc_q}" '
        f"duration={duration_ms}ms "
        f"confidence={confidence:.1f} "
        f"attempts={attempts}"
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "req_id": req_id,
        "paper_id": paper_id,
        "question": question[:200],
        "duration_ms": duration_ms,
        "confidence": round(float(confidence), 2),
        "attempts": attempts,
        "passed": bool(passed),
        "llm_calls": llm_calls,
        "providers": providers,
    }
    try:
        with open(_QUERIES_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[logger] failed to write JSONL record: {exc}")


def log_operation(
    operation: str,
    status: str,
    req_id: str | None = None,
    user_id: str | None = None,
    duration_ms: int | None = None,
    error: Exception | str | None = None,
    context: dict | None = None,
) -> None:
    """Log a structured operation (database, storage, ingestion, etc.).

    Args:
        operation: e.g. "upload_pdf", "ingest_document", "query_paper"
        status: "success" or "error"
        req_id: request ID for tracing
        user_id: user ID for ownership tracking
        duration_ms: operation duration
        error: Exception or error message
        context: extra context dict (paper_id, file_size, etc.)
    """
    context = context or {}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "status": status,
        "req_id": req_id,
        "user_id": user_id,
        "duration_ms": duration_ms,
        **context,
    }

    if error:
        error_str = str(error) if isinstance(error, Exception) else error
        record["error"] = error_str

    # Write to disk (never let logging take down a request).
    try:
        _op_logger.info(json.dumps(record, ensure_ascii=False))
    except Exception as exc:
        print(f"[logger] failed to write operation log: {exc}")

    # Report errors to Sentry with context.
    if status == "error" and error and os.getenv("SENTRY_DSN"):
        try:
            sentry_sdk.with_scope(
                lambda scope: (
                    scope.set_context("operation", {
                        "operation": operation,
                        "req_id": req_id,
                        "user_id": user_id,
                        "duration_ms": duration_ms,
                        **context,
                    }),
                    sentry_sdk.capture_exception(error)
                    if isinstance(error, Exception)
                    else sentry_sdk.capture_message(error_str, level="error"),
                )
            )
        except Exception:
            pass  # Sentry capture must never crash the app.

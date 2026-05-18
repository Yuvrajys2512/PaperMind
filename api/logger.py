"""
api/logger.py  — Session 6 + Phase-6 prep

Structured request logger.

Public API
----------
generate_request_id() -> str
log_query(req_id, paper_id, question, duration_ms, confidence, attempts,
          passed, llm_calls=0, providers=None) -> None

Each query is printed as a one-line stdout summary AND appended to
``logs/queries.jsonl`` as a JSON record so the eval harness can replay
runs without scraping stdout.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR        = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_QUERIES_JSONL  = _LOG_DIR / "queries.jsonl"


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
    trunc_q   = question[:80].replace("\n", " ")
    status    = "PASS" if passed else "FAIL"

    print(
        f"[{req_id}] {status} "
        f"paper={paper_id[:8]} "
        f'query="{trunc_q}" '
        f"duration={duration_ms}ms "
        f"confidence={confidence:.1f} "
        f"attempts={attempts}"
    )

    record = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "req_id":      req_id,
        "paper_id":    paper_id,
        "question":    question[:200],
        "duration_ms": duration_ms,
        "confidence":  round(float(confidence), 2),
        "attempts":    attempts,
        "passed":      bool(passed),
        "llm_calls":   llm_calls,
        "providers":   providers,
    }
    try:
        with open(_QUERIES_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Never let logging take down a request.
        print(f"[logger] failed to write JSONL record: {exc}")

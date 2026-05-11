"""
api/logger.py  — Session 6

Structured request logger.

Public API
----------
generate_request_id() -> str
log_query(req_id, paper_id, question, duration_ms, confidence, attempts, passed) -> None
"""

import uuid


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
) -> None:
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

"""
Regression test for Launch Checklist 1.4's billing fix: a startup Chroma
rebuild ('regenerate') must never be attributed to the paper's owner as usage.

AST-based because api/ingestion_runner.py imports api.storage, which connects
to live Postgres/R2 at import time (see 2.12) — this suite must never touch
production.
"""

import ast
from pathlib import Path

RUNNER = Path(__file__).resolve().parent.parent / "api" / "ingestion_runner.py"


def _tree():
    return ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_record_usage_is_guarded_against_kind_regenerate():
    fn = _find_function(_tree(), "run_ingestion_from_storage")
    assert fn is not None

    record_usage_call = next(
        (n for n in ast.walk(fn)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "record_usage"),
        None,
    )
    assert record_usage_call is not None, "run_ingestion_from_storage no longer calls record_usage"

    # Walk up from the call to find the nearest enclosing If and confirm its
    # test mentions both `kind` and a comparison against "regenerate" — i.e.
    # record_usage is conditioned on NOT being a regenerate run.
    guards = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and record_usage_call in ast.walk(n)
    ]
    assert guards, "record_usage call has no surrounding If guard"

    found_kind_guard = False
    for guard in guards:
        src = ast.dump(guard.test)
        if "kind" in src and "regenerate" in src:
            found_kind_guard = True
    assert found_kind_guard, (
        "no guard conditions record_usage on kind != 'regenerate' — a startup "
        "rebuild would be billed to the paper's owner, who did nothing"
    )


def test_regenerate_tries_a_snapshot_restore_before_reingesting():
    fn = _find_function(_tree(), "regenerate_missing_collections")
    assert fn is not None
    called = {
        c.func.id for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "restore_collection" in called, (
        "regenerate_missing_collections no longer tries restore_collection — "
        "it would fall back straight to a full, LLM-costing re-ingest"
    )
    assert "run_ingestion_from_storage" in called, (
        "the full re-ingest fallback is gone — a paper with no snapshot would "
        "never come back at all"
    )


def test_successful_ingest_uploads_a_snapshot():
    fn = _find_function(_tree(), "run_ingestion_from_storage")
    assert fn is not None
    called = {
        c.func.id for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "_snapshot_chroma_collection" in called

"""
Regression tests for where SSE endpoints do their usage accounting.

The bug these pin
-----------------
Every streaming endpoint runs its real work in a background task and yields
frames from a separate async generator. Caching the report and recording usage
used to live in that GENERATOR, in the `done` branch. That looks equivalent to
doing it on the worker and is not: when a client disconnects mid-stream the
generator is closed and never reaches `done`, while the worker task runs to
completion.

So an aborted audit burned the full LLM cost, wrote no cache, and recorded no
usage — and since `enforce_audit_quota` counts rows in `usage_events`, "start an
audit, abort it, repeat" was an unlimited-audit quota bypass with unbounded
provider spend.

Why these tests are AST-based
-----------------------------
`api.main` can't be imported in the offline suite (it needs DATABASE_URL, the R2
credentials and the torch stack at import time), and the defect is a property of
*where* the calls sit rather than of any single function's return value. Parsing
the source pins exactly that, and runs anywhere.
"""

import ast
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parent.parent / "api" / "main.py"

# Calls that must never sit inside an SSE generator, because a disconnected
# client means the generator never runs to completion.
ACCOUNTING_CALLS = {
    "record_usage",
    "log_query",
    "upload_audit_report",
    "upload_review_report",
    "upload_novelty_report",
    "upload_structure_report",
    "upload_numbers_report",
    "upload_citation_gap_report",
    "upload_overlap_report",
}

# The generator functions defined inside each streaming endpoint.
GENERATOR_NAMES = {"event_stream", "cached_stream"}


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def _called_names(node) -> set[str]:
    """Every plain function name called anywhere under `node`."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            names.add(child.func.id)
    return names


def _functions_named(tree, names):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            yield node


def test_the_generators_actually_exist(tree):
    """Guard against this whole file silently passing because a rename made the
    searches match nothing."""
    found = list(_functions_named(tree, GENERATOR_NAMES))
    assert len(found) >= 8, (
        f"expected at least 8 SSE generators, found {len(found)} — "
        "if they were renamed, update GENERATOR_NAMES"
    )


def test_no_accounting_inside_sse_generators(tree):
    """The core regression: accounting must not live where a disconnect skips it."""
    offenders = []
    for fn in _functions_named(tree, GENERATOR_NAMES):
        leaked = _called_names(fn) & ACCOUNTING_CALLS
        if leaked:
            offenders.append((fn.name, fn.lineno, sorted(leaked)))

    assert not offenders, (
        "usage accounting / report caching found inside an SSE generator — a "
        "client disconnect would skip it, letting the work run unbilled and "
        "uncached (quota bypass):\n"
        + "\n".join(f"  {name} (line {line}): {calls}" for name, line, calls in offenders)
    )


def _stream_endpoints(tree):
    """Every async route handler decorated with a `/stream` POST route."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and dec.args
                and isinstance(dec.args[0], ast.Constant)
                and isinstance(dec.args[0].value, str)
                and dec.args[0].value.endswith("/stream")
            ):
                yield node


def test_every_stream_endpoint_accounts_somewhere(tree):
    """The mirror of the test above: having removed accounting from the
    generators, each endpoint must still do it — on the worker side."""
    endpoints = list(_stream_endpoints(tree))
    assert len(endpoints) == 8, f"expected 8 stream endpoints, found {len(endpoints)}"

    missing = []
    for fn in endpoints:
        called = _called_names(fn)
        if not (called & ({"_finalize_stream_work", "record_usage"})):
            missing.append((fn.name, fn.lineno))

    assert not missing, (
        "stream endpoint does no usage accounting at all — the work would be "
        "free and uncounted:\n"
        + "\n".join(f"  {name} (line {line})" for name, line in missing)
    )


def test_finalize_helper_documents_the_worker_requirement(tree):
    """The helper's docstring is the only thing stopping someone moving these
    calls back into the generator, so make sure it stays."""
    helper = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_finalize_stream_work"),
        None,
    )
    assert helper is not None, "_finalize_stream_work is gone"
    doc = ast.get_docstring(helper) or ""
    assert "worker" in doc.lower(), (
        "_finalize_stream_work must document that it belongs on the worker task"
    )

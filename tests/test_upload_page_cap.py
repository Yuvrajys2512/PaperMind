"""
Regression test for Launch Checklist 2.6 — the /upload endpoint must reject
an over-length PDF before any storage or ingestion work starts.

AST-based: `api.main` connects to live Postgres/R2/Clerk at import time
(see 2.12).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "api" / "main.py"


def _tree():
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _called_names(node) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_upload_paper_checks_page_count():
    fn = _find_function(_tree(), "upload_paper")
    assert fn is not None, "upload_paper handler is gone from api/main.py"
    called = _called_names(fn)
    assert "count_pdf_pages" in called, (
        "upload_paper must call count_pdf_pages so an over-length PDF is "
        "rejected before storage/ingestion work starts"
    )


def test_upload_paper_rejects_before_creating_paper_record():
    """The page-count check must happen before create_paper_record — otherwise
    a rejected upload still leaves an orphaned DB row."""
    fn = _find_function(_tree(), "upload_paper")
    source_lines = ast.unparse(fn)
    cap_check_pos = source_lines.find("count_pdf_pages")
    create_record_pos = source_lines.find("create_paper_record")
    assert cap_check_pos != -1 and create_record_pos != -1
    assert cap_check_pos < create_record_pos, (
        "count_pdf_pages must run before create_paper_record, or a "
        "rejected over-length upload orphans a DB row"
    )

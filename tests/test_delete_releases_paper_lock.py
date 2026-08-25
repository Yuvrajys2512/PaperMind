"""
Regression test for Launch Checklist 2.11 — both places a paper is
permanently deleted (DELETE /papers/{id} and the Clerk account-deletion
webhook) must evict its lock entry, not just the shared cascade helper.

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


def test_delete_paper_releases_the_lock():
    fn = _find_function(_tree(), "delete_paper")
    assert fn is not None
    assert "release_paper_lock" in _called_names(fn)


def test_clerk_webhook_releases_the_lock():
    fn = _find_function(_tree(), "clerk_webhook")
    assert fn is not None
    assert "release_paper_lock" in _called_names(fn)

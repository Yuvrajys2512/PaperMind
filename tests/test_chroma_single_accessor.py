"""
Regression tests for Launch Checklist 2.9 — four separate Chroma clients on
one hardcoded relative path.

api.concurrency, ingestion.retriever, ingestion.bm25_retriever and
ingestion.chroma_snapshot are all safe to import directly (no live services
touched at import time — same reasoning as tests/test_chroma_snapshot.py and
tests/test_concurrency.py), so those are real imports.

ingestion.embedder is the one exception: it imports ingestion.models at
module level, which imports sentence-transformers/torch — importing torch in
the same process as docling/ingestion_runner segfaults on Windows (see
CLAUDE.md / memory). That's pre-existing, not something 2.9 changed, so
embedder's use of the shared accessor is checked via AST instead of a real
import.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import api.concurrency as concurrency
import ingestion.retriever as retriever
import ingestion.bm25_retriever as bm25_retriever
import ingestion.chroma_snapshot as chroma_snapshot

EMBEDDER = Path(__file__).resolve().parent.parent / "ingestion" / "embedder.py"


def test_chroma_path_is_absolute_by_default():
    assert os.path.isabs(concurrency.CHROMA_PATH)
    assert concurrency.CHROMA_PATH.endswith(os.path.join("data", "chroma_db"))


def test_chroma_path_is_configurable():
    """Verified out-of-process (rather than importlib.reload(concurrency) in
    this process) so this test can't leave a second `get_chroma_client`
    function object behind that breaks the identity checks below for every
    other module that imported the first one."""
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", "from api.concurrency import CHROMA_PATH; print(CHROMA_PATH)"],
        cwd=str(root),
        env={**os.environ, "PAPERMIND_CHROMA_PATH": "/tmp/custom-chroma"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/custom-chroma"


def test_retriever_uses_the_shared_accessor():
    assert retriever.get_chroma_client is concurrency.get_chroma_client


def test_embedder_uses_the_shared_accessor():
    source = EMBEDDER.read_text(encoding="utf-8")
    assert "from api.concurrency import get_chroma_client" in source
    assert "chromadb.PersistentClient(" not in source


def test_bm25_retriever_uses_the_shared_accessor():
    assert bm25_retriever.get_chroma_client is concurrency.get_chroma_client


def test_chroma_snapshot_uses_the_shared_accessor():
    assert chroma_snapshot.get_chroma_client is concurrency.get_chroma_client


def test_no_module_constructs_its_own_persistent_client():
    """Only api/concurrency.py may call chromadb.PersistentClient(...) —
    every reader (and the delete/snapshot paths) must go through
    get_chroma_client() instead, or the locking concurrency.py claims to
    provide never actually covers reads."""
    root = Path(__file__).resolve().parent.parent
    modules_that_must_not_construct_their_own = [
        root / "ingestion" / "retriever.py",
        root / "ingestion" / "embedder.py",
        root / "ingestion" / "bm25_retriever.py",
        root / "ingestion" / "chroma_snapshot.py",
    ]
    for path in modules_that_must_not_construct_their_own:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                assert name != "PersistentClient", (
                    f"{path.name} constructs its own chromadb.PersistentClient — "
                    "it must use api.concurrency.get_chroma_client() instead"
                )

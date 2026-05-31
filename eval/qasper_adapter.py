"""
eval/qasper_adapter.py — Ingest a QASPER paper into PaperMind's vector store.

QASPER already provides clean structured sections (``section_name`` +
``paragraphs``), so we skip the two most expensive ingestion steps — PDF text
extraction and LLM section detection — and go straight to the project's
standard chunker (512/100) and the existing embedder subprocess.

The ChromaDB collection name is derived from ``paper_id`` by the exact cleaning
rule used in ``ingestion/embedder.py`` and ``ingestion/retriever.py``, so a
later ``answer_query(question, paper_id)`` resolves to this same collection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ingestion.chunker import chunk_sections  # noqa: E402

_WORKER = _ROOT / "ingestion" / "embedder_worker.py"


def build_sections(paper: dict) -> list:
    """Convert a QASPER paper into the section list ``chunk_sections`` expects.

    Each section is ``{heading, type, page_num, body}``. QASPER has no PDF
    layout, so ``page_num`` carries the section ordinal purely to satisfy
    downstream code that expects an int — it is not a real page.
    """
    sections = []

    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        sections.append(
            {"heading": "Abstract", "type": "abstract", "page_num": 0, "body": abstract}
        )

    for i, sec in enumerate(paper.get("full_text", []), start=1):
        name = (sec.get("section_name") or f"Section {i}").strip()
        paragraphs = [p for p in sec.get("paragraphs", []) if p and p.strip()]
        body = "\n\n".join(paragraphs)
        if not body.strip():
            continue
        sections.append(
            {"heading": name, "type": "body", "page_num": i, "body": body}
        )

    return sections


def ingest_qasper_paper(paper_id: str, paper: dict) -> dict:
    """Chunk and embed one QASPER paper. Returns a summary dict.

    Embedding runs in the same isolated subprocess the main pipeline uses
    (``embedder_worker.py``), which keeps the heavy PyTorch import out of this
    process and reuses battle-tested storage code.
    """
    sections = build_sections(paper)
    chunks = chunk_sections(sections, chunk_size=512, overlap=100)

    if not chunks:
        return {
            "success": False, "paper_id": paper_id,
            "num_sections": len(sections), "num_chunks": 0,
            "error": "no chunks produced (empty full_text?)",
        }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(chunks, f)
        temp_path = f.name

    error = None
    try:
        subprocess.run(
            [sys.executable, str(_WORKER), temp_path, paper_id],
            check=True, capture_output=True, text=True,
        )
        success = True
    except subprocess.CalledProcessError as e:
        success = False
        error = f"embedder failed (exit {e.returncode}): {(e.stderr or '')[-300:]}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "success": success, "paper_id": paper_id,
        "num_sections": len(sections), "num_chunks": len(chunks),
        "error": error,
    }

"""
Regression tests for Launch Checklist 2.5 — `subprocess.run` with no timeout,
inside the paper lock.

`ingestion.ingest_document` is safe to import directly: unlike
`ingestion.embedder` (torch) it doesn't touch the model at module level, and
it doesn't connect to any live service.
"""

import subprocess

from ingestion import ingest_document


# ── _should_embed_in_process ─────────────────────────────────────────────────

def test_windows_defaults_to_subprocess(monkeypatch):
    monkeypatch.setattr(ingest_document, "_EMBED_IN_PROCESS_OVERRIDE", None)
    monkeypatch.setattr(ingest_document.platform, "system", lambda: "Windows")
    assert ingest_document._should_embed_in_process() is False


def test_linux_defaults_to_in_process(monkeypatch):
    monkeypatch.setattr(ingest_document, "_EMBED_IN_PROCESS_OVERRIDE", None)
    monkeypatch.setattr(ingest_document.platform, "system", lambda: "Linux")
    assert ingest_document._should_embed_in_process() is True


def test_env_override_forces_subprocess_on_linux(monkeypatch):
    monkeypatch.setattr(ingest_document, "_EMBED_IN_PROCESS_OVERRIDE", "0")
    monkeypatch.setattr(ingest_document.platform, "system", lambda: "Linux")
    assert ingest_document._should_embed_in_process() is False


def test_env_override_forces_in_process_on_windows(monkeypatch):
    monkeypatch.setattr(ingest_document, "_EMBED_IN_PROCESS_OVERRIDE", "1")
    monkeypatch.setattr(ingest_document.platform, "system", lambda: "Windows")
    assert ingest_document._should_embed_in_process() is True


# ── subprocess timeout ───────────────────────────────────────────────────────

def _minimal_chunks():
    return [{"text": "hello", "section": "Intro", "section_type": "prose",
              "page_num": 1, "chunk_index": 0, "total_chunks_in_section": 1,
              "token_count": 1}]


def test_subprocess_run_is_called_with_a_timeout(monkeypatch):
    """The whole point of 2.5: a hung embedder child must not deadlock the
    paper lock forever."""
    monkeypatch.setattr(ingest_document, "_should_embed_in_process", lambda: False)

    captured_kwargs = {}

    def fake_run(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(ingest_document.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ingest_document, "extract_text_from_pdf",
        lambda path: {"pages": [{"text": "x"}], "full_text": "x", "total_pages": 1, "tables": []},
    )
    monkeypatch.setattr(ingest_document, "remove_credits_block", lambda t: t)
    monkeypatch.setattr(ingest_document, "remove_references_section", lambda t: t)
    monkeypatch.setattr(ingest_document, "build_candidates", lambda pages: [])
    monkeypatch.setattr(ingest_document, "confirm_headings_with_llm", lambda c: [])
    monkeypatch.setattr(
        ingest_document, "assemble_sections",
        lambda pages, confirmed, candidates: [{"heading": "Intro", "section_type": "prose", "text": "x", "page_num": 1}],
    )
    monkeypatch.setattr(ingest_document, "chunk_sections", lambda sections, chunk_size, overlap: _minimal_chunks())
    monkeypatch.setattr(ingest_document, "tables_to_chunks", lambda records: [])

    result = ingest_document.ingest_document("fake.pdf", "paper-1")

    assert result["success"] is True
    assert "timeout" in captured_kwargs, "subprocess.run must be called with timeout="
    assert captured_kwargs["timeout"] == ingest_document.EMBEDDER_SUBPROCESS_TIMEOUT_SECONDS
    assert captured_kwargs["timeout"] is not None


def test_subprocess_timeout_marks_ingestion_failed(monkeypatch):
    monkeypatch.setattr(ingest_document, "_should_embed_in_process", lambda: False)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 300))

    monkeypatch.setattr(ingest_document.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ingest_document, "extract_text_from_pdf",
        lambda path: {"pages": [{"text": "x"}], "full_text": "x", "total_pages": 1, "tables": []},
    )
    monkeypatch.setattr(ingest_document, "remove_credits_block", lambda t: t)
    monkeypatch.setattr(ingest_document, "remove_references_section", lambda t: t)
    monkeypatch.setattr(ingest_document, "build_candidates", lambda pages: [])
    monkeypatch.setattr(ingest_document, "confirm_headings_with_llm", lambda c: [])
    monkeypatch.setattr(
        ingest_document, "assemble_sections",
        lambda pages, confirmed, candidates: [{"heading": "Intro", "section_type": "prose", "text": "x", "page_num": 1}],
    )
    monkeypatch.setattr(ingest_document, "chunk_sections", lambda sections, chunk_size, overlap: _minimal_chunks())
    monkeypatch.setattr(ingest_document, "tables_to_chunks", lambda records: [])

    result = ingest_document.ingest_document("fake.pdf", "paper-1")

    assert result["success"] is False
    assert "timed out" in result["error"].lower()


def test_in_process_path_calls_embed_and_store_directly(monkeypatch):
    monkeypatch.setattr(ingest_document, "_should_embed_in_process", lambda: True)

    called = {}

    class FakeEmbedderModule:
        @staticmethod
        def embed_and_store(chunks, paper_name):
            called["chunks"] = chunks
            called["paper_name"] = paper_name

    monkeypatch.setitem(__import__("sys").modules, "ingestion.embedder", FakeEmbedderModule)

    monkeypatch.setattr(
        ingest_document, "extract_text_from_pdf",
        lambda path: {"pages": [{"text": "x"}], "full_text": "x", "total_pages": 1, "tables": []},
    )
    monkeypatch.setattr(ingest_document, "remove_credits_block", lambda t: t)
    monkeypatch.setattr(ingest_document, "remove_references_section", lambda t: t)
    monkeypatch.setattr(ingest_document, "build_candidates", lambda pages: [])
    monkeypatch.setattr(ingest_document, "confirm_headings_with_llm", lambda c: [])
    monkeypatch.setattr(
        ingest_document, "assemble_sections",
        lambda pages, confirmed, candidates: [{"heading": "Intro", "section_type": "prose", "text": "x", "page_num": 1}],
    )
    monkeypatch.setattr(ingest_document, "chunk_sections", lambda sections, chunk_size, overlap: _minimal_chunks())
    monkeypatch.setattr(ingest_document, "tables_to_chunks", lambda records: [])

    result = ingest_document.ingest_document("fake.pdf", "paper-1")

    assert result["success"] is True
    assert called["paper_name"] == "paper-1"

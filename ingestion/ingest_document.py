import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path

# Make sure the project root is on the path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.pdf_parser import (
    extract_text_from_pdf,
    remove_credits_block,
    remove_references_section,
)
from ingestion.section_detector import (
    build_candidates,
    confirm_headings_with_llm,
    assemble_sections,
)
from ingestion.chunker import chunk_sections


def ingest_document(pdf_path: str, paper_name: str) -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Args:
        pdf_path:   Absolute or relative path to the PDF file on disk.
        paper_name: A unique identifier for this paper (used as the
                    ChromaDB collection name). Typically the paper_id UUID.

    Returns:
        A dict with ingestion summary:
        {
            "success": bool,
            "paper_name": str,
            "total_pages": int,
            "num_chunks": int,
            "error": str | None
        }
    """
    try:
        # ── Step 1: Parse the PDF ──────────────────────────────────────────
        print(f"[ingest] Parsing PDF: {pdf_path}")
        parsed = extract_text_from_pdf(pdf_path)
        pages = parsed["pages"]
        full_text = parsed["full_text"]
        total_pages = parsed["total_pages"]
        print(f"[ingest] Parsed {total_pages} pages")

        # ── Step 2: Clean the full text ────────────────────────────────────
        # (used internally if needed; pages list passes through unchanged)
        full_text = remove_credits_block(full_text)
        full_text = remove_references_section(full_text)
        print(f"[ingest] Text cleaned ({len(full_text)} chars after trimming)")

        # ── Step 3: Detect sections ────────────────────────────────────────
        print("[ingest] Building section heading candidates...")
        candidates = build_candidates(pages)
        print(f"[ingest] {len(candidates)} candidates found, confirming with LLM...")
        confirmed = confirm_headings_with_llm(candidates)
        sections = assemble_sections(pages, confirmed, candidates)
        print(f"[ingest] {len(sections)} sections assembled")

        # ── Step 4: Chunk ──────────────────────────────────────────────────
        print("[ingest] Chunking sections...")
        chunks = chunk_sections(sections, chunk_size=512, overlap=100)
        print(f"[ingest] {len(chunks)} chunks created")

        # ── Step 5: Embed and store ────────────────────────────────────────
        print(f"[ingest] Embedding and storing into ChromaDB as '{paper_name}'...")
        
        # We run the embedder in a subprocess to prevent a silent native Windows DLL
        # collision between PyTorch (SentenceTransformers) and Groq/pdfplumber
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(chunks, f)
            temp_path = f.name
            
        try:
            worker_script = os.path.join(os.path.dirname(__file__), "embedder_worker.py")
            result = subprocess.run(
                [sys.executable, worker_script, temp_path, paper_name],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ingest] Embedding process failed with exit code {e.returncode}")
            print(f"[ingest] STDOUT: {e.stdout}")
            print(f"[ingest] STDERR: {e.stderr}")
            raise RuntimeError("Embedder subprocess failed") from e
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        print("[ingest] Done.")

        return {
            "success": True,
            "paper_name": paper_name,
            "total_pages": total_pages,
            "num_chunks": len(chunks),
            "error": None
        }

    except Exception as e:
        print(f"[ingest] ERROR: {e}")
        return {
            "success": False,
            "paper_name": paper_name,
            "total_pages": 0,
            "num_chunks": 0,
            "error": str(e)
        }
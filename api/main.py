import sys
import asyncio
import time
import shutil
from typing import Optional
from pydantic import BaseModel

# Ensure Unicode LLM output never crashes the server on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ingestion.pipeline import answer_query, compare_papers
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.storage import (
    create_paper_record,
    update_paper_status,
    get_paper,
    list_papers,
    get_paper_pdf_path,
    delete_paper_record,
)
from api.logger import generate_request_id, log_query
from ingestion.ingest_document import ingest_document
from ingestion.bm25_retriever  import invalidate_bm25_cache

app = FastAPI(
    title="PaperMind API",
    description="AI-powered research paper Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_ingestion(paper_id: str, pdf_path: str, paper_name: str):
    result = ingest_document(pdf_path=pdf_path, paper_name=paper_id)
    if result["success"]:
        update_paper_status(paper_id, "ready")
    else:
        update_paper_status(paper_id, "failed", error=result["error"])


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    paper_id = create_paper_record(file.filename)
    pdf_path = get_paper_pdf_path(paper_id)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(
        run_ingestion,
        paper_id=paper_id,
        pdf_path=str(pdf_path),
        paper_name=paper_id
    )

    return {"paper_id": paper_id, "filename": file.filename, "status": "processing"}


@app.get("/status/{paper_id}")
def get_status(paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@app.get("/papers")
def get_all_papers():
    return list_papers()


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Remove from registry
    delete_paper_record(paper_id)

    # Delete PDF from disk
    pdf_path = get_paper_pdf_path(paper_id)
    if pdf_path.exists():
        pdf_path.unlink()

    # Drop ChromaDB collection
    try:
        import chromadb
        chroma = chromadb.PersistentClient(path="data/chroma_db")
        clean_name = "".join(
            c if c.isalnum() or c == "-" else "-" for c in paper_id
        ).strip("-").lower()
        chroma.delete_collection(name=clean_name)
    except Exception as exc:
        # The collection may legitimately be missing (ingestion never
        # completed) — log it so a real failure is still visible.
        print(f"[delete] Chroma collection drop skipped for {paper_id}: {exc}")

    # Invalidate the BM25 cache so a re-ingest of the same paper_id
    # doesn't serve stale tokens.
    try:
        invalidate_bm25_cache(paper_id)
    except Exception as exc:
        print(f"[delete] BM25 cache invalidation failed for {paper_id}: {exc}")

    return {"deleted": paper_id}


class QueryRequest(BaseModel):
    paper_id: Optional[str] = ""
    paper_ids: Optional[list[str]] = []
    question: str


@app.post("/query")
async def query_paper(request: QueryRequest):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()

    # ── Multi-paper comparison ────────────────────────────────────────────
    if request.paper_ids and len(request.paper_ids) == 2:
        paper_id_a, paper_id_b = request.paper_ids[0], request.paper_ids[1]

        for pid in (paper_id_a, paper_id_b):
            p = get_paper(pid)
            if not p:
                raise HTTPException(status_code=404, detail=f"Paper {pid} not found.")
            if p["status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Paper {pid} is not ready yet.")

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, compare_papers, request.question, paper_id_a, paper_id_b),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Comparison timed out after 120 seconds.")

        duration_ms = round((time.monotonic() - t0) * 1000)
        log_query(
            req_id=req_id,
            paper_id=f"{paper_id_a[:4]}+{paper_id_b[:4]}",
            question=request.question,
            duration_ms=duration_ms,
            confidence=result.get("confidence", 0),
            attempts=result.get("attempts", 1),
            passed=result.get("passed", False),
            llm_calls=result.get("llm_calls", 0),
            providers=result.get("providers_used", []),
        )
        result["request_id"] = req_id
        return result

    # ── Single-paper query ────────────────────────────────────────────────
    paper_id = request.paper_id or (request.paper_ids[0] if request.paper_ids else "")
    if not paper_id:
        raise HTTPException(status_code=400, detail="Provide paper_id or paper_ids.")

    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Paper is not ready yet. Current status: {paper['status']}"
        )

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, answer_query, request.question, paper_id),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out after 60 seconds.")

    duration_ms = round((time.monotonic() - t0) * 1000)
    log_query(
        req_id=req_id,
        paper_id=paper_id,
        question=request.question,
        duration_ms=duration_ms,
        confidence=result.get("confidence", 0),
        attempts=result.get("attempts", 1),
        passed=result.get("passed", False),
        llm_calls=result.get("llm_calls", 0),
        providers=result.get("providers_used", []),
    )
    result["request_id"] = req_id
    return result

import sys
import asyncio
import json
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
from fastapi.responses import StreamingResponse
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
from discovery.router import router as discovery_router
from discovery.search  import search_papers

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

app.include_router(discovery_router)


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


@app.get("/papers/{paper_id}/glossary")
async def get_glossary(paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(status_code=400, detail="Paper not ready yet.")

    loop = asyncio.get_running_loop()

    def _extract():
        from ingestion.retriever import retrieve
        from ingestion.llm_client import chat_completion
        chunks = retrieve("technical terms methods algorithms definitions equations notation", paper_id, top_k=10)
        context = "\n\n---\n\n".join(c["text"][:600] for c in chunks)
        raw = chat_completion(
            messages=[{"role": "user", "content": (
                "Extract every domain-specific technical term, acronym, and piece of jargon "
                "from the passages below. For each, write a plain-English definition (1-2 sentences).\n\n"
                "Return ONLY a JSON array — no markdown, no preamble:\n"
                '[{"term":"...","definition":"...","category":"method|metric|dataset|concept|model"}]\n\n'
                f"Passages:\n{context}"
            )}],
            max_tokens=1800,
            temperature=0.1,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    try:
        terms = await asyncio.wait_for(loop.run_in_executor(None, _extract), timeout=40.0)
        return {"terms": terms}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Glossary extraction failed: {exc}")


@app.get("/papers/{paper_id}/recommendations")
async def get_recommendations(paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(status_code=400, detail="Paper not ready yet.")

    loop = asyncio.get_running_loop()

    def _get_queries():
        from ingestion.retriever import retrieve
        from ingestion.llm_client import chat_completion
        chunks = retrieve("main contribution methodology results key findings", paper_id, top_k=5)
        context = "\n\n".join(c["text"][:400] for c in chunks)
        raw = chat_completion(
            messages=[{"role": "user", "content": (
                "Based on this research paper excerpt, generate 3 short academic search queries "
                "to find closely related papers. Return ONLY a JSON array of strings.\n\n"
                f"Paper excerpt:\n{context}"
            )}],
            max_tokens=150,
            temperature=0.1,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    try:
        queries = await asyncio.wait_for(loop.run_in_executor(None, _get_queries), timeout=20.0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not extract topics: {exc}")

    seen, results = set(), []
    for q in queries[:3]:
        try:
            found = await search_papers(q, limit=5)
            for r in found:
                rid = r.get("id") or r.get("title", "")
                if rid not in seen:
                    seen.add(rid)
                    r["search_query"] = q
                    results.append(r)
        except Exception:
            continue

    return {"results": results[:12], "queries": queries}


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


# ─────────────────────────────────────────────────────────────────────────────
# Streaming endpoint — Server-Sent Events
# ─────────────────────────────────────────────────────────────────────────────
#
# Why SSE and not WebSocket: the channel is one-way (server → client), the
# payload is small JSON deltas, and EventSource / fetch-stream parsing on
# the browser is trivial. SSE also survives ordinary proxies that block
# WebSocket upgrades.
#
# Architecture:
#   1. Client POSTs to /query/stream.
#   2. We spawn the pipeline in run_in_executor (it's sync CPU/LLM work).
#   3. The executor passes a thread-safe on_progress callback that
#      drops {stage, message, ...} dicts onto an asyncio.Queue.
#   4. The SSE generator drains the queue, yielding each as a
#      `data: <json>\n\n` frame. The final frame's stage is "done"
#      (carrying the full result) or "error".

def _sse_format(event_type: str, payload: dict) -> str:
    """Encode one Server-Sent Event frame."""
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _make_progress_pusher(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Return a callback the pipeline can invoke from its worker thread.
    Uses call_soon_threadsafe so the asyncio.Queue is touched only from
    the loop thread — putting from another thread directly is undefined
    behaviour on some Python versions.
    """
    def push(event: dict):
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", event))
    return push


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()

    is_compare = bool(request.paper_ids and len(request.paper_ids) == 2)

    # Validate inputs the same way /query does — bail fast with HTTPException
    # before opening the event stream, so the client gets a real 4xx.
    if is_compare:
        paper_id_a, paper_id_b = request.paper_ids[0], request.paper_ids[1]
        for pid in (paper_id_a, paper_id_b):
            p = get_paper(pid)
            if not p:
                raise HTTPException(status_code=404, detail=f"Paper {pid} not found.")
            if p["status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Paper {pid} is not ready yet.")
        log_paper_id = f"{paper_id_a[:4]}+{paper_id_b[:4]}"
    else:
        paper_id = request.paper_id or (request.paper_ids[0] if request.paper_ids else "")
        if not paper_id:
            raise HTTPException(status_code=400, detail="Provide paper_id or paper_ids.")
        paper = get_paper(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found.")
        if paper["status"] != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Paper is not ready yet. Current status: {paper['status']}",
            )
        log_paper_id = paper_id

    on_progress = _make_progress_pusher(queue, loop)

    async def run_pipeline():
        try:
            if is_compare:
                fn = lambda: compare_papers(
                    request.question, paper_id_a, paper_id_b, on_progress=on_progress,
                )
                timeout = 120.0
            else:
                fn = lambda: answer_query(
                    request.question, paper_id, request_id=req_id, on_progress=on_progress,
                )
                timeout = 60.0

            result = await asyncio.wait_for(
                loop.run_in_executor(None, fn),
                timeout=timeout,
            )
            await queue.put(("done", result))
        except asyncio.TimeoutError:
            await queue.put(("error", {
                "message": "Query timed out. The paper may be unusually large or the LLM providers slow.",
            }))
        except Exception as exc:
            await queue.put(("error", {"message": f"Pipeline failed: {exc}"}))

    asyncio.create_task(run_pipeline())

    async def event_stream():
        # First frame so the client knows the channel is open before any
        # heavy work lands. Helps clients detect connection success.
        yield _sse_format("open", {"req_id": req_id})

        while True:
            kind, payload = await queue.get()

            if kind == "progress":
                yield _sse_format("progress", payload)
                continue

            if kind == "error":
                yield _sse_format("error", payload)
                # Log the failure as a FAIL so it shows up in queries.jsonl
                log_query(
                    req_id=req_id,
                    paper_id=log_paper_id,
                    question=request.question,
                    duration_ms=round((time.monotonic() - t0) * 1000),
                    confidence=0,
                    attempts=0,
                    passed=False,
                )
                return

            if kind == "done":
                result = payload
                result["request_id"] = req_id
                log_query(
                    req_id=req_id,
                    paper_id=log_paper_id,
                    question=request.question,
                    duration_ms=round((time.monotonic() - t0) * 1000),
                    confidence=result.get("confidence", 0),
                    attempts=result.get("attempts", 1),
                    passed=result.get("passed", False),
                    llm_calls=result.get("llm_calls", 0),
                    providers=result.get("providers_used", []),
                )
                yield _sse_format("done", result)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # Disable nginx/proxy buffering so progress events flush
        # immediately rather than batching.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

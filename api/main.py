import sys
import asyncio
import json
import os
import tempfile
import time
from typing import Optional
from pydantic import BaseModel

# Ensure Unicode LLM output never crashes the server on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import threading

from ingestion.pipeline import answer_query, compare_papers
from ingestion.rewriter import rewrite_text
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from api.auth import get_current_user_id
from api.ingestion_runner import run_ingestion_from_storage, regenerate_missing_collections
from api.storage import (
    create_paper_record,
    update_paper_status,
    get_owned_paper,
    list_papers,
    upload_pdf,
    get_pdf_stream,
    delete_pdf,
    delete_paper_record,
)
from api.usage import enforce_paper_quota, enforce_query_quota, get_usage_summary, record_usage
from api.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, PDF_MAGIC
from api.logger import generate_request_id, log_query
from ingestion.bm25_retriever  import invalidate_bm25_cache
from ingestion.retriever import collection_name
from api.billing import router as billing_router
from discovery.router import router as discovery_router
from discovery.search  import search_papers

# CORS: a comma-separated allow-list, locked down from the old "*". Defaults to
# the Vite dev origin; set ALLOWED_ORIGINS to the real domain(s) in production.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

# Per-IP rate limiting. 120/min is generous for real humans but caps scripted
# abuse on the public endpoints. The Stripe webhook and /health pings ride the
# same global limit — fine at launch volume; add a path exemption if needed.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
    title="PaperMind API",
    description="AI-powered research paper Q&A",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(discovery_router)
app.include_router(billing_router)


@app.on_event("startup")
def _regenerate_chroma_on_startup():
    """On an ephemeral-disk host (e.g. HF Spaces free tier), data/chroma_db is
    wiped on redeploy. Rebuild any missing collection from R2 in a background
    daemon thread so /health comes up immediately and papers become queryable
    as they finish rebuilding. Disable with PAPERMIND_REGENERATE_ON_STARTUP=0."""
    if os.getenv("PAPERMIND_REGENERATE_ON_STARTUP", "1") == "0":
        return
    threading.Thread(target=regenerate_missing_collections, daemon=True).start()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(enforce_paper_quota),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Stream to a temp file with server-side validation: the bytes must
    # actually be a PDF (magic number), and stay under the size cap. A .pdf
    # extension alone is trivially spoofable.
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    total = 0
    first = True
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                if first:
                    if not chunk.startswith(PDF_MAGIC):
                        raise HTTPException(status_code=400, detail="File is not a valid PDF.")
                    first = False
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_UPLOAD_MB} MB upload limit.",
                    )
                f.write(chunk)
        if first:  # never entered the loop → empty file
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except HTTPException:
        os.remove(temp_path)
        raise
    except Exception:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    paper_id = create_paper_record(file.filename, user_id)
    try:
        upload_pdf(paper_id, temp_path)
    except Exception as exc:
        update_paper_status(paper_id, "failed", error=f"Upload to storage failed: {exc}")
        raise HTTPException(status_code=502, detail="Failed to store the uploaded PDF.")
    finally:
        os.remove(temp_path)

    background_tasks.add_task(run_ingestion_from_storage, paper_id)

    return {"paper_id": paper_id, "filename": file.filename, "status": "processing"}


@app.get("/status/{paper_id}")
def get_status(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_owned_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@app.get("/papers")
def get_all_papers(user_id: str = Depends(get_current_user_id)):
    return list_papers(user_id)


@app.get("/usage")
def get_usage(user_id: str = Depends(get_current_user_id)):
    return get_usage_summary(user_id)


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_owned_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Delete the PDF from R2 before the registry row. If this raises, the
    # row is left in place (a recoverable "row exists, blob gone" state) —
    # deleting the row first would risk an unreachable orphaned blob if the
    # PDF delete then failed.
    try:
        delete_pdf(paper_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to delete PDF from storage: {exc}")

    # Remove from registry
    delete_paper_record(paper_id)

    # Drop ChromaDB collection
    try:
        import chromadb
        chroma = chromadb.PersistentClient(path="data/chroma_db")
        chroma.delete_collection(name=collection_name(paper_id))
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


@app.get("/papers/{paper_id}/pdf")
def serve_paper_pdf(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_owned_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    try:
        stream = get_pdf_stream(paper_id)
    except Exception:
        raise HTTPException(status_code=404, detail="PDF file not found.")
    filename = paper.get("filename", f"{paper_id}.pdf")
    return StreamingResponse(
        stream.iter_chunks(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/papers/{paper_id}/glossary")
async def get_glossary(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_owned_paper(paper_id, user_id)
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
async def get_recommendations(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_owned_paper(paper_id, user_id)
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


class RewriteRequest(BaseModel):
    text: str
    mode: str  # "academic" | "plain" | "concise"


@app.post("/rewrite")
async def rewrite(request: RewriteRequest, user_id: str = Depends(get_current_user_id)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if request.mode not in ("academic", "plain", "concise"):
        raise HTTPException(status_code=400, detail="mode must be academic, plain, or concise.")

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, rewrite_text, request.text, request.mode),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Rewrite timed out.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rewrite failed: {exc}")

    return {"result": result, "mode": request.mode}


class QueryRequest(BaseModel):
    paper_id: Optional[str] = ""
    paper_ids: Optional[list[str]] = []
    question: str


@app.post("/query")
async def query_paper(request: QueryRequest, user_id: str = Depends(enforce_query_quota)):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()

    # ── Multi-paper comparison ────────────────────────────────────────────
    if request.paper_ids and len(request.paper_ids) == 2:
        paper_id_a, paper_id_b = request.paper_ids[0], request.paper_ids[1]

        for pid in (paper_id_a, paper_id_b):
            p = get_owned_paper(pid, user_id)
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
        record_usage(
            user_id=user_id,
            kind="query",
            req_id=req_id,
            paper_id=f"{paper_id_a[:4]}+{paper_id_b[:4]}",
            llm_calls=result.get("llm_calls", 0),
            tokens_in=result.get("tokens_in", 0),
            tokens_out=result.get("tokens_out", 0),
            cost_usd=result.get("cost_usd", 0.0),
        )
        result["request_id"] = req_id
        return result

    # ── Single-paper query ────────────────────────────────────────────────
    paper_id = request.paper_id or (request.paper_ids[0] if request.paper_ids else "")
    if not paper_id:
        raise HTTPException(status_code=400, detail="Provide paper_id or paper_ids.")

    paper = get_owned_paper(paper_id, user_id)
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
    record_usage(
        user_id=user_id,
        kind="query",
        req_id=req_id,
        paper_id=paper_id,
        llm_calls=result.get("llm_calls", 0),
        tokens_in=result.get("tokens_in", 0),
        tokens_out=result.get("tokens_out", 0),
        cost_usd=result.get("cost_usd", 0.0),
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
async def query_stream(request: QueryRequest, user_id: str = Depends(enforce_query_quota)):
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
            p = get_owned_paper(pid, user_id)
            if not p:
                raise HTTPException(status_code=404, detail=f"Paper {pid} not found.")
            if p["status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Paper {pid} is not ready yet.")
        log_paper_id = f"{paper_id_a[:4]}+{paper_id_b[:4]}"
    else:
        paper_id = request.paper_id or (request.paper_ids[0] if request.paper_ids else "")
        if not paper_id:
            raise HTTPException(status_code=400, detail="Provide paper_id or paper_ids.")
        paper = get_owned_paper(paper_id, user_id)
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
                def fn():
                    return compare_papers(
                        request.question, paper_id_a, paper_id_b, on_progress=on_progress,
                    )
                timeout = 120.0
            else:
                def fn():
                    return answer_query(
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
                record_usage(
                    user_id=user_id,
                    kind="query",
                    req_id=req_id,
                    paper_id=log_paper_id,
                    llm_calls=result.get("llm_calls", 0),
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=result.get("tokens_out", 0),
                    cost_usd=result.get("cost_usd", 0.0),
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

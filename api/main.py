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
from ingestion.claim_auditor import audit_paper
from ingestion.reviewer_auditor import review_paper
from ingestion.rewriter import rewrite_text
from ingestion.llm_client import get_stats, reset_stats
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from api.auth import get_current_user_id, require_admin
from api.ingestion_runner import run_ingestion_from_storage, regenerate_missing_collections
from api.logger import generate_request_id
from api.concurrency import get_chroma_client, paper_locked
from api.storage import (
    create_paper_record,
    update_paper_status,
    get_owned_paper,
    get_readable_paper,
    list_papers,
    list_demo_papers,
    DEMO_USER_ID,
    upload_pdf,
    get_pdf_stream,
    delete_pdf,
    delete_paper_record,
    upload_audit_report,
    get_audit_report,
    delete_audit_report,
    upload_review_report,
    get_review_report,
    delete_review_report,
)
from api.usage import (
    enforce_paper_quota,
    enforce_query_quota,
    enforce_audit_quota,
    get_aggregate_usage,
    get_usage_summary,
    record_usage,
)
from api.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, PDF_MAGIC
from api.logger import generate_request_id, log_query
from ingestion.bm25_retriever  import invalidate_bm25_cache
from ingestion.retriever import collection_name
from api.billing import router as billing_router
from discovery.router import router as discovery_router
from discovery.search  import search_papers

# Error tracking. Guarded by SENTRY_DSN so it's a complete no-op when unset —
# observability is optional and must never break local dev or CI (unlike auth/
# storage/billing, which fail loud). sentry-sdk auto-instruments FastAPI when
# initialized early, so unhandled route exceptions are captured automatically.
import sentry_sdk
from api.logger import log_operation

_sentry_dsn = os.getenv("SENTRY_DSN")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        traces_sample_rate=0.1,
        send_default_pii=False,
    )

# CORS: a comma-separated allow-list, locked down from the old "*". Defaults to
# the Vite dev origin; set ALLOWED_ORIGINS to the real domain(s) in production.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

# Caps on free-text LLM inputs. Without these a single request can carry
# megabytes of text straight into the providers — blowing up token cost and
# risking provider context-window errors. Override via env if a real workload
# needs more headroom.
MAX_QUESTION_CHARS = int(os.getenv("PAPERMIND_MAX_QUESTION_CHARS", "2000"))
MAX_REWRITE_CHARS = int(os.getenv("PAPERMIND_MAX_REWRITE_CHARS", "8000"))

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


# Middleware to attach request_id to Sentry scope for error tracing.
@app.middleware("http")
async def attach_request_id(request, call_next):
    req_id = generate_request_id()
    request.state.req_id = req_id
    if _sentry_dsn:
        sentry_sdk.set_tag("request_id", req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


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
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    paper_type: str = Form("paper"),
    user_id: str = Depends(enforce_paper_quota),
):
    req_id = getattr(request.state, 'req_id', 'unknown')
    t0 = time.monotonic()

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if paper_type not in ("paper", "draft"):
        raise HTTPException(status_code=400, detail="paper_type must be 'paper' or 'draft'.")

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
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        log_operation(
            "upload_read_file",
            "error",
            req_id=req_id,
            user_id=user_id,
            error=exc,
            context={"filename": file.filename},
        )
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    paper_id = create_paper_record(file.filename, user_id, paper_type=paper_type)
    try:
        upload_pdf(paper_id, temp_path)
        log_operation(
            "upload_pdf",
            "success",
            req_id=req_id,
            user_id=user_id,
            duration_ms=round((time.monotonic() - t0) * 1000),
            context={"paper_id": paper_id, "filename": file.filename},
        )
    except Exception as exc:
        update_paper_status(paper_id, "failed", error=f"Upload to storage failed: {exc}")
        log_operation(
            "upload_pdf",
            "error",
            req_id=req_id,
            user_id=user_id,
            error=exc,
            context={"paper_id": paper_id, "filename": file.filename},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=502, detail="Failed to store the uploaded PDF.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    background_tasks.add_task(run_ingestion_from_storage, paper_id)

    return {"paper_id": paper_id, "filename": file.filename, "status": "processing", "paper_type": paper_type}


@app.get("/status/{paper_id}")
def get_status(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_readable_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@app.get("/papers")
def get_all_papers(paper_type: str = "paper", user_id: str = Depends(get_current_user_id)):
    """The user's own papers (newest first) followed by the shared, read-only
    demo set. `is_demo` lets the frontend badge them and hide their delete
    control — they're quota-exempt and not owned by the requester.

    paper_type defaults to 'paper' (today's behavior, unchanged) so every
    existing caller keeps seeing only reference papers + the demo set. Pass
    'draft' to list unpublished drafts instead — the demo set never contains
    drafts, so it's omitted in that case."""
    if paper_type not in ("paper", "draft"):
        raise HTTPException(status_code=400, detail="paper_type must be 'paper' or 'draft'.")
    own = [{**p, "is_demo": False} for p in list_papers(user_id, paper_type=paper_type)]
    demo = [{**p, "is_demo": True} for p in list_demo_papers()] if paper_type == "paper" else []
    return own + demo


@app.get("/usage")
def get_usage(user_id: str = Depends(get_current_user_id)):
    return get_usage_summary(user_id)


@app.get("/admin/usage")
def get_admin_usage(admin_id: str = Depends(require_admin)):
    """Aggregate cross-user usage + upstream-capacity projection. Gated to the
    Clerk user ids in PAPERMIND_ADMIN_USER_IDS. See product/llm_api.md."""
    return get_aggregate_usage()


@app.delete("/papers/{paper_id}")
def delete_paper(request: Request, paper_id: str, user_id: str = Depends(get_current_user_id)):
    req_id = getattr(request.state, 'req_id', 'unknown')

    paper = get_owned_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Acquire exclusive lock on this paper to prevent concurrent deletes/queries
    with paper_locked(paper_id):
        # Delete the PDF from R2 before the registry row. If this raises, the
        # row is left in place (a recoverable "row exists, blob gone" state) —
        # deleting the row first would risk an unreachable orphaned blob if the
        # PDF delete then failed.
        try:
            delete_pdf(paper_id)
        except Exception as exc:
            log_operation(
                "delete_pdf",
                "error",
                req_id=req_id,
                user_id=user_id,
                error=exc,
                context={"paper_id": paper_id},
            )
            raise HTTPException(status_code=502, detail=f"Failed to delete PDF from storage: {exc}")

        # Remove from registry
        try:
            delete_paper_record(paper_id)
        except Exception as exc:
            log_operation(
                "delete_paper_record",
                "error",
                req_id=req_id,
                user_id=user_id,
                error=exc,
                context={"paper_id": paper_id},
            )
            raise HTTPException(status_code=500, detail="Failed to delete paper from registry.")

        # Drop ChromaDB collection using thread-safe client
        try:
            chroma = get_chroma_client()
            chroma.delete_collection(name=collection_name(paper_id))
        except Exception as exc:
            # The collection may legitimately be missing (ingestion never
            # completed) — log it so a real failure is still visible.
            log_operation(
                "delete_chroma_collection",
                "error",
                req_id=req_id,
                user_id=user_id,
                error=exc,
                context={"paper_id": paper_id},
            )

        # Invalidate the BM25 cache so a re-ingest of the same paper_id
        # doesn't serve stale tokens.
        try:
            invalidate_bm25_cache(paper_id)
        except Exception as exc:
            log_operation(
                "invalidate_bm25_cache",
                "error",
                req_id=req_id,
                user_id=user_id,
                error=exc,
                context={"paper_id": paper_id},
            )

        # Drop the cached claim-audit and reviewer-audit reports, if any.
        # Best-effort — an orphaned blob is harmless, so a missing/failed delete
        # must not block the rest.
        for _drop, _op in (
            (delete_audit_report, "delete_audit_report"),
            (delete_review_report, "delete_review_report"),
        ):
            try:
                _drop(paper_id)
            except Exception as exc:
                log_operation(
                    _op,
                    "error",
                    req_id=req_id,
                    user_id=user_id,
                    error=exc,
                    context={"paper_id": paper_id},
                )

    log_operation(
        "delete_paper",
        "success",
        req_id=req_id,
        user_id=user_id,
        context={"paper_id": paper_id},
    )

    return {"deleted": paper_id}


@app.get("/papers/{paper_id}/pdf")
def serve_paper_pdf(paper_id: str, user_id: str = Depends(get_current_user_id)):
    paper = get_readable_paper(paper_id, user_id)
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
async def get_glossary(paper_id: str, user_id: str = Depends(enforce_query_quota)):
    t0 = time.monotonic()
    req_id = generate_request_id()
    paper = get_readable_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(status_code=400, detail="Paper not ready yet.")

    loop = asyncio.get_running_loop()

    def _extract():
        from ingestion.retriever import retrieve
        from ingestion.llm_client import chat_completion
        reset_stats()
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
        return json.loads(raw.strip()), get_stats()

    try:
        terms, stats = await asyncio.wait_for(loop.run_in_executor(None, _extract), timeout=40.0)
        record_usage(
            user_id=user_id,
            kind="query",
            req_id=req_id,
            paper_id=paper_id,
            llm_calls=stats["call_count"],
            tokens_in=stats["tokens_in"],
            tokens_out=stats["tokens_out"],
            cost_usd=stats["cost_usd"],
        )
        log_operation(
            "extract_glossary",
            "success",
            req_id=req_id,
            user_id=user_id,
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        return {"terms": terms}
    except asyncio.TimeoutError:
        log_operation(
            "extract_glossary",
            "error",
            user_id=user_id,
            error="timeout",
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=504, detail="Glossary extraction timed out.")
    except Exception as exc:
        log_operation(
            "extract_glossary",
            "error",
            user_id=user_id,
            error=exc,
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=500, detail="Glossary extraction failed.")


@app.get("/papers/{paper_id}/recommendations")
async def get_recommendations(paper_id: str, user_id: str = Depends(enforce_query_quota)):
    t0 = time.monotonic()
    req_id = generate_request_id()
    paper = get_readable_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(status_code=400, detail="Paper not ready yet.")

    loop = asyncio.get_running_loop()

    def _get_queries():
        from ingestion.retriever import retrieve
        from ingestion.llm_client import chat_completion
        reset_stats()
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
        return json.loads(raw.strip()), get_stats()

    try:
        queries, stats = await asyncio.wait_for(loop.run_in_executor(None, _get_queries), timeout=20.0)
        record_usage(
            user_id=user_id,
            kind="query",
            req_id=req_id,
            paper_id=paper_id,
            llm_calls=stats["call_count"],
            tokens_in=stats["tokens_in"],
            tokens_out=stats["tokens_out"],
            cost_usd=stats["cost_usd"],
        )
    except asyncio.TimeoutError:
        log_operation(
            "extract_recommendations",
            "error",
            user_id=user_id,
            error="timeout",
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=504, detail="Recommendation extraction timed out.")
    except Exception as exc:
        log_operation(
            "extract_recommendations",
            "error",
            user_id=user_id,
            error=exc,
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=500, detail="Could not extract topics.")

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
        except Exception as exc:
            log_operation(
                "search_related_papers",
                "error",
                user_id=user_id,
                error=exc,
                context={"paper_id": paper_id, "query": q},
            )
            continue

    log_operation(
        "extract_recommendations",
        "success",
        user_id=user_id,
        context={"paper_id": paper_id, "results_count": len(results)},
        duration_ms=round((time.monotonic() - t0) * 1000),
    )
    return {"results": results[:12], "queries": queries}


class RewriteRequest(BaseModel):
    text: str
    mode: str  # "academic" | "plain" | "concise"


@app.post("/rewrite")
async def rewrite(request: RewriteRequest, user_id: str = Depends(enforce_query_quota)):
    t0 = time.monotonic()
    req_id = generate_request_id()
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(request.text) > MAX_REWRITE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long ({len(request.text)} chars). Max is {MAX_REWRITE_CHARS}.",
        )
    if request.mode not in ("academic", "plain", "concise"):
        raise HTTPException(status_code=400, detail="mode must be academic, plain, or concise.")

    loop = asyncio.get_running_loop()

    def _do_rewrite():
        # reset_stats / get_stats are thread-local and must run in the same
        # executor thread as the LLM call, so the token/cost numbers belong to
        # this request only.
        reset_stats()
        out = rewrite_text(request.text, request.mode)
        return out, get_stats()

    try:
        result, stats = await asyncio.wait_for(
            loop.run_in_executor(None, _do_rewrite),
            timeout=30.0,
        )
        record_usage(
            user_id=user_id,
            kind="query",
            req_id=req_id,
            llm_calls=stats["call_count"],
            tokens_in=stats["tokens_in"],
            tokens_out=stats["tokens_out"],
            cost_usd=stats["cost_usd"],
        )
        log_operation(
            "rewrite_text",
            "success",
            req_id=req_id,
            user_id=user_id,
            context={"mode": request.mode, "text_length": len(request.text)},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
    except asyncio.TimeoutError:
        log_operation(
            "rewrite_text",
            "error",
            user_id=user_id,
            error="timeout",
            context={"mode": request.mode},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=504, detail="Rewrite timed out.")
    except Exception as exc:
        log_operation(
            "rewrite_text",
            "error",
            user_id=user_id,
            error=exc,
            context={"mode": request.mode},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=500, detail="Rewrite failed.")

    return {"result": result, "mode": request.mode}


class QueryRequest(BaseModel):
    paper_id: Optional[str] = ""
    paper_ids: Optional[list[str]] = []
    question: str


def _validate_question(question: str) -> None:
    """Reject empty or oversized questions before any LLM work fires. Shared by
    /query and /query/stream so both endpoints enforce the same cap."""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long ({len(question)} chars). Max is {MAX_QUESTION_CHARS}.",
        )


@app.post("/query")
async def query_paper(req: QueryRequest, user_id: str = Depends(enforce_query_quota)):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()

    _validate_question(req.question)

    # ── Multi-paper comparison ────────────────────────────────────────────
    if req.paper_ids and len(req.paper_ids) == 2:
        paper_id_a, paper_id_b = req.paper_ids[0], req.paper_ids[1]

        for pid in (paper_id_a, paper_id_b):
            p = get_readable_paper(pid, user_id)
            if not p:
                raise HTTPException(status_code=404, detail=f"Paper {pid} not found.")
            if p["status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Paper {pid} is not ready yet.")

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, compare_papers, req.question, paper_id_a, paper_id_b),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            log_operation(
                "compare_papers",
                "error",
                req_id=req_id,
                user_id=user_id,
                error="timeout",
                context={"paper_id_a": paper_id_a, "paper_id_b": paper_id_b},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            raise HTTPException(status_code=504, detail="Comparison timed out after 120 seconds.")
        except Exception as exc:
            log_operation(
                "compare_papers",
                "error",
                req_id=req_id,
                user_id=user_id,
                error=exc,
                context={"paper_id_a": paper_id_a, "paper_id_b": paper_id_b},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            raise HTTPException(status_code=500, detail="Comparison failed.")

        duration_ms = round((time.monotonic() - t0) * 1000)
        log_query(
            req_id=req_id,
            paper_id=f"{paper_id_a[:4]}+{paper_id_b[:4]}",
            question=req.question,
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
    paper_id = req.paper_id or (req.paper_ids[0] if req.paper_ids else "")
    if not paper_id:
        raise HTTPException(status_code=400, detail="Provide paper_id or paper_ids.")

    paper = get_readable_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Paper is not ready yet. Current status: {paper['status']}"
        )

    # Acquire lock to prevent concurrent queries from corrupting Chroma state
    def _locked_query():
        with paper_locked(paper_id):
            return answer_query(req.question, paper_id)

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _locked_query),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        log_operation(
            "answer_query",
            "error",
            req_id=req_id,
            user_id=user_id,
            error="timeout",
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=504, detail="Query timed out after 60 seconds.")
    except Exception as exc:
        log_operation(
            "answer_query",
            "error",
            req_id=req_id,
            user_id=user_id,
            error=exc,
            context={"paper_id": paper_id},
            duration_ms=round((time.monotonic() - t0) * 1000),
        )
        raise HTTPException(status_code=500, detail="Query failed.")

    duration_ms = round((time.monotonic() - t0) * 1000)
    log_query(
        req_id=req_id,
        paper_id=paper_id,
        question=req.question,
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
async def query_stream(req: QueryRequest, user_id: str = Depends(enforce_query_quota)):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()

    _validate_question(req.question)

    is_compare = bool(req.paper_ids and len(req.paper_ids) == 2)

    # Validate inputs the same way /query does — bail fast with HTTPException
    # before opening the event stream, so the client gets a real 4xx.
    if is_compare:
        paper_id_a, paper_id_b = req.paper_ids[0], req.paper_ids[1]
        for pid in (paper_id_a, paper_id_b):
            p = get_readable_paper(pid, user_id)
            if not p:
                raise HTTPException(status_code=404, detail=f"Paper {pid} not found.")
            if p["status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Paper {pid} is not ready yet.")
        log_paper_id = f"{paper_id_a[:4]}+{paper_id_b[:4]}"
    else:
        paper_id = req.paper_id or (req.paper_ids[0] if req.paper_ids else "")
        if not paper_id:
            raise HTTPException(status_code=400, detail="Provide paper_id or paper_ids.")
        paper = get_readable_paper(paper_id, user_id)
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
                    # For comparison, lock both papers to prevent concurrent access
                    with paper_locked(paper_id_a):
                        with paper_locked(paper_id_b):
                            return compare_papers(
                                req.question, paper_id_a, paper_id_b, on_progress=on_progress,
                            )
                timeout = 120.0
            else:
                def fn():
                    with paper_locked(paper_id):
                        return answer_query(
                            req.question, paper_id, request_id=req_id, on_progress=on_progress,
                        )
                timeout = 60.0

            result = await asyncio.wait_for(
                loop.run_in_executor(None, fn),
                timeout=timeout,
            )
            await queue.put(("done", result))
        except asyncio.TimeoutError:
            log_operation(
                "query_stream" if not is_compare else "compare_stream",
                "error",
                req_id=req_id,
                user_id=user_id,
                error="timeout",
                context={"paper_id": log_paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Query timed out. The paper may be unusually large or the LLM providers slow.",
            }))
        except Exception as exc:
            log_operation(
                "query_stream" if not is_compare else "compare_stream",
                "error",
                req_id=req_id,
                user_id=user_id,
                error=exc,
                context={"paper_id": log_paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
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
                    question=req.question,
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
                    question=req.question,
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


# ─────────────────────────────────────────────────────────────────────────────
# Claim audit — claim→evidence grounding + overclaim detection (SSE)
# ─────────────────────────────────────────────────────────────────────────────
#
# Reuses the /query/stream architecture (progress pusher + SSE generator). An
# audit is the heaviest LLM operation in the app (claim extraction + per-claim
# retrieval + batched verdicts), so the result is cached in R2 and re-served
# without LLM cost unless ?force=1 is passed.

@app.post("/papers/{paper_id}/audit/stream")
async def audit_paper_stream(
    paper_id: str,
    request: Request,
    force: bool = False,
    user_id: str = Depends(enforce_audit_quota),
):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()

    paper = get_readable_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Paper is not ready yet. Current status: {paper['status']}",
        )

    # Serve the cached report instantly when present (no LLM, no usage recorded).
    if not force:
        cached = get_audit_report(paper_id)
        if cached:
            async def cached_stream():
                yield _sse_format("open", {"req_id": req_id})
                cached["cached"] = True
                yield _sse_format("done", cached)
            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    on_progress = _make_progress_pusher(queue, loop)

    async def run_audit():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # audit's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = audit_paper(paper_id, on_progress=on_progress)
                    return report, get_stats()

            report, stats = await asyncio.wait_for(
                loop.run_in_executor(None, fn),
                timeout=180.0,
            )
            await queue.put(("done", (report, stats)))
        except asyncio.TimeoutError:
            log_operation(
                "audit_paper", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Audit timed out. The paper may be unusually large — please try again.",
            }))
        except Exception as exc:
            log_operation(
                "audit_paper", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Audit failed: {exc}"}))

    asyncio.create_task(run_audit())

    async def event_stream():
        yield _sse_format("open", {"req_id": req_id})

        while True:
            kind, payload = await queue.get()

            if kind == "progress":
                yield _sse_format("progress", payload)
                continue

            if kind == "error":
                yield _sse_format("error", payload)
                return

            if kind == "done":
                report, stats = payload
                # Only record usage + cache when the audit actually produced a
                # report (a hard failure shouldn't be cached or billed).
                if not report.get("audit_failed"):
                    try:
                        upload_audit_report(paper_id, report)
                    except Exception as exc:
                        log_operation(
                            "upload_audit_report", "error", req_id=req_id,
                            user_id=user_id, error=exc, context={"paper_id": paper_id},
                        )
                    record_usage(
                        user_id=user_id,
                        kind="audit",
                        req_id=req_id,
                        paper_id=paper_id,
                        llm_calls=stats["call_count"],
                        tokens_in=stats["tokens_in"],
                        tokens_out=stats["tokens_out"],
                        cost_usd=stats["cost_usd"],
                    )
                log_operation(
                    "audit_paper", "success", req_id=req_id, user_id=user_id,
                    context={
                        "paper_id": paper_id,
                        "claims_checked": report.get("claims_checked", 0),
                        "flagged": report.get("flagged", 0),
                    },
                    duration_ms=round((time.monotonic() - t0) * 1000),
                )
                report["request_id"] = req_id
                report["cached"] = False
                yield _sse_format("done", report)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer / weakness audit — methodological completeness vs. venue norms (SSE)
# ─────────────────────────────────────────────────────────────────────────────
#
# Complement of the claim audit above: instead of "do the claims match the
# paper's own evidence?", this asks "does the paper meet the methodological bar
# a reviewer would hold it to?" (missing baselines / ablations / error bars /
# small N / threats to validity / thin related work). Same architecture: progress
# pusher + SSE, R2-cached, re-served without LLM cost unless ?force=1 is passed.

@app.post("/papers/{paper_id}/review/stream")
async def review_paper_stream(
    paper_id: str,
    request: Request,
    force: bool = False,
    user_id: str = Depends(enforce_audit_quota),
):
    req_id = generate_request_id()
    loop   = asyncio.get_running_loop()
    t0     = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()

    paper = get_readable_paper(paper_id, user_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Paper is not ready yet. Current status: {paper['status']}",
        )

    # Serve the cached report instantly when present (no LLM, no usage recorded).
    if not force:
        cached = get_review_report(paper_id)
        if cached:
            async def cached_stream():
                yield _sse_format("open", {"req_id": req_id})
                cached["cached"] = True
                yield _sse_format("done", cached)
            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    on_progress = _make_progress_pusher(queue, loop)

    async def run_review():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # review's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = review_paper(paper_id, on_progress=on_progress)
                    return report, get_stats()

            report, stats = await asyncio.wait_for(
                loop.run_in_executor(None, fn),
                timeout=180.0,
            )
            await queue.put(("done", (report, stats)))
        except asyncio.TimeoutError:
            log_operation(
                "review_paper", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Review timed out. The paper may be unusually large — please try again.",
            }))
        except Exception as exc:
            log_operation(
                "review_paper", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Review failed: {exc}"}))

    asyncio.create_task(run_review())

    async def event_stream():
        yield _sse_format("open", {"req_id": req_id})

        while True:
            kind, payload = await queue.get()

            if kind == "progress":
                yield _sse_format("progress", payload)
                continue

            if kind == "error":
                yield _sse_format("error", payload)
                return

            if kind == "done":
                report, stats = payload
                # Only record usage + cache when the review actually produced a
                # report (a hard failure shouldn't be cached or billed).
                if not report.get("review_failed"):
                    try:
                        upload_review_report(paper_id, report)
                    except Exception as exc:
                        log_operation(
                            "upload_review_report", "error", req_id=req_id,
                            user_id=user_id, error=exc, context={"paper_id": paper_id},
                        )
                    record_usage(
                        user_id=user_id,
                        kind="audit",
                        req_id=req_id,
                        paper_id=paper_id,
                        llm_calls=stats["call_count"],
                        tokens_in=stats["tokens_in"],
                        tokens_out=stats["tokens_out"],
                        cost_usd=stats["cost_usd"],
                    )
                log_operation(
                    "review_paper", "success", req_id=req_id, user_id=user_id,
                    context={
                        "paper_id": paper_id,
                        "dimensions_checked": report.get("dimensions_checked", 0),
                        "weak": report.get("weak", 0),
                        "missing": report.get("missing", 0),
                    },
                    duration_ms=round((time.monotonic() - t0) * 1000),
                )
                report["request_id"] = req_id
                report["cached"] = False
                yield _sse_format("done", report)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

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
from ingestion.novelty_scout import find_related_work
from ingestion.structure_auditor import check_structure, list_venues
from ingestion.numbers_auditor import audit_numbers
from ingestion.citation_gap_auditor import audit_citation_gaps
from ingestion.plagiarism_auditor import audit_overlap
from ingestion.rewriter import rewrite_text
from ingestion.llm_client import get_stats, reset_stats
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from api.auth import get_current_user_id, require_admin, limiter
from api.ingestion_runner import run_ingestion_from_storage, regenerate_missing_collections
from ingestion.pdf_parser import count_pdf_pages, MAX_PDF_PAGES
from api.logger import generate_request_id
from api.concurrency import get_chroma_client, paper_locked, release_paper_lock, run_on_executor, track_task
from api.storage import (
    create_paper_record,
    update_paper_status,
    get_owned_paper,
    get_readable_paper,
    list_papers,
    list_demo_papers,
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
    upload_novelty_report,
    get_novelty_report,
    delete_novelty_report,
    upload_structure_report,
    get_structure_report,
    delete_structure_report,
    upload_numbers_report,
    get_numbers_report,
    delete_numbers_report,
    upload_citation_gap_report,
    get_citation_gap_report,
    delete_citation_gap_report,
    upload_overlap_report,
    get_overlap_report,
    delete_overlap_report,
    delete_chroma_snapshot,
)
from api.usage import (
    enforce_paper_quota,
    enforce_query_quota,
    enforce_audit_quota,
    get_aggregate_usage,
    get_usage_summary,
    record_usage,
    delete_user_usage,
)
from api.content_disposition import content_disposition
from api.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, PDF_MAGIC
from api.logger import log_query
from ingestion.bm25_retriever  import invalidate_bm25_cache
from ingestion.retriever import collection_name
from api.billing import router as billing_router, delete_customer_for_user
from svix.webhooks import Webhook, WebhookVerificationError
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

# Clerk account-deletion webhook. Guarded the same way as billing: absent
# means the route 503s instead of the app crashing at import, so local dev
# and pre-Clerk-webhook deploys stay runnable.
CLERK_WEBHOOK_SECRET = os.getenv("CLERK_WEBHOOK_SECRET")

# Caps on free-text LLM inputs. Without these a single request can carry
# megabytes of text straight into the providers — blowing up token cost and
# risking provider context-window errors. Override via env if a real workload
# needs more headroom.
MAX_QUESTION_CHARS = int(os.getenv("PAPERMIND_MAX_QUESTION_CHARS", "2000"))
MAX_REWRITE_CHARS = int(os.getenv("PAPERMIND_MAX_REWRITE_CHARS", "8000"))

# Rate limiting keyed per authenticated user, falling back to IP — see
# api.auth.rate_limit_key for why keying on the client address alone was broken
# behind a proxy. The Limiter instance itself lives in api.auth (not here) so
# discovery/router.py can add its own per-route limits without an import
# cycle back to api.main.
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


@app.post("/webhooks/clerk")
async def clerk_webhook(request: Request):
    """Handles Clerk's `user.deleted` event: cascades the delete across every
    store this app owns, so a deleted Clerk account doesn't leave orphaned
    data behind — required by our own Privacy Policy §5.

    No Clerk session auth — Clerk calls this directly. Authenticity comes from
    the Svix signature headers over the raw body (Clerk webhooks are sent via
    Svix), verified below."""
    if not CLERK_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Clerk webhook is not configured on this server.")

    payload = await request.body()
    try:
        event = Webhook(CLERK_WEBHOOK_SECRET).verify(payload, dict(request.headers))
    except WebhookVerificationError as exc:
        log_operation(
            "clerk_webhook",
            "error",
            error=exc,
            context={"reason": "invalid_signature"},
        )
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    event_type = event.get("type")
    if event_type != "user.deleted":
        return {"received": True}

    user_id = (event.get("data") or {}).get("id")
    if not user_id:
        return {"received": True}

    # Deliberately not try/except-and-swallow like the Stripe webhook (2.1
    # flags that pattern as a bug there). A failure here must surface as a
    # non-2xx so Svix retries — the alternative is silently keeping a deleted
    # user's data forever, which is the exact compliance gap this closes. Each
    # per-paper/store delete below is already best-effort internally, so a
    # retry after a partial failure is safe: it just re-deletes what's left.
    papers = list_papers(user_id)
    for paper in papers:
        paper_id = paper["paper_id"]
        with paper_locked(paper_id):
            delete_pdf(paper_id)
            delete_paper_record(paper_id)
            _delete_paper_cascade(paper_id, user_id, req_id="clerk-webhook")
        release_paper_lock(paper_id)

    delete_customer_for_user(user_id)
    delete_user_usage(user_id)

    log_operation(
        "clerk_user_deleted",
        "success",
        user_id=user_id,
        context={"event_id": event.get("id"), "papers_deleted": len(papers)},
    )
    return {"received": True}


@app.post("/upload")
@limiter.limit("5/minute")
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

    try:
        num_pages = count_pdf_pages(temp_path)
    except Exception as exc:
        os.remove(temp_path)
        log_operation(
            "upload_count_pages",
            "error",
            req_id=req_id,
            user_id=user_id,
            error=exc,
            context={"filename": file.filename},
        )
        raise HTTPException(status_code=400, detail="Could not read the uploaded PDF.")
    if num_pages > MAX_PDF_PAGES:
        os.remove(temp_path)
        raise HTTPException(
            status_code=413,
            detail=f"PDF has {num_pages} pages, exceeding the {MAX_PDF_PAGES}-page limit.",
        )

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


def _delete_paper_cascade(paper_id: str, user_id: str, req_id: str = "unknown") -> None:
    """Deletes the ChromaDB collection, BM25 cache entry, R2 Chroma snapshot,
    and every cached audit report for one paper. Caller has already deleted
    the R2 PDF and the registry row (kept separate so DELETE /papers/{id} can
    turn a failure there into a 502/500, while an account-deletion cascade
    should not stop
    partway through a user's library over one bad blob). Must be called with
    paper_locked(paper_id) held. Shared by DELETE /papers/{id} and the Clerk
    account-deletion webhook (POST /webhooks/clerk)."""
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
        (delete_novelty_report, "delete_novelty_report"),
        (delete_structure_report, "delete_structure_report"),
        (delete_numbers_report, "delete_numbers_report"),
        (delete_citation_gap_report, "delete_citation_gap_report"),
        (delete_overlap_report, "delete_overlap_report"),
        (delete_chroma_snapshot, "delete_chroma_snapshot"),
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

        _delete_paper_cascade(paper_id, user_id, req_id)

    release_paper_lock(paper_id)

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
        # Never interpolate the filename raw — Starlette encodes headers as
        # latin-1, so a CJK/Cyrillic name would raise UnicodeEncodeError and
        # 500 this route. See api/content_disposition.py.
        headers={"Content-Disposition": content_disposition(filename)},
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
        terms, stats = await run_on_executor(_extract, 40.0)
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
        queries, stats = await run_on_executor(_get_queries, 20.0)
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

    def _do_rewrite():
        # reset_stats / get_stats are thread-local and must run in the same
        # executor thread as the LLM call, so the token/cost numbers belong to
        # this request only.
        reset_stats()
        out = rewrite_text(request.text, request.mode)
        return out, get_stats()

    try:
        result, stats = await run_on_executor(_do_rewrite, 30.0)
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
            result = await run_on_executor(compare_papers, 120.0, req.question, paper_id_a, paper_id_b)
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
        result = await run_on_executor(_locked_query, 60.0)
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
#   2. We spawn the pipeline via api.concurrency.run_on_executor (it's sync
#      CPU/LLM work, run on the app's dedicated, explicitly-sized pool).
#   3. The executor passes a thread-safe on_progress callback that
#      drops {stage, message, ...} dicts onto an asyncio.Queue.
#   4. The SSE generator drains the queue, yielding each as a
#      `data: <json>\n\n` frame. The final frame's stage is "done"
#      (carrying the full result) or "error".

def _sse_format(event_type: str, payload: dict) -> str:
    """Encode one Server-Sent Event frame."""
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _finalize_stream_work(
    *,
    report: dict,
    stats: dict,
    persist,
    persist_op: str,
    failed_key: str,
    op: str,
    user_id: str,
    req_id: str,
    paper_id: str,
    duration_ms: int,
    log_context: dict,
) -> dict:
    """Cache a finished report and bill its LLM usage. Returns the report.

    **This must be called from the worker task, never from the SSE generator.**

    It used to live in the generator's `done` branch, which looked equivalent
    and was not. When a client disconnects mid-stream the generator is closed
    and never reaches that branch — but the worker task keeps running to
    completion. So an aborted audit burned the full LLM cost while recording no
    usage and writing no cache.

    That was also a quota bypass, not just waste: `enforce_audit_quota` counts
    rows in `usage_events`, so "start an audit, abort it, repeat" gave
    unlimited free-tier audits and unbounded provider spend. Keeping this on the
    worker side means the accounting happens whether or not anyone is still
    listening.
    """
    # A hard failure shouldn't be cached or billed — the user got nothing.
    if not report.get(failed_key):
        try:
            persist(paper_id, report)
        except Exception as exc:
            log_operation(
                persist_op, "error", req_id=req_id,
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
        op, "success", req_id=req_id, user_id=user_id,
        context=log_context,
        duration_ms=duration_ms,
    )
    report["request_id"] = req_id
    report["cached"] = False
    return report


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
@limiter.limit("30/minute")
async def query_stream(request: Request, req: QueryRequest, user_id: str = Depends(enforce_query_quota)):
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

            result = await run_on_executor(fn, timeout)
            # Log + bill here, on the worker, NOT in the SSE generator below.
            # A client that disconnects mid-query closes the generator before
            # its `done` branch ever runs, but this task still completes — so
            # doing it there meant an abandoned query cost real tokens and
            # recorded no usage, which `enforce_query_quota` counts. Same bug
            # (and same fix) as _finalize_stream_work for the audits.
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
            # Log the failure as a FAIL so it shows up in queries.jsonl. On the
            # worker, so an aborted stream still records the attempt.
            log_query(
                req_id=req_id,
                paper_id=log_paper_id,
                question=req.question,
                duration_ms=round((time.monotonic() - t0) * 1000),
                confidence=0,
                attempts=0,
                passed=False,
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
            # Log the failure as a FAIL so it shows up in queries.jsonl. On the
            # worker, so an aborted stream still records the attempt.
            log_query(
                req_id=req_id,
                paper_id=log_paper_id,
                question=req.question,
                duration_ms=round((time.monotonic() - t0) * 1000),
                confidence=0,
                attempts=0,
                passed=False,
            )
            await queue.put(("error", {"message": f"Pipeline failed: {exc}"}))

    track_task(run_pipeline())

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
                # The FAIL row for queries.jsonl is written by run_pipeline,
                # not here — a client that has already disconnected would
                # otherwise leave the failure unlogged.
                yield _sse_format("error", payload)
                return

            if kind == "done":
                # Logging + usage accounting already happened in the worker
                # (see run_pipeline) so an aborted stream cannot skip them.
                yield _sse_format("done", payload)
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
@limiter.limit("5/minute")
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

            report, stats = await run_on_executor(fn, 180.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_audit_report,
                persist_op="upload_audit_report",
                failed_key="audit_failed",
                op="audit_paper",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "claims_checked": report.get("claims_checked", 0),
                    "flagged": report.get("flagged", 0),
                },
            )
            await queue.put(("done", report))
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

    track_task(run_audit())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
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
@limiter.limit("5/minute")
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

            report, stats = await run_on_executor(fn, 180.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_review_report,
                persist_op="upload_review_report",
                failed_key="review_failed",
                op="review_paper",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "dimensions_checked": report.get("dimensions_checked", 0),
                    "weak": report.get("weak", 0),
                    "missing": report.get("missing", 0),
                },
            )
            await queue.put(("done", report))
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

    track_task(run_review())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Novelty / related-work scan — write-mode literature search (SSE)
# ─────────────────────────────────────────────────────────────────────────────
#
# Unlike the two audits above (which reason over the paper's OWN Chroma
# collection), this one searches *across the literature* via Semantic Scholar to
# surface the prior work closest to a draft. Same architecture regardless:
# progress pusher + SSE, R2-cached, re-served without LLM/network cost unless
# ?force=1 is passed. Shares the audit quota pool (kind="audit").

@app.post("/papers/{paper_id}/novelty/stream")
@limiter.limit("5/minute")
async def novelty_scan_stream(
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
        cached = get_novelty_report(paper_id)
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

    async def run_scan():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # scan's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = find_related_work(paper_id, on_progress=on_progress)
                    return report, get_stats()

            report, stats = await run_on_executor(fn, 180.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_novelty_report,
                persist_op="upload_novelty_report",
                failed_key="scan_failed",
                op="novelty_scan",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "candidates_found": report.get("candidates_found", 0),
                },
            )
            await queue.put(("done", report))
        except asyncio.TimeoutError:
            log_operation(
                "novelty_scan", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Novelty scan timed out — the literature search may be slow right now. Please try again.",
            }))
        except Exception as exc:
            log_operation(
                "novelty_scan", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Novelty scan failed: {exc}"}))

    track_task(run_scan())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Venue-fit / structure check — write-mode section-completeness check (SSE)
# ─────────────────────────────────────────────────────────────────────────────
#
# Sibling of the reviewer audit, but a presence/completeness check against a
# target venue's expected structure (Limitations, Ethics, Broader Impacts,
# Reproducibility, …) rather than a methodological-quality judgement. The report
# is venue-dependent, so the cache is keyed by paper but stores the venue inside
# it — a cached report is served only when its venue matches the request.

@app.get("/venues")
def get_venues():
    """Static list of target venues + labels for the structure-check selector."""
    return {"venues": list_venues()}


@app.post("/papers/{paper_id}/structure/stream")
@limiter.limit("5/minute")
async def structure_check_stream(
    paper_id: str,
    request: Request,
    venue: str = "generic",
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

    # Serve the cached report instantly — but only when it was computed for the
    # SAME target venue (a different venue means a different rubric → recompute).
    if not force:
        cached = get_structure_report(paper_id)
        if cached and cached.get("venue") == venue:
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

    async def run_check():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # check's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = check_structure(paper_id, venue, on_progress=on_progress)
                    return report, get_stats()

            report, stats = await run_on_executor(fn, 180.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_structure_report,
                persist_op="upload_structure_report",
                failed_key="structure_failed",
                op="structure_check",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "venue": venue,
                    "required_missing": report.get("required_missing", 0),
                    "missing": report.get("missing", 0),
                },
            )
            await queue.put(("done", report))
        except asyncio.TimeoutError:
            log_operation(
                "structure_check", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id, "venue": venue},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Structure check timed out. The draft may be unusually large — please try again.",
            }))
        except Exception as exc:
            log_operation(
                "structure_check", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id, "venue": venue},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Structure check failed: {exc}"}))

    track_task(run_check())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Numbers-consistency check — write-mode abstract↔results reconciliation (SSE)
# ─────────────────────────────────────────────────────────────────────────────
#
# Cousin of the claim audit: instead of "are the qualitative claims grounded?",
# this asks "do the exact figures in the abstract/intro match the results tables?"
# — catching stale/copy-paste/transcription number errors. Same architecture:
# progress pusher + SSE, R2-cached, re-served without LLM cost unless ?force=1.

@app.post("/papers/{paper_id}/numbers/stream")
@limiter.limit("5/minute")
async def numbers_check_stream(
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
        cached = get_numbers_report(paper_id)
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

    async def run_numbers():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # check's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = audit_numbers(paper_id, on_progress=on_progress)
                    return report, get_stats()

            report, stats = await run_on_executor(fn, 180.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_numbers_report,
                persist_op="upload_numbers_report",
                failed_key="audit_failed",
                op="numbers_check",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "numbers_checked": report.get("numbers_checked", 0),
                    "mismatched": report.get("mismatched", 0),
                },
            )
            await queue.put(("done", report))
        except asyncio.TimeoutError:
            log_operation(
                "numbers_check", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Numbers check timed out. The paper may be unusually large — please try again.",
            }))
        except Exception as exc:
            log_operation(
                "numbers_check", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Numbers check failed: {exc}"}))

    track_task(run_numbers())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Citation gap check — write-mode "[citation needed]" pass (SSE)
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirror image of the claim audit: instead of checking the draft's claims against
# its own evidence, it finds statements that assert something about the OUTSIDE
# world with no citation, and (for a bounded sample) asks Semantic Scholar
# whether a plausible missing reference exists. Closest sibling of the novelty
# scan — external-API-backed, cacheable, no venue parameter — so it clones that
# endpoint's shape, including the 180s timeout for the S2 + LLM slow path.
# Shares the audit quota pool (kind="audit").

@app.post("/papers/{paper_id}/citation-gaps/stream")
@limiter.limit("5/minute")
async def citation_gap_stream(
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
        cached = get_citation_gap_report(paper_id)
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

    async def run_check():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # check's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = audit_citation_gaps(paper_id, on_progress=on_progress)
                    return report, get_stats()

            report, stats = await run_on_executor(fn, 180.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_citation_gap_report,
                persist_op="upload_citation_gap_report",
                failed_key="audit_failed",
                op="citation_gap_check",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "gaps_found": report.get("gaps_found", 0),
                },
            )
            await queue.put(("done", report))
        except asyncio.TimeoutError:
            log_operation(
                "citation_gap_check", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Citation gap check timed out — the literature search may be slow right now. Please try again.",
            }))
        except Exception as exc:
            log_operation(
                "citation_gap_check", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Citation gap check failed: {exc}"}))

    track_task(run_check())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Overlap check (write mode) ───────────────────────────────────────────────
# Streams the plagiarism/overlap check: passages of the draft that also appear
# in the other papers in the user's library. Purely local (Chroma + one gating
# LLM call), so it clones the claim-audit shape rather than the novelty scan's.
#
# The one structural difference from every sibling endpoint: this check needs a
# CORPUS, so the set of comparable papers is resolved from Postgres HERE and
# passed into the engine. Doing the lookup in the endpoint is what keeps
# plagiarism_auditor free of any Postgres/tenancy coupling, like the others —
# and it also puts the security boundary in one place: a user can only ever be
# compared against papers they can already read.
# Shares the audit quota pool (kind="audit").

def _overlap_corpus(user_id: str, paper_id: str) -> list[dict]:
    """The papers this draft may be compared against: the user's own library
    plus the shared demo set, minus the draft itself and anything not ready.

    Demo papers are included deliberately — a new account with one draft and an
    empty library would otherwise get an empty check with nothing to show."""
    seen: set[str] = {paper_id}
    corpus: list[dict] = []
    for row in list(list_papers(user_id)) + list(list_demo_papers()):
        pid = row["paper_id"]
        if pid in seen or row.get("status") != "ready":
            continue
        seen.add(pid)
        name = (row.get("filename") or "Untitled").rsplit(".pdf", 1)[0]
        corpus.append({"paper_id": pid, "title": name})
    return corpus


@app.post("/papers/{paper_id}/overlap/stream")
@limiter.limit("5/minute")
async def overlap_stream(
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
        cached = get_overlap_report(paper_id)
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

    corpus = _overlap_corpus(user_id, paper_id)
    on_progress = _make_progress_pusher(queue, loop)

    async def run_check():
        try:
            def fn():
                # reset_stats/get_stats are thread-local — run them in the same
                # executor thread as the LLM work so the usage numbers are this
                # check's only. Lock the paper like a query so a concurrent
                # delete/query can't pull Chroma state out from under us.
                with paper_locked(paper_id):
                    reset_stats()
                    report = audit_overlap(paper_id, corpus, on_progress=on_progress)
                    return report, get_stats()

            # Longer timeout than the local-only siblings: the cost scales with
            # the size of the library, since every corpus paper is read out of
            # Chroma and indexed.
            report, stats = await run_on_executor(fn, 240.0)
            _finalize_stream_work(
                report=report,
                stats=stats,
                persist=upload_overlap_report,
                persist_op="upload_overlap_report",
                failed_key="audit_failed",
                op="overlap_check",
                user_id=user_id,
                req_id=req_id,
                paper_id=paper_id,
                duration_ms=round((time.monotonic() - t0) * 1000),
                log_context={
                    "paper_id": paper_id,
                    "corpus": len(corpus),
                    "matches_found": report.get("matches_found", 0),
                },
            )
            await queue.put(("done", report))
        except asyncio.TimeoutError:
            log_operation(
                "overlap_check", "error", req_id=req_id, user_id=user_id,
                error="timeout", context={"paper_id": paper_id, "corpus": len(corpus)},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {
                "message": "Overlap check timed out — your library may be large. Please try again.",
            }))
        except Exception as exc:
            log_operation(
                "overlap_check", "error", req_id=req_id, user_id=user_id,
                error=exc, context={"paper_id": paper_id},
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
            await queue.put(("error", {"message": f"Overlap check failed: {exc}"}))

    track_task(run_check())

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
                # Caching + usage accounting already happened in the
                # worker (see _finalize_stream_work) so an aborted
                # stream cannot skip them. Nothing to do but emit.
                yield _sse_format("done", payload)
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Request
from pydantic import BaseModel

from discovery.search import search_papers
from discovery.fetcher import download_paper
from discovery.url_guard import UnsafeUrl
from api.auth import get_current_user_id, limiter
from api.logger import log_operation
from api.usage import enforce_paper_quota
from api.ingestion_runner import run_ingestion_from_storage

router = APIRouter(prefix="/discovery", tags=["discovery"])


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


class ImportRequest(BaseModel):
    title: str
    pdf_url: str
    source_id: str
    authors: list[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None


@router.post("/search")
async def search(request: SearchRequest, _user_id: str = Depends(get_current_user_id)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = await search_papers(request.query, limit=min(request.limit, 40))
    return {"results": results, "total": len(results)}


@router.post("/import")
@limiter.limit("5/minute")
async def import_paper(
    request: Request,
    body: ImportRequest,
    background_tasks: BackgroundTasks,
    # Same paper cap as a direct /upload — importing also runs the (LLM-heavy)
    # ingestion pipeline, so it must count against the free-tier paper limit
    # rather than offering a quota-free side door to it.
    user_id: str = Depends(enforce_paper_quota),
):
    if not body.pdf_url:
        raise HTTPException(status_code=400, detail="No PDF URL provided.")
    try:
        paper_id = await download_paper(body.pdf_url, body.title, user_id, source_id=body.source_id)
    except UnsafeUrl as exc:
        # The guard's own messages are written to be safe to show: they say the
        # URL was refused without revealing anything about what is reachable.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Deliberately NOT `f"...: {exc}"`. Echoing the underlying error turned
        # this endpoint into an internal port scanner: "connection refused" vs.
        # "timeout" vs. "did not return a PDF" tells an attacker exactly which
        # internal hosts and ports are live. Detail goes to the logs instead.
        log_operation(
            "discovery_import",
            "error",
            user_id=user_id,
            error=exc,
            context={"pdf_url": body.pdf_url[:200]},
        )
        raise HTTPException(
            status_code=502,
            detail="Could not download that PDF. Check the link and try again.",
        )
    background_tasks.add_task(run_ingestion_from_storage, paper_id)
    return {
        "paper_id": paper_id,
        "status": "processing",
        "filename": body.title[:80] + ".pdf",
    }

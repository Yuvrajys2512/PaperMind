from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel

from discovery.search import search_papers
from discovery.fetcher import download_paper
from api.auth import get_current_user_id
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
async def import_paper(
    request: ImportRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    if not request.pdf_url:
        raise HTTPException(status_code=400, detail="No PDF URL provided.")
    try:
        paper_id = await download_paper(request.pdf_url, request.title, user_id, source_id=request.source_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not download PDF: {exc}")
    background_tasks.add_task(run_ingestion_from_storage, paper_id)
    return {
        "paper_id": paper_id,
        "status": "processing",
        "filename": request.title[:80] + ".pdf",
    }

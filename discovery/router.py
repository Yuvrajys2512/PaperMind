from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from discovery.search import search_papers
from discovery.fetcher import download_paper
from api.storage import update_paper_status, get_paper_pdf_path
from ingestion.ingest_document import ingest_document

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


def _run_ingestion(paper_id: str):
    pdf_path = str(get_paper_pdf_path(paper_id))
    result = ingest_document(pdf_path=pdf_path, paper_name=paper_id)
    if result["success"]:
        update_paper_status(paper_id, "ready")
    else:
        update_paper_status(paper_id, "failed", error=result.get("error"))


@router.post("/search")
async def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = await search_papers(request.query, limit=min(request.limit, 40))
    return {"results": results, "total": len(results)}


@router.post("/import")
async def import_paper(request: ImportRequest, background_tasks: BackgroundTasks):
    if not request.pdf_url:
        raise HTTPException(status_code=400, detail="No PDF URL provided.")
    try:
        paper_id = await download_paper(request.pdf_url, request.title)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not download PDF: {exc}")
    background_tasks.add_task(_run_ingestion, paper_id)
    return {
        "paper_id": paper_id,
        "status": "processing",
        "filename": request.title[:80] + ".pdf",
    }

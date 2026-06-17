import os

from api.storage import download_pdf_to_tempfile, get_paper, update_paper_status
from api.usage import record_usage
from ingestion.ingest_document import ingest_document
from ingestion.llm_client import get_stats, reset_stats


def run_ingestion_from_storage(paper_id: str):
    """Downloads paper_id's PDF from R2 into a temp file, ingests it, and
    updates its status. Shared by the direct-upload and discovery-import
    flows so neither duplicates the download/cleanup logic."""
    reset_stats()
    temp_path = download_pdf_to_tempfile(paper_id)
    try:
        result = ingest_document(pdf_path=temp_path, paper_name=paper_id)
        if result["success"]:
            update_paper_status(paper_id, "ready")
        else:
            update_paper_status(paper_id, "failed", error=result["error"])
    finally:
        os.remove(temp_path)

    stats = get_stats()
    paper = get_paper(paper_id)
    if paper:
        record_usage(
            user_id=paper["user_id"],
            kind="upload",
            paper_id=paper_id,
            llm_calls=stats["call_count"],
            tokens_in=stats["tokens_in"],
            tokens_out=stats["tokens_out"],
            cost_usd=stats["cost_usd"],
        )

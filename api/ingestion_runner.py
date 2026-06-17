import os

from api.storage import download_pdf_to_tempfile, update_paper_status
from ingestion.ingest_document import ingest_document


def run_ingestion_from_storage(paper_id: str):
    """Downloads paper_id's PDF from R2 into a temp file, ingests it, and
    updates its status. Shared by the direct-upload and discovery-import
    flows so neither duplicates the download/cleanup logic."""
    temp_path = download_pdf_to_tempfile(paper_id)
    try:
        result = ingest_document(pdf_path=temp_path, paper_name=paper_id)
        if result["success"]:
            update_paper_status(paper_id, "ready")
        else:
            update_paper_status(paper_id, "failed", error=result["error"])
    finally:
        os.remove(temp_path)

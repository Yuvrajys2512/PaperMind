import os
import tempfile

import httpx
from api.storage import create_paper_record, update_paper_status, upload_pdf
from api.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, PDF_MAGIC

_TIMEOUT = 45.0
_HEADERS = {
    "User-Agent": "Mozilla/5.0 PaperMind/1.0 (research tool)",
    "Accept": "application/pdf,application/octet-stream,*/*",
}


async def download_paper(pdf_url: str, title: str, user_id: str, source_id: str = None) -> str:
    """Download a PDF from pdf_url, store it in R2, register it, and return the new paper_id."""
    safe_name = (title[:80].strip() or "paper") + ".pdf"

    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    total = 0
    first = True
    try:
        with os.fdopen(fd, "wb") as f:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_TIMEOUT,
                headers=_HEADERS,
            ) as client:
                async with client.stream("GET", pdf_url) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        # A URL can return non-PDF or arbitrarily large content —
                        # validate the magic number and enforce the size cap, same
                        # as the direct-upload path.
                        if first:
                            if not chunk.startswith(PDF_MAGIC):
                                raise ValueError("URL did not return a PDF.")
                            first = False
                        total += len(chunk)
                        if total > MAX_UPLOAD_BYTES:
                            raise ValueError(f"PDF exceeds the {MAX_UPLOAD_MB} MB limit.")
                        f.write(chunk)
        if first:
            raise ValueError("Downloaded file was empty.")

        # Only create the registry row once the download has actually
        # succeeded — creating it beforehand left orphaned "processing"
        # rows on download failure with nothing to clean them up.
        paper_id = create_paper_record(safe_name, user_id, source_id=source_id)
        try:
            upload_pdf(paper_id, temp_path)
        except Exception as exc:
            update_paper_status(paper_id, "failed", error=f"Upload to storage failed: {exc}")
            raise
    finally:
        os.remove(temp_path)

    return paper_id

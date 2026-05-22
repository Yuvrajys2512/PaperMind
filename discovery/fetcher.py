import httpx
from api.storage import create_paper_record, get_paper_pdf_path

_TIMEOUT = 45.0
_HEADERS = {
    "User-Agent": "Mozilla/5.0 PaperMind/1.0 (research tool)",
    "Accept": "application/pdf,application/octet-stream,*/*",
}


async def download_paper(pdf_url: str, title: str, source_id: str = None) -> str:
    """Download a PDF from pdf_url, register it, and return the new paper_id."""
    safe_name = (title[:80].strip() or "paper") + ".pdf"
    paper_id = create_paper_record(safe_name, source_id=source_id)
    pdf_path = get_paper_pdf_path(paper_id)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_TIMEOUT,
        headers=_HEADERS,
    ) as client:
        async with client.stream("GET", pdf_url) as resp:
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

    return paper_id

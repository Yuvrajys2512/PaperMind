import os
import tempfile

import httpx
from api.storage import create_paper_record, update_paper_status, upload_pdf
from api.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, PDF_MAGIC
from discovery.url_guard import UnsafeUrl, validate_pdf_url

_TIMEOUT = 45.0
_HEADERS = {
    "User-Agent": "Mozilla/5.0 PaperMind/1.0 (research tool)",
    "Accept": "application/pdf,application/octet-stream,*/*",
}

# Redirects are followed by hand (see below), so this bounds the chain the way
# httpx's own follow_redirects would have.
_MAX_REDIRECTS = 5


async def download_paper(pdf_url: str, title: str, user_id: str, source_id: str = None) -> str:
    """Download a PDF from pdf_url, store it in R2, register it, and return the new paper_id.

    `pdf_url` is user-supplied, so every hop is vetted by discovery.url_guard
    before the server will connect to it — see that module for why this is a
    deny-list on private address space rather than a host allow-list.
    """
    safe_name = (title[:80].strip() or "paper") + ".pdf"

    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    total = 0
    first = True
    try:
        with os.fdopen(fd, "wb") as f:
            # follow_redirects=False is load-bearing, not a style choice. A
            # public URL that 302s to http://169.254.169.254/ is the standard
            # way past a naive URL check, so each hop is re-validated here
            # before we connect to it.
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=_TIMEOUT,
                headers=_HEADERS,
            ) as client:
                target = validate_pdf_url(pdf_url)

                for _hop in range(_MAX_REDIRECTS + 1):
                    async with client.stream("GET", target) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                raise UnsafeUrl("Redirect without a destination.")
                            # Resolve relative Locations against the current URL,
                            # then re-run the full guard on the result.
                            target = validate_pdf_url(
                                str(resp.url.join(location))
                            )
                            continue

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
                        break
                else:
                    raise UnsafeUrl("Too many redirects.")
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

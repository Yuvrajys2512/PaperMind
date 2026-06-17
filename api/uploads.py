"""
api/uploads.py

Server-side PDF upload validation, shared by the direct-upload route
(api/main.py) and the discovery-import download (discovery/fetcher.py).

A `.pdf` extension proves nothing — anyone can rename a file or point an
import URL at arbitrary content. We enforce two cheap, real checks:
  * the bytes actually start with the PDF magic number, and
  * the file stays under a size cap.
"""

import os

MAX_UPLOAD_MB = int(os.getenv("PAPERMIND_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

PDF_MAGIC = b"%PDF-"


class UploadTooLarge(Exception):
    """Raised when an upload exceeds MAX_UPLOAD_BYTES."""


class NotAPdf(Exception):
    """Raised when the leading bytes aren't a PDF header."""


def check_pdf_header(first_bytes: bytes) -> None:
    """Raise NotAPdf unless `first_bytes` begins with the PDF magic number."""
    if not first_bytes.startswith(PDF_MAGIC):
        raise NotAPdf("File does not look like a PDF (missing %PDF- header).")

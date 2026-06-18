"""
Unit tests for api/uploads.py — the server-side PDF upload guard.

A .pdf extension proves nothing, so the upload path checks the actual leading
bytes against the PDF magic number. These tests pin that check: a real header
passes, anything else (renamed file, HTML, empty body, header not at offset 0)
is rejected with NotAPdf.
"""

import pytest

from api import uploads
from api.uploads import check_pdf_header, NotAPdf, PDF_MAGIC


def test_valid_pdf_header_passes():
    # A real PDF starts with "%PDF-1.x"; the leading bytes are enough.
    check_pdf_header(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3")  # must not raise


def test_exact_magic_only_passes():
    # The bare magic number with nothing after it is still a valid header.
    check_pdf_header(PDF_MAGIC)


def test_non_pdf_bytes_rejected():
    with pytest.raises(NotAPdf):
        check_pdf_header(b"<!DOCTYPE html><html>")


def test_empty_bytes_rejected():
    with pytest.raises(NotAPdf):
        check_pdf_header(b"")


def test_magic_not_at_start_rejected():
    # The header must be at offset 0 — a PDF magic buried later doesn't count
    # (this is exactly the polyglot/renamed-file case the check defends against).
    with pytest.raises(NotAPdf):
        check_pdf_header(b"GIF89a%PDF-")


def test_size_cap_is_positive_and_consistent():
    # MAX_UPLOAD_BYTES is derived from MAX_UPLOAD_MB; keep them in lockstep.
    assert uploads.MAX_UPLOAD_BYTES == uploads.MAX_UPLOAD_MB * 1024 * 1024
    assert uploads.MAX_UPLOAD_BYTES > 0

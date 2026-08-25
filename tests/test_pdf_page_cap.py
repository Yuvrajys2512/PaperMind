"""
Regression tests for Launch Checklist 2.6 — no PDF page cap.

`ingestion.pdf_parser` only imports pdfplumber/re/statistics at module level
(no torch, no docling, no live services), so it's safe to import directly.
"""

import pytest

from ingestion import pdf_parser


class _FakePdf:
    def __init__(self, num_pages):
        self.pages = [object()] * num_pages

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_count_pdf_pages_reports_page_count(monkeypatch):
    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda path: _FakePdf(5))
    assert pdf_parser.count_pdf_pages("fake.pdf") == 5


def test_extract_text_from_pdf_rejects_over_the_cap(monkeypatch):
    monkeypatch.setattr(
        pdf_parser.pdfplumber, "open",
        lambda path: _FakePdf(pdf_parser.MAX_PDF_PAGES + 1),
    )
    with pytest.raises(pdf_parser.PDFTooLongError):
        pdf_parser.extract_text_from_pdf("fake.pdf")


def test_extract_text_from_pdf_error_names_the_limit(monkeypatch):
    monkeypatch.setattr(
        pdf_parser.pdfplumber, "open",
        lambda path: _FakePdf(pdf_parser.MAX_PDF_PAGES + 50),
    )
    with pytest.raises(pdf_parser.PDFTooLongError, match=str(pdf_parser.MAX_PDF_PAGES)):
        pdf_parser.extract_text_from_pdf("fake.pdf")


def test_at_the_cap_is_not_rejected(monkeypatch):
    """The over-length check must reject only PDFs strictly over the cap —
    a paper landing exactly on MAX_PDF_PAGES should still get a chance to
    parse (it'll fail downstream for unrelated reasons in this fake, which
    is fine — we're only asserting the cap check itself doesn't fire)."""
    monkeypatch.setattr(
        pdf_parser.pdfplumber, "open",
        lambda path: _FakePdf(pdf_parser.MAX_PDF_PAGES),
    )
    try:
        pdf_parser.extract_text_from_pdf("fake.pdf")
    except pdf_parser.PDFTooLongError:
        pytest.fail("a PDF exactly at MAX_PDF_PAGES must not be rejected as too long")
    except Exception:
        pass  # the fake page objects don't support real parsing — expected

"""
Unit tests for api.storage.download_pdf_to_tempfile — temp-file hygiene.

`mkstemp` creates the file before the download runs, and a raising download
never returns the path. The caller (api/ingestion_runner) does clean up, but it
tracks `temp_path = None` until the call returns — so on failure it has nothing
to delete and the file stays in the temp directory. On a long-lived host every
failed ingestion (missing R2 key, network blip) leaked one file.
"""

import glob
import os
import tempfile

import pytest

from api import storage


def _temp_pdfs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))


def test_successful_download_returns_a_real_path(monkeypatch):
    def fake_download(bucket, key, path):
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4 fake")

    monkeypatch.setattr(storage._s3, "download_file", fake_download)

    before = _temp_pdfs()
    path = storage.download_pdf_to_tempfile("paper-123")
    try:
        assert os.path.exists(path)
        assert open(path, "rb").read().startswith(b"%PDF")
        # Caller owns cleanup on the success path.
        assert path not in before
    finally:
        os.remove(path)


def test_failed_download_leaves_no_temp_file_behind(monkeypatch):
    def boom(bucket, key, path):
        assert os.path.exists(path), "sanity: mkstemp created the file before we raised"
        raise RuntimeError("NoSuchKey")

    monkeypatch.setattr(storage._s3, "download_file", boom)

    before = _temp_pdfs()
    with pytest.raises(RuntimeError, match="NoSuchKey"):
        storage.download_pdf_to_tempfile("missing-paper")
    assert _temp_pdfs() == before, "a failed download must not leak a temp file"


def test_the_original_error_is_raised_not_the_cleanup_error(monkeypatch):
    """Cleanup is best-effort: if removing the temp file also fails, the caller
    must still see the download failure, which is the actionable one."""
    def boom(bucket, key, path):
        raise RuntimeError("NoSuchKey")

    def bad_remove(path):
        raise OSError("file is locked")

    monkeypatch.setattr(storage._s3, "download_file", boom)
    monkeypatch.setattr(storage.os, "remove", bad_remove)

    with pytest.raises(RuntimeError, match="NoSuchKey"):
        storage.download_pdf_to_tempfile("missing-paper")

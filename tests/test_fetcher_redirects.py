"""
Redirect-hop tests for discovery/fetcher.download_paper.

Validating only the URL the client submitted is the classic SSRF half-fix: an
attacker registers a perfectly ordinary public host and has it 302 to
http://169.254.169.254/ or http://127.0.0.1/. httpx's `follow_redirects=True`
would chase that without ever consulting the guard again, which is why the
fetcher follows redirects by hand and re-validates every hop.

These tests drive the real fetcher through an httpx MockTransport, so they
exercise the actual hop loop rather than a reimplementation of it.

Storage is stubbed per-test on the `discovery.fetcher` module namespace, NOT via
sys.modules. An earlier version of this file installed a fake `api.storage` into
sys.modules guarded by `if "api.storage" not in sys.modules` — which silently
did nothing, because pytest imports every test module during collection and
`tests/test_storage_tempfiles.py` does `from api import storage` at module
level. The real module was therefore already loaded, the stub was skipped, and
these tests wrote live rows to Neon and live objects to R2.

`fetcher` does `from api.storage import create_paper_record, ...`, so those
names are bound in the fetcher module's own namespace and monkeypatching them
there is both sufficient and order-independent.
"""

import asyncio

import httpx
import pytest


@pytest.fixture
def fetcher(monkeypatch):
    """discovery.fetcher with every storage call replaced by a fake.

    The fakes raise rather than no-op on anything unexpected: a test that
    reaches real storage must fail loudly, not quietly mutate production.
    """
    import discovery.fetcher as mod

    def _refuse(*_a, **_k):
        raise AssertionError(
            "a test reached real storage — the fetcher fixture is not patching "
            "everything it needs to"
        )

    monkeypatch.setattr(mod, "create_paper_record", lambda *a, **k: "paper-123")
    monkeypatch.setattr(mod, "update_paper_status", lambda *a, **k: None)
    monkeypatch.setattr(mod, "upload_pdf", lambda *a, **k: None)
    # Anything else storage-shaped that gets added later trips the guard.
    for name in dir(mod):
        obj = getattr(mod, name)
        if callable(obj) and name.startswith(("get_", "delete_")) and "pdf" in name:
            monkeypatch.setattr(mod, name, _refuse)
    return mod


PDF_BODY = b"%PDF-1.7\n" + b"x" * 512


def _run(coro):
    """Drive one coroutine to completion.

    Deliberately not pytest-asyncio: CI installs a minimal package set on
    purpose, and these tests don't need an event-loop fixture.
    """
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Make every hostname in these tests resolve to a public address.

    Without this the guard rejects `papers.example` simply because it does not
    resolve — and every test here would pass for the wrong reason, proving
    nothing about redirect handling.
    """
    from discovery import url_guard
    monkeypatch.setattr(url_guard, "_resolve_all", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(url_guard, "_ALLOWLIST", ())


# Captured before any patching. `fetcher.httpx` IS the httpx module, so
# monkeypatching `fetcher.httpx.AsyncClient` replaces it globally — including
# for the factory below, which would then recurse into itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _client_factory(handler):
    """A drop-in for httpx.AsyncClient that routes through MockTransport."""
    def _make(**kwargs):
        # The fetcher passes follow_redirects=False; MockTransport doesn't care,
        # and the hop loop under test is the fetcher's own.
        kwargs.pop("follow_redirects", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
    return _make


async def _download(fetcher, monkeypatch, handler, url):
    monkeypatch.setattr(fetcher.httpx, "AsyncClient", _client_factory(handler))
    return await fetcher.download_paper(url, "Test Paper", "user_1")


def test_redirect_to_link_local_metadata_is_refused(fetcher, monkeypatch):
    """The AWS/GCP metadata endpoint, reached via a public-looking front door."""
    def handler(request):
        if request.url.host == "papers.example":
            return httpx.Response(302, headers={"location": "https://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, content=PDF_BODY)

    with pytest.raises(fetcher.UnsafeUrl):
        _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))


def test_redirect_to_loopback_is_refused(fetcher, monkeypatch):
    """A redirect at the app's own API — the way to reach admin routes."""
    def handler(request):
        if request.url.host == "papers.example":
            return httpx.Response(302, headers={"location": "https://127.0.0.1:443/admin/usage"})
        return httpx.Response(200, content=PDF_BODY)

    with pytest.raises(fetcher.UnsafeUrl):
        _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))


def test_redirect_downgrading_to_http_is_refused(fetcher, monkeypatch):
    def handler(request):
        if request.url.scheme == "https":
            return httpx.Response(302, headers={"location": "http://papers.example/p.pdf"})
        return httpx.Response(200, content=PDF_BODY)

    with pytest.raises(fetcher.UnsafeUrl):
        _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))


def test_redirect_chain_is_bounded(fetcher, monkeypatch):
    """An endless public→public redirect loop must terminate, not hang."""
    def handler(request):
        return httpx.Response(302, headers={"location": "https://papers.example/next.pdf"})

    with pytest.raises(fetcher.UnsafeUrl):
        _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))


def test_relative_redirect_is_resolved_and_still_checked(fetcher, monkeypatch):
    """A relative Location must resolve against the current URL — and the result
    still goes through the guard."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/p.pdf":
            return httpx.Response(302, headers={"location": "/real/paper.pdf"})
        return httpx.Response(200, content=PDF_BODY)

    paper_id = _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))
    assert paper_id == "paper-123"
    assert seen[-1] == "https://papers.example/real/paper.pdf"


def test_ordinary_public_redirect_still_works(fetcher, monkeypatch):
    """The guard must not break the common case — publishers redirect constantly."""
    def handler(request):
        if request.url.host == "papers.example":
            return httpx.Response(302, headers={"location": "https://cdn.example/paper.pdf"})
        return httpx.Response(200, content=PDF_BODY)

    assert _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf")) == "paper-123"


def test_non_pdf_body_is_still_rejected(fetcher, monkeypatch):
    """The magic-number check must survive the rewrite to manual redirects."""
    def handler(request):
        return httpx.Response(200, content=b"<html>not a pdf</html>")

    with pytest.raises(ValueError):
        _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))


def test_oversized_body_is_still_rejected(fetcher, monkeypatch):
    """So must the size cap."""
    monkeypatch.setattr(fetcher, "MAX_UPLOAD_BYTES", 1024)

    def handler(request):
        return httpx.Response(200, content=b"%PDF-1.7\n" + b"x" * 5000)

    with pytest.raises(ValueError):
        _run(_download(fetcher, monkeypatch, handler, "https://papers.example/p.pdf"))

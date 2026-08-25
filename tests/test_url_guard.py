"""
Unit tests for discovery/url_guard.py — the SSRF gate on user-supplied PDF URLs.

`POST /discovery/import` makes the SERVER fetch a URL the client chose, so this
is a security boundary: it must refuse anything that resolves inside the
deployment (cloud metadata, localhost, RFC1918) while still allowing the
arbitrary public publisher hosts that Semantic Scholar's openAccessPdf returns.
"""

import pytest

from discovery import url_guard
from discovery.url_guard import UnsafeUrl, validate_pdf_url


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Resolve every hostname to a public address by default, so a test that is
    about *scheme* or *port* isn't accidentally passing because DNS failed.
    Tests that care about resolution override this."""
    monkeypatch.setattr(url_guard, "_resolve_all", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(url_guard, "_ALLOWLIST", ())


# ── The addresses that matter ────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://169.254.169.254/latest/meta-data/",   # AWS/GCP/Azure metadata
    "https://127.0.0.1/admin",                     # loopback
    "https://localhost/admin",                     # loopback by name
    "https://10.0.0.5/internal",                   # RFC1918
    "https://192.168.1.1/router",                  # RFC1918
    "https://172.16.0.1/internal",                 # RFC1918
    "https://[::1]/admin",                         # IPv6 loopback
    "https://[fd00::1]/internal",                  # IPv6 unique-local
    "https://0.0.0.0/",                            # unspecified
])
def test_internal_addresses_are_refused(url, monkeypatch):
    # Literal IPs skip DNS; 'localhost' must be refused via resolution too.
    monkeypatch.setattr(url_guard, "_resolve_all", lambda host: ["127.0.0.1"])
    with pytest.raises(UnsafeUrl):
        validate_pdf_url(url)


def test_public_hostname_is_allowed():
    assert validate_pdf_url("https://arxiv.org/pdf/1706.03762.pdf")


def test_arbitrary_public_publisher_host_is_allowed():
    """The S2 case — an allow-list would have broken this, which is why the
    guard blocks address ranges instead of vetting hostnames."""
    assert validate_pdf_url("https://aclanthology.org/2020.acl-main.703.pdf")


def test_hostname_resolving_to_private_is_refused(monkeypatch):
    """The interesting case: a perfectly ordinary-looking public hostname whose
    DNS record points inside. Name-based checks miss this entirely."""
    monkeypatch.setattr(url_guard, "_resolve_all", lambda host: ["10.1.2.3"])
    with pytest.raises(UnsafeUrl):
        validate_pdf_url("https://totally-normal-papers.example/paper.pdf")


def test_any_private_address_in_a_multi_record_answer_is_refused(monkeypatch):
    """A host that resolves to both a public and a private address must be
    refused — otherwise which one httpx picks decides whether we're safe."""
    monkeypatch.setattr(
        url_guard, "_resolve_all", lambda host: ["93.184.216.34", "127.0.0.1"]
    )
    with pytest.raises(UnsafeUrl):
        validate_pdf_url("https://mixed.example/paper.pdf")


def test_unresolvable_host_is_refused(monkeypatch):
    def _boom(host):
        raise UnsafeUrl("Could not resolve the host for that URL.")
    monkeypatch.setattr(url_guard, "_resolve_all", _boom)
    with pytest.raises(UnsafeUrl):
        validate_pdf_url("https://nope.invalid/paper.pdf")


# ── Scheme, credentials, ports ───────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://arxiv.org/pdf/x.pdf",      # plaintext
    "file:///etc/passwd",              # local file
    "gopher://arxiv.org/",             # protocol smuggling
    "ftp://arxiv.org/x.pdf",
])
def test_non_https_schemes_are_refused(url):
    with pytest.raises(UnsafeUrl):
        validate_pdf_url(url)


def test_embedded_credentials_are_refused():
    with pytest.raises(UnsafeUrl):
        validate_pdf_url("https://user:pass@arxiv.org/pdf/x.pdf")


@pytest.mark.parametrize("url", [
    "https://arxiv.org:8000/pdf/x.pdf",   # the app's own dev port
    "https://arxiv.org:7860/pdf/x.pdf",   # the app's own prod port
    "https://arxiv.org:22/pdf/x.pdf",
    "https://arxiv.org:6379/pdf/x.pdf",
])
def test_non_standard_ports_are_refused(url):
    with pytest.raises(UnsafeUrl):
        validate_pdf_url(url)


def test_explicit_443_is_allowed():
    assert validate_pdf_url("https://arxiv.org:443/pdf/x.pdf")


@pytest.mark.parametrize("url", ["", "   ", "not a url", "https://"])
def test_junk_input_is_refused(url):
    with pytest.raises(UnsafeUrl):
        validate_pdf_url(url)


# ── Optional strict mode ─────────────────────────────────────────────────────

def test_allowlist_restricts_hosts_when_set(monkeypatch):
    monkeypatch.setattr(url_guard, "_ALLOWLIST", ("arxiv.org",))
    assert validate_pdf_url("https://arxiv.org/pdf/x.pdf")
    assert validate_pdf_url("https://export.arxiv.org/pdf/x.pdf")  # subdomain
    with pytest.raises(UnsafeUrl):
        validate_pdf_url("https://elsewhere.example/x.pdf")


def test_allowlist_does_not_match_on_suffix_confusion(monkeypatch):
    """'evil-arxiv.org' must not pass an 'arxiv.org' allow-list."""
    monkeypatch.setattr(url_guard, "_ALLOWLIST", ("arxiv.org",))
    with pytest.raises(UnsafeUrl):
        validate_pdf_url("https://evil-arxiv.org/x.pdf")


# ── Error messages must not leak reachability ────────────────────────────────

def test_messages_do_not_reveal_internal_detail(monkeypatch):
    """The whole point of the generic 502 in the router is that an attacker
    can't tell 'refused' from 'timed out'. The guard's own messages must not
    reintroduce that signal by naming the address it found."""
    monkeypatch.setattr(url_guard, "_resolve_all", lambda host: ["10.1.2.3"])
    with pytest.raises(UnsafeUrl) as exc:
        validate_pdf_url("https://internal.example/paper.pdf")
    assert "10.1.2.3" not in str(exc.value)

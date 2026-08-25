"""
discovery/url_guard.py — SSRF protection for user-supplied PDF URLs.

`POST /discovery/import` takes a `pdf_url` straight from the client and the
SERVER fetches it. Without a guard that is a textbook SSRF: any signed-in user
can point it at cloud metadata (169.254.169.254), at the app's own admin routes
on localhost, or at anything else reachable from inside the deployment.

Why a deny-list and not a host allow-list
-----------------------------------------
The obvious fix — "only allow arxiv.org" — would break the feature. arXiv URLs
are predictable, but Semantic Scholar's `openAccessPdf.url` legitimately points
at arbitrary publisher and university repositories (ACL, PMC, bioRxiv, one-off
institutional hosts). An allow-list would silently reduce S2 imports to zero.

So the default posture is: any public host is fine, no private/internal address
ever is. That is the standard SSRF mitigation and it preserves the feature.
Operators who want the stricter posture can set PAPERMIND_PDF_HOST_ALLOWLIST to
a comma-separated suffix list (e.g. "arxiv.org,aclanthology.org") and only those
hosts will be reachable.

What is enforced
----------------
  * scheme must be https (http would also leak the request in plaintext)
  * no credentials embedded in the URL (https://user:pass@host)
  * explicit ports other than 443 are rejected — internal services live on odd
    ports, and no real PDF host needs one
  * the hostname is RESOLVED, and every returned address must be globally
    routable: loopback, private (RFC1918), link-local (incl. 169.254.169.254),
    unique-local, reserved, and multicast are all refused
  * redirects are followed MANUALLY so each hop is re-validated — a public URL
    that 302s to 127.0.0.1 is the classic bypass

Residual risk (accepted, documented)
------------------------------------
DNS rebinding: we resolve, approve, then httpx resolves again when it connects,
and an attacker controlling their own DNS can return a public IP to the first
lookup and a private one to the second. Closing that hole entirely means
connecting to the vetted IP directly with a Host/SNI override, which httpx does
not expose cleanly. The window is small and the payload must additionally be a
valid PDF under the size cap, so this is accepted for now.

Public API
----------
validate_pdf_url(url) -> str    returns the normalised URL, or raises UnsafeUrl
UnsafeUrl                        raised for any URL that fails the checks
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

# Optional strict mode: comma-separated host suffixes. Empty (default) means
# "any public host", which is what keeps Semantic Scholar imports working.
_ALLOWLIST = tuple(
    h.strip().lower().lstrip(".")
    for h in os.getenv("PAPERMIND_PDF_HOST_ALLOWLIST", "").split(",")
    if h.strip()
)


class UnsafeUrl(ValueError):
    """Raised when a user-supplied URL must not be fetched by the server."""


def _is_public_address(raw: str) -> bool:
    """True only if `raw` is a globally routable IP.

    `is_global` already excludes loopback, private, link-local, reserved and
    unspecified ranges; multicast is excluded separately because a multicast
    address is not something a PDF is ever served from and treating it as
    "not global" is version-dependent.
    """
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return False
    return ip.is_global and not ip.is_multicast


def _resolve_all(host: str) -> list[str]:
    """Every A/AAAA address `host` resolves to. Raises UnsafeUrl if it doesn't
    resolve at all — an unresolvable host is not something to hand to httpx."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrl("Could not resolve the host for that URL.") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise UnsafeUrl("Could not resolve the host for that URL.")
    return sorted(addresses)


def validate_pdf_url(url: str) -> str:
    """Return `url` unchanged if the server may safely fetch it.

    Raises UnsafeUrl otherwise. The message is deliberately generic and
    non-diagnostic: distinguishing "connection refused" from "not a PDF" from
    "blocked" turns this endpoint into an internal port scanner. Callers must
    not echo any more detail than this to the client.
    """
    if not url or not url.strip():
        raise UnsafeUrl("No PDF URL provided.")

    parts = urlsplit(url.strip())

    if parts.scheme.lower() != "https":
        raise UnsafeUrl("Only https:// PDF URLs can be imported.")

    if parts.username or parts.password:
        raise UnsafeUrl("PDF URLs must not contain credentials.")

    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeUrl("That URL has no host.")

    # An explicit non-443 port is the shape of an internal-service probe.
    # `parts.port` raises on a malformed port, which is itself a rejection.
    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeUrl("That URL has an invalid port.") from exc
    if port not in (None, 443):
        raise UnsafeUrl("Only the standard https port is allowed.")

    if _ALLOWLIST and not any(
        host == allowed or host.endswith("." + allowed) for allowed in _ALLOWLIST
    ):
        raise UnsafeUrl("That host is not on this server's import allow-list.")

    # A literal IP in the URL skips DNS entirely — check it directly, then fall
    # through to the resolver for real hostnames.
    literal = host.strip("[]")
    try:
        ipaddress.ip_address(literal)
        candidates = [literal]
    except ValueError:
        candidates = _resolve_all(host)

    for address in candidates:
        if not _is_public_address(address):
            raise UnsafeUrl("That URL resolves to a non-public address.")

    return url.strip()

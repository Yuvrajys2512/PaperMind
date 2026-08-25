"""
api/content_disposition.py — safe Content-Disposition headers for user filenames.

The PDF route serves a filename the user chose at upload time. Interpolating it
straight into the header, as this used to do:

    f'inline; filename="{filename}"'

has two failure modes, one of which is a guaranteed 500:

  * **Encoding.** Starlette encodes response headers as latin-1. A filename
    containing CJK, Cyrillic, Greek — anything outside latin-1 — raises
    UnicodeEncodeError while building the response, so the PDF preview is
    permanently broken for that paper. Verified with `研究.pdf`.
  * **Injection.** A `"` in the filename closes the quoted string early, letting
    the user append their own header parameters. A CR/LF would be worse still,
    though h11 rejects those at the transport layer.

RFC 6266/5987 solves both: send an ASCII-sanitised `filename=` for old clients
plus a percent-encoded UTF-8 `filename*=` that every current browser prefers.

Public API
----------
content_disposition(filename, disposition="inline") -> str
"""

from __future__ import annotations

import re
from urllib.parse import quote

# Characters that must not survive into the quoted ASCII fallback: quotes and
# backslashes would break out of it, control characters would corrupt the header.
_UNSAFE = re.compile(r'[\x00-\x1f\x7f"\\]')

_FALLBACK_NAME = "document.pdf"

# Long filenames are legal but pointless in a header, and a very long one is a
# cheap way to bloat every response.
_MAX_LEN = 120


def _sanitise_ascii(text: str) -> str:
    """Drop anything that can't safely appear in a quoted header parameter."""
    cleaned = _UNSAFE.sub("", text)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned.strip().strip(". ")


def _ascii_fallback(filename: str) -> str:
    """A latin-1-safe, quote-safe version of `filename` for the plain parameter.

    Non-ASCII characters are dropped rather than transliterated: the `filename*`
    parameter carries the real name, so this only has to be *something* legible
    and safe.

    Stem and extension are sanitised separately, and deliberately so. Sanitising
    the whole string at once turns "研究.pdf" into ".pdf" and then — after the
    leading dot is stripped — into the bare word "pdf", which reads as a
    filename with no extension and makes clients offer to save an extensionless
    file. Keeping the extension attached to a generic stem gives "document.pdf",
    which is what a user would expect to see.
    """
    stem, dot, ext = filename.rpartition(".")
    if not dot:  # no extension at all
        return _sanitise_ascii(filename)[:_MAX_LEN] or _FALLBACK_NAME

    clean_stem = _sanitise_ascii(stem)
    clean_ext = _sanitise_ascii(ext)
    if not clean_ext:
        return clean_stem[:_MAX_LEN] or _FALLBACK_NAME
    if not clean_stem:
        clean_stem = "document"
    return f"{clean_stem[:_MAX_LEN]}.{clean_ext}"


def content_disposition(filename: str, disposition: str = "inline") -> str:
    """Build a Content-Disposition header value that is safe to send verbatim.

    Always latin-1 encodable, never breakable by the filename's contents.
    """
    if disposition not in ("inline", "attachment"):
        raise ValueError("disposition must be 'inline' or 'attachment'")

    name = (filename or "").strip()[:_MAX_LEN] or _FALLBACK_NAME

    ascii_name = _ascii_fallback(name)
    # `quote` with an empty safe-list percent-encodes everything RFC 5987 wants
    # encoded, and its output is ASCII by construction.
    encoded = quote(name, safe="", encoding="utf-8")

    header = f'{disposition}; filename="{ascii_name}"'
    # Only add the extended parameter when it actually says something different;
    # for a plain ASCII name it would be pure noise.
    if encoded != ascii_name:
        header += f"; filename*=UTF-8''{encoded}"
    return header

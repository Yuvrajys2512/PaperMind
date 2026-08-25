"""
Unit tests for api/content_disposition.py.

The bug this pins: `GET /papers/{id}/pdf` interpolated the user's uploaded
filename straight into the header. Starlette encodes response headers as
latin-1, so any filename outside that range (CJK, Cyrillic, Greek, most
emoji) raised UnicodeEncodeError while building the response — a hard 500,
permanently, for that paper. A `"` in the name was separately able to break
out of the quoted parameter.

The invariant that matters is the first test: whatever comes in, the header
must always be latin-1 encodable.
"""

import pytest

from api.content_disposition import content_disposition

# Names that broke the old implementation, plus ordinary ones that must survive.
FILENAMES = [
    "paper.pdf",
    "Attention Is All You Need.pdf",
    "études-sur-les-transformeurs.pdf",   # latin-1: fine before, must stay fine
    "研究.pdf",                            # CJK — the reported 500
    "статья.pdf",                          # Cyrillic
    "μελέτη.pdf",                          # Greek
    "논문.pdf",                             # Hangul
    "papier–résumé…pdf",                   # en dash + ellipsis (not latin-1)
    "🎓 thesis.pdf",                        # emoji
    'quote".pdf',                          # parameter break-out
    "back\\slash.pdf",
    "semi;colon.pdf",
    "",                                    # missing filename
    "   ",
    "…….pdf",                              # nothing ASCII survives sanitising
    "x" * 500,                             # absurd length
]


@pytest.mark.parametrize("name", FILENAMES)
def test_header_is_always_latin1_encodable(name):
    """The core invariant. This is the exact failure the old code hit."""
    header = content_disposition(name)
    header.encode("latin-1")  # must not raise


@pytest.mark.parametrize("name", FILENAMES)
def test_header_has_no_control_characters(name):
    header = content_disposition(name)
    assert not any(ord(c) < 0x20 or ord(c) == 0x7F for c in header), (
        "control characters in a header value can corrupt the response"
    )


@pytest.mark.parametrize("name", FILENAMES)
def test_quoted_parameter_cannot_be_broken_out_of(name):
    """Exactly two quotes: the ones delimiting the ASCII filename parameter."""
    header = content_disposition(name)
    assert header.count('"') == 2, f"unbalanced quoting in: {header!r}"


def test_plain_ascii_name_is_passed_through_unchanged():
    assert content_disposition("paper.pdf") == 'inline; filename="paper.pdf"'


def test_plain_ascii_name_gets_no_redundant_extended_parameter():
    assert "filename*" not in content_disposition("paper.pdf")


def test_non_ascii_name_is_carried_in_the_extended_parameter():
    header = content_disposition("研究.pdf")
    assert "filename*=UTF-8''" in header
    # The real name survives, percent-encoded, for clients that understand it.
    assert "%E7%A0%94%E7%A9%B6" in header


def test_name_with_no_ascii_content_gets_a_usable_fallback():
    header = content_disposition("研究.pdf")
    assert 'filename="document.pdf"' in header, (
        "the ASCII fallback must not be empty — old clients would show nothing"
    )


def test_empty_filename_falls_back():
    assert 'filename="document.pdf"' in content_disposition("")


def test_disposition_can_be_attachment():
    assert content_disposition("paper.pdf", "attachment").startswith("attachment;")


def test_invalid_disposition_is_rejected():
    with pytest.raises(ValueError):
        content_disposition("paper.pdf", "inline; evil")


def test_long_names_are_bounded():
    header = content_disposition("y" * 5000 + ".pdf")
    assert len(header) < 1000, "a filename must not be able to bloat every response"


def test_the_old_implementation_would_have_failed_this():
    """Documents the regression directly: the previous one-liner raises on the
    exact input the tests above now cover."""
    with pytest.raises(UnicodeEncodeError):
        f'inline; filename="{"研究.pdf"}"'.encode("latin-1")

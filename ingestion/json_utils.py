"""
ingestion/json_utils.py

Lenient JSON extraction for LLM outputs.

Free-tier / smaller models frequently break strict json.loads in two ways:
  1. They wrap the JSON in a markdown fence (```json ... ```), so the first
     character is a backtick → "Expecting value: line 1 column 1".
  2. They emit invalid backslash escapes inside string values — LaTeX/math
     like "\\(n\\)", "\\%", "\\textit" — which is not valid JSON →
     "Invalid \\escape: line N column M".

Both are recoverable. parse_llm_json() strips the wrapper and repairs the
common escape error so the pipeline keeps a usable plan/grade instead of
falling back to a degraded one.
"""

from __future__ import annotations

import json
import re

# A backslash that does NOT start a valid JSON escape sequence
# (\" \\ \/ \b \f \n \r \t \uXXXX). Anything else after a backslash is an
# invalid escape; we double it so the backslash becomes a literal character.
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _strip_fences(raw: str) -> str:
    """Remove a leading ```json / ``` opener and a trailing ``` closer."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _extract_span(raw: str, open_ch: str, close_ch: str) -> str:
    """Slice from the first open_ch to the last close_ch (inclusive).

    Drops any stray prose the model printed before/after the JSON value.
    Returns raw unchanged if the delimiters aren't found.
    """
    start = raw.find(open_ch)
    end = raw.rfind(close_ch)
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def parse_llm_json(raw: str, expect: str = "object"):
    """Parse JSON emitted by an LLM, tolerating fences and invalid escapes.

    Parameters
    ----------
    raw : str
        Raw model output.
    expect : str
        "object" -> extract the {...} span (use for a JSON object)
        "array"  -> extract the [...] span (use for a JSON array)
        "any"    -> no span extraction

    Returns
    -------
    The parsed JSON value (dict / list / etc.).

    Raises
    ------
    json.JSONDecodeError if the text cannot be parsed even after repair.
    """
    text = _strip_fences(raw)

    if expect == "object":
        text = _extract_span(text, "{", "}")
    elif expect == "array":
        text = _extract_span(text, "[", "]")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair invalid backslash escapes and retry once.
        return json.loads(_BAD_ESCAPE.sub(r"\\\\", text))

"""
Unit tests for ingestion/off_topic_guard.py — the cheap pre-flight classifier
that decides whether a question is on-topic for a research paper.

The single LLM call is mocked, so these run offline. We verify the verdict
parsing (OFFTOPIC vs RESEARCH), its tolerance of case/whitespace/punctuation,
and the fail-open behavior when the LLM call raises.
"""

from ingestion import off_topic_guard


def _patch_llm(monkeypatch, reply=None, exc=None):
    def fake_chat_completion(*args, **kwargs):
        if exc is not None:
            raise exc
        return reply
    monkeypatch.setattr(off_topic_guard, "chat_completion", fake_chat_completion)


def test_offtopic_verdict_blocks_with_message(monkeypatch):
    _patch_llm(monkeypatch, reply="OFFTOPIC")
    is_off, msg = off_topic_guard.check("what's the football score?")
    assert is_off is True
    assert msg  # non-empty friendly message


def test_research_verdict_passes_through(monkeypatch):
    _patch_llm(monkeypatch, reply="RESEARCH")
    is_off, msg = off_topic_guard.check("what attention mechanism do they use?")
    assert is_off is False
    assert msg == ""


def test_verdict_is_case_and_whitespace_insensitive(monkeypatch):
    _patch_llm(monkeypatch, reply="  offtopic\n")
    is_off, _ = off_topic_guard.check("anything")
    assert is_off is True


def test_verdict_tolerates_trailing_punctuation(monkeypatch):
    _patch_llm(monkeypatch, reply="OFFTOPIC.")
    is_off, _ = off_topic_guard.check("anything")
    assert is_off is True


def test_llm_failure_fails_open(monkeypatch):
    # On any LLM error the guard must NOT block the user.
    _patch_llm(monkeypatch, exc=RuntimeError("provider down"))
    is_off, msg = off_topic_guard.check("anything")
    assert is_off is False
    assert msg == ""

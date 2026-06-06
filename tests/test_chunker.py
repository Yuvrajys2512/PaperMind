"""
Unit tests for ingestion/chunker.py — the token-window chunking logic.

`chunk_text` takes the tokenizer as an argument, so we inject a trivial
word-based fake tokenizer. This makes the tests fully offline and deterministic
(no tiktoken encoding download, no network) while still exercising the real
windowing + overlap arithmetic. `chunk_sections` is tested by monkeypatching
the module's get_tokenizer() to return the same fake.
"""

from ingestion import chunker
from ingestion.chunker import chunk_sections, chunk_text


class WordTokenizer:
    """Stand-in tokenizer: one token per whitespace-separated word."""

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


tok = WordTokenizer()


# --- chunk_text -------------------------------------------------------------

def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=3, overlap=1, tokenizer=tok) == []


def test_text_shorter_than_chunk_is_one_chunk():
    assert chunk_text("a b", chunk_size=5, overlap=1, tokenizer=tok) == ["a b"]


def test_windowing_with_overlap():
    # 6 tokens, size 3, overlap 1 -> step 2 -> windows [0:3],[2:5],[4:7]
    out = chunk_text("w0 w1 w2 w3 w4 w5", chunk_size=3, overlap=1, tokenizer=tok)
    assert out == ["w0 w1 w2", "w2 w3 w4", "w4 w5"]


def test_overlap_actually_shares_tokens():
    out = chunk_text("w0 w1 w2 w3 w4 w5", chunk_size=3, overlap=1, tokenizer=tok)
    # last token of chunk i reappears as first token of chunk i+1
    assert out[0].split()[-1] == out[1].split()[0]  # w2
    assert out[1].split()[-1] == out[2].split()[0]  # w4


def test_zero_overlap_produces_disjoint_chunks():
    out = chunk_text("a b c d", chunk_size=2, overlap=0, tokenizer=tok)
    assert out == ["a b", "c d"]
    joined = " ".join(out).split()
    assert len(joined) == len(set(joined))  # no token repeated


# --- chunk_sections ---------------------------------------------------------

def test_chunk_sections_skips_empty_bodies_and_attaches_metadata(monkeypatch):
    monkeypatch.setattr(chunker, "get_tokenizer", lambda: tok)
    sections = [
        {"heading": "Intro", "type": "text", "page_num": 1, "body": "a b c d"},
        {"heading": "Blank", "type": "text", "page_num": 2, "body": "   "},
    ]
    out = chunk_sections(sections, chunk_size=2, overlap=0)

    # Blank section is skipped; Intro -> 2 chunks of 2 tokens
    assert len(out) == 2
    assert all(c["section"] == "Intro" for c in out)
    assert [c["chunk_index"] for c in out] == [0, 1]
    assert all(c["total_chunks_in_section"] == 2 for c in out)
    assert all(c["page_num"] == 1 and c["section_type"] == "text" for c in out)
    assert [c["chunk_id"] for c in out] == [1, 2]
    assert all(c["token_count"] == 2 for c in out)


def test_chunk_sections_empty_list():
    assert chunk_sections([], chunk_size=2, overlap=0) == []

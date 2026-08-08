"""
Unit tests for ingestion.retriever._reading_order_key — restoring a paper's
reading order after chunks come back from the vector store.

Why this is worth pinning: `get_all_chunks` used to sort on a `chunk_id`
metadata key that the embedder never writes (it writes `chunk_index`), so
`.get("chunk_id", 0)` returned 0 for every chunk and the sort did nothing.
Callers were handed the store's own ordering, which matched the document only
by coincidence.

Several callers quietly depend on the order being right:
  - citation_gap_auditor decides a statement is cited if the marker sits in the
    *next* sentence,
  - plagiarism_auditor flattens chunks into one word stream and matches runs
    across chunk boundaries,
  - claim_auditor slices sections out by position.
None of them fail loudly on scrambled input; they just return wrong answers.
"""

from ingestion.retriever import _reading_order_key


def _chunk(text, **meta):
    return {"text": text, "metadata": meta}


def _order(chunks):
    ordered = sorted(chunks, key=_reading_order_key(chunks))
    return [c["text"] for c in ordered]


# --- exact ordering, when doc_index is present -------------------------------

def test_doc_index_restores_document_order():
    scrambled = [
        _chunk("third",  doc_index=2, page_num=9, chunk_index=0),
        _chunk("first",  doc_index=0, page_num=1, chunk_index=0),
        _chunk("second", doc_index=1, page_num=4, chunk_index=1),
    ]
    assert _order(scrambled) == ["first", "second", "third"]


def test_doc_index_beats_the_page_heuristic():
    """doc_index is authoritative: an appendix chunk printed on an early page
    still belongs where the document put it."""
    chunks = [
        _chunk("body",     doc_index=1, page_num=7, chunk_index=0),
        _chunk("abstract", doc_index=0, page_num=1, chunk_index=0),
    ]
    assert _order(chunks) == ["abstract", "body"]


def test_doc_index_zero_is_not_treated_as_missing():
    """`or 0` on a falsy-but-valid 0 is a classic way to break this."""
    chunks = [
        _chunk("later", doc_index=1, page_num=1, chunk_index=0),
        _chunk("first", doc_index=0, page_num=1, chunk_index=0),
    ]
    assert _order(chunks) == ["first", "later"]


# --- fallback, for collections ingested before doc_index existed -------------

def test_falls_back_to_page_then_section_index():
    chunks = [
        _chunk("p2-b", page_num=2, chunk_index=1),
        _chunk("p1-a", page_num=1, chunk_index=0),
        _chunk("p2-a", page_num=2, chunk_index=0),
        _chunk("p1-b", page_num=1, chunk_index=1),
    ]
    assert _order(chunks) == ["p1-a", "p1-b", "p2-a", "p2-b"]


def test_partial_doc_index_uses_the_fallback_for_everything():
    """A half-migrated collection must not sort the annotated chunks as a block
    ahead of the rest — that is worse than using one scheme consistently."""
    chunks = [
        _chunk("p2", page_num=2, chunk_index=0, doc_index=1),
        _chunk("p1", page_num=1, chunk_index=0),   # no doc_index
    ]
    assert _order(chunks) == ["p1", "p2"]


def test_ties_keep_the_incoming_order():
    """Several sections can start on one page, each reporting chunk_index 0.
    A stable sort leaves those in the order the store returned rather than
    shuffling them."""
    chunks = [
        _chunk("2.1 Models",   page_num=3, chunk_index=0),
        _chunk("2.2 Retriever", page_num=3, chunk_index=0),
        _chunk("2.3 Generator", page_num=3, chunk_index=0),
    ]
    assert _order(chunks) == ["2.1 Models", "2.2 Retriever", "2.3 Generator"]


# --- defensive: real metadata is not always complete -------------------------

def test_missing_page_and_index_do_not_raise():
    """A None page_num compared against an int would be a TypeError, taking
    down every audit on that paper."""
    chunks = [
        _chunk("has-page", page_num=2, chunk_index=0),
        _chunk("no-meta"),
        _chunk("null-page", page_num=None, chunk_index=None),
    ]
    assert _order(chunks) == ["no-meta", "null-page", "has-page"]


def test_empty_collection_is_handled():
    assert _order([]) == []

"""
Unit tests for the caption-anchored table pipeline:
  - ingestion/table_extractor.py  (formatting records -> chunks)
  - ingestion/pdf_parser.py        (caption regex + column-confined crop range)

These are pure-function tests — no PDF, no network — so they stay fast and
deterministic. The PDF-level detection is exercised separately via the manual
smoke checks against real papers.
"""

from ingestion.table_extractor import tables_to_chunks
from ingestion.pdf_parser import _TABLE_CAPTION_RE, _caption_x_range


def _chars(x0, x1):
    """A minimal caption-line char list (only x extents matter here)."""
    return [{"x0": x0, "x1": x1}]


# --- caption regex ----------------------------------------------------------

def test_caption_regex_matches_real_captions():
    assert _TABLE_CAPTION_RE.match("Table 1: GLUE results")
    assert _TABLE_CAPTION_RE.match("Table 6. In this table we report")
    assert _TABLE_CAPTION_RE.match("TABLE 3: Variations")


def test_caption_regex_rejects_cross_references():
    # In-body references must not be mistaken for captions (no colon/period).
    assert _TABLE_CAPTION_RE.match("Table 2 summarizes our results") is None
    assert _TABLE_CAPTION_RE.match("see Table 4 for details") is None


# --- column-confined crop range ---------------------------------------------

def test_x_range_single_column_uses_full_width():
    assert _caption_x_range(_chars(50, 300), col_boundary=None, page_width=600) == (0.0, 600)


def test_x_range_left_column():
    # caption sits left of the gutter -> crop the left column only
    assert _caption_x_range(_chars(50, 250), col_boundary=300, page_width=600) == (0.0, 300)


def test_x_range_right_column():
    assert _caption_x_range(_chars(320, 540), col_boundary=300, page_width=600) == (300, 600)


def test_x_range_full_width_caption_spans_gutter():
    # caption text flows across the gutter -> full-width table, keep whole width
    assert _caption_x_range(_chars(50, 550), col_boundary=300, page_width=600) == (0.0, 600)


# --- tables_to_chunks formatting --------------------------------------------

def _record(rows, caption="", page_num=1):
    return {"rows": rows, "caption": caption, "page_num": page_num, "bbox": (0, 0, 1, 1)}


def test_chunk_leads_with_caption_and_linearizes_headers():
    rows = [["Model", "EM", "F1"], ["BERT", "84.1", "90.9"]]
    out = tables_to_chunks([_record(rows, caption="Table 2: SQuAD results")])
    assert len(out) == 1
    c = out[0]
    assert c["section"] == "Table 2: SQuAD results"
    assert c["section_type"] == "table"
    assert c["chunk_id"] is None
    # caption is the semantic anchor and comes first
    assert c["text"].startswith("Table 2: SQuAD results")
    # cells are linearized against their column headers
    assert "BERT — EM: 84.1; F1: 90.9" in c["text"]
    # markdown grid is retained for display
    assert "| Model | EM | F1 |" in c["text"]


def test_chunk_without_caption_falls_back_to_generic_name():
    rows = [["a", "b"], ["1", "2"]]
    out = tables_to_chunks([_record(rows, caption="", page_num=5)])
    assert out[0]["section"] == "Table 1 (page 5)"


def test_single_row_and_single_column_records_are_filtered():
    one_row = _record([["a", "b"]])
    one_col = _record([["a"], ["1"], ["2"]])
    assert tables_to_chunks([one_row, one_col]) == []


def test_none_cells_become_empty_strings():
    rows = [["Model", "EM", "F1"], ["BERT", None, "90.9"]]
    out = tables_to_chunks([_record(rows, caption="Table 1: x")])
    # missing EM is dropped from the linearization, F1 still anchored
    assert "BERT — F1: 90.9" in out[0]["text"]
    assert "EM:" not in out[0]["text"].split("| Model")[0]  # not in the linearized part


def test_empty_input_returns_empty():
    assert tables_to_chunks([]) == []

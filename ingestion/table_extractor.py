"""
ingestion/table_extractor.py

Formats table records (produced by pdf_parser's single detection pass) into
chunk dicts ready for embedding.

The key job here is making a table *retrievable*. A naked markdown grid —
`| 0.85 | 0.79 |` — embeds as a soup of digits with no lexical or semantic
anchor, so a question like "how did the model do on SQuAD vs BERT?" never
matches it. We fix that two ways:
  1. Prepend the bound caption ("Table 3: F1 scores on SQuAD ...").
  2. Linearize each cell against its row + column headers, so the meaning that
     normally lives only in the visual intersection becomes plain text
     ("BERT — SQuAD F1: 0.79").
The original markdown is kept too, for display.

Public API
----------
tables_to_chunks(table_records) -> list[dict]
    Format parser table records into chunk dicts (section_type="table").
"""

import tiktoken

_enc = None


def _get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _cell(value) -> str:
    """Normalize a raw table cell to a stripped string ('' for None)."""
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")


def _table_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    out = []
    for row in rows:
        cells = [_cell(c) for c in row]
        out.append("| " + " | ".join(cells) + " |")
    if len(out) >= 2:
        sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        out.insert(1, sep)
    return "\n".join(out)


def _linearize(rows: list[list]) -> str:
    """Render data rows as header-anchored sentences.

    Treats row 0 as column headers and column 0 of each data row as that row's
    label, then emits 'row_label — header: value; header: value' lines. This is
    what gives the chunk real retrieval anchors over the numbers.
    """
    if len(rows) < 2:
        return ""
    headers = [_cell(c) for c in rows[0]]
    lines = []
    for row in rows[1:]:
        cells = [_cell(c) for c in row]
        if not any(cells):
            continue
        row_label = cells[0]
        parts = []
        for j in range(1, len(cells)):
            val = cells[j]
            if not val:
                continue
            header = headers[j] if j < len(headers) and headers[j] else f"col{j}"
            parts.append(f"{header}: {val}")
        if row_label and parts:
            lines.append(f"{row_label} — " + "; ".join(parts))
        elif parts:
            lines.append("; ".join(parts))
        elif row_label:
            lines.append(row_label)
    return "\n".join(lines)


def tables_to_chunks(table_records: list[dict]) -> list[dict]:
    """
    Format table records from pdf_parser into chunk dicts.

    Each record is {"rows", "page_num", "caption", "bbox"}.

    Filters out single-row / single-column artifacts (noise). Each chunk:
      section      : the bound caption, else "Table N (page P)"
      section_type : "table"
      page_num     : page the table appeared on
    chunk_id is set to None — ingest_document reassigns sequential IDs.
    """
    enc = _get_enc()
    chunks = []
    table_num = 0

    for rec in table_records:
        rows = rec.get("rows") or []
        if len(rows) < 2:
            continue
        if not rows[0] or len(rows[0]) < 2:
            continue

        markdown = _table_to_markdown(rows)
        if not markdown.strip():
            continue

        table_num += 1
        page_num = rec.get("page_num")
        caption = (rec.get("caption") or "").strip()
        section_name = caption if caption else f"Table {table_num} (page {page_num})"

        # Embedded text leads with the caption + linearized rows (the semantic
        # anchors), then the markdown grid for faithful display.
        linear = _linearize(rows)
        parts = []
        if caption:
            parts.append(caption)
        if linear:
            parts.append(linear)
        parts.append(markdown)
        text = "\n\n".join(parts)

        token_count = len(enc.encode(text))

        chunks.append({
            "chunk_id":                None,
            "section":                 section_name,
            "section_type":            "table",
            "page_num":                page_num,
            "chunk_index":             0,
            "total_chunks_in_section": 1,
            "text":                    text,
            "token_count":             token_count,
        })

    return chunks

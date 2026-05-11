"""
ingestion/table_extractor.py  — Session 5

Extracts tables from PDF pages using pdfplumber and converts them to
markdown-style text chunks ready for embedding.

Public API
----------
extract_tables_from_pdf(pdf_path) -> list[dict]
    Returns chunk dicts (same schema as chunk_sections) with section_type="table".
"""

import pdfplumber
import tiktoken

_enc = None


def _get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _table_to_markdown(table: list[list]) -> str:
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    if len(rows) >= 2:
        sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
        rows.insert(1, sep)
    return "\n".join(rows)


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract all tables from a PDF, return as chunk dicts.

    Filters out single-row or single-column artifacts (noise).
    Each table chunk gets:
      section      : "Table N (page P)"
      section_type : "table"
      page_num     : page the table appeared on
    chunk_id is set to None — ingest_document reassigns sequential IDs.
    """
    enc = _get_enc()
    chunks = []
    table_num = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if len(table) < 2:
                    continue
                if not table[0] or len(table[0]) < 2:
                    continue

                markdown = _table_to_markdown(table)
                if not markdown.strip():
                    continue

                table_num += 1
                section_name = f"Table {table_num} (page {page_num})"
                token_count = len(enc.encode(markdown))

                chunks.append({
                    "chunk_id":                None,
                    "section":                 section_name,
                    "section_type":            "table",
                    "page_num":                page_num,
                    "chunk_index":             0,
                    "total_chunks_in_section": 1,
                    "text":                    markdown,
                    "token_count":             token_count,
                })

    return chunks

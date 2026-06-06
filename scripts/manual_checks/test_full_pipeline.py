"""
test_full_pipeline.py — End-to-end test of the ingestion pipeline.

Runs each stage sequentially for both papers and prints intermediate
outputs so you can verify correctness at every step.

Pipeline:  PDF → Parser → Section Detector → Chunker → Embedder
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ingestion.pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section
from ingestion.section_detector import build_candidates, confirm_headings_with_llm, assemble_sections
from ingestion.chunker import chunk_sections


# ── Config ────────────────────────────────────────────────────────────────────
PAPERS = [
    ("Attention is all you need", "data/Attention is all you need.pdf"),
    ("BERT paper", "data/BERT paper.pdf"),
]

CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

DIVIDER = "=" * 70


def test_paper(paper_name: str, pdf_path: str):
    print(f"\n{DIVIDER}")
    print(f"  PAPER: {paper_name}")
    print(f"{DIVIDER}\n")

    if not os.path.exists(pdf_path):
        print(f"  [SKIP] File not found: {pdf_path}")
        return

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: PDF Parsing
    # ──────────────────────────────────────────────────────────────────────
    print("STEP 1: PDF Parsing")
    print("-" * 40)

    result = extract_text_from_pdf(pdf_path)
    full_text = result["full_text"]
    cleaned_text = remove_credits_block(full_text)
    cleaned_text = remove_references_section(cleaned_text)

    print(f"  Total pages      : {result['total_pages']}")
    print(f"  Raw chars        : {len(full_text):,}")
    print(f"  Cleaned chars    : {len(cleaned_text):,}")
    print(f"  Pages with text  : {len(result['pages'])}")
    print(f"\n  -- First 300 chars of cleaned text --")
    print(f"  {cleaned_text[:300]}")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Section Detection (candidates + LLM confirmation)
    # ──────────────────────────────────────────────────────────────────────
    print("STEP 2: Section Detection")
    print("-" * 40)

    pages = result["pages"]
    candidates = build_candidates(pages)
    print(f"  Heading candidates found: {len(candidates)}")
    print()

    print("  -- All Candidates --")
    for c in candidates:
        print(f"    [{c['id']:>2}] p{c['page_num']} | score={c['score']:>2} | \"{c['candidate_line'].strip()[:60]}\"")
    print()

    print("  Sending candidates to LLM for confirmation...")
    confirmed = confirm_headings_with_llm(candidates)
    print(f"  LLM verdicts received: {len(confirmed)}")
    print()

    print("  -- LLM Verdicts --")
    for c in candidates:
        verdict = confirmed.get(c["id"], "???")
        marker = "+" if verdict in ("SECTION", "SUBSECTION") else " "
        print(f"    {marker} [{c['id']:>2}] {verdict:>12} | \"{c['candidate_line'].strip()[:60]}\"")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Section Assembly
    # ──────────────────────────────────────────────────────────────────────
    print("STEP 3: Section Assembly")
    print("-" * 40)

    sections = assemble_sections(pages, confirmed, candidates)
    print(f"  Sections assembled: {len(sections)}")
    print()

    print("  -- Sections Overview --")
    for i, sec in enumerate(sections):
        body_preview = sec["body"][:80].replace("\n", " ")
        print(f"    [{i+1:>2}] {sec['type']:>12} | p{sec['page_num']} | \"{sec['heading'][:50]}\"")
        print(f"         Body ({len(sec['body']):,} chars): {body_preview}...")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Chunking
    # ──────────────────────────────────────────────────────────────────────
    print("STEP 4: Chunking")
    print("-" * 40)

    chunks = chunk_sections(sections, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"  Total chunks created: {len(chunks)}")
    print()

    print("  -- Chunks Overview --")
    for chunk in chunks:
        text_preview = chunk["text"][:70].replace("\n", " ")
        print(f"    chunk_{chunk['chunk_id']:>3} | {chunk['section'][:30]:>30} | "
              f"{chunk['token_count']:>4} tokens | {text_preview}...")
    print()

    # Show a full sample chunk
    if chunks:
        sample = chunks[0]
        print("  -- Sample Chunk (first chunk, full text) --")
        print(f"    Section : {sample['section']}")
        print(f"    Type    : {sample['section_type']}")
        print(f"    Page    : {sample['page_num']}")
        print(f"    Tokens  : {sample['token_count']}")
        print(f"    Text    :")
        for line in sample["text"].split("\n")[:10]:
            print(f"      {line}")
        if sample["text"].count("\n") > 10:
            print(f"      ... ({sample['text'].count(chr(10)) - 10} more lines)")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Embedding (just verify it would work, don't store)
    # ──────────────────────────────────────────────────────────────────────
    print("STEP 5: Embedding (dry run)")
    print("-" * 40)
    print(f"  Ready to embed {len(chunks)} chunks for '{paper_name}'")
    print(f"  To actually embed and store, uncomment the embed_and_store() call below.")
    print()

    # Uncomment the next 2 lines to actually embed and store in ChromaDB:
    # from ingestion.embedder import embed_and_store
    # embed_and_store(chunks, paper_name)

    print(f"  PIPELINE COMPLETE for: {paper_name}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + DIVIDER)
    print("  FULL PIPELINE TEST")
    print(DIVIDER)

    for paper_name, pdf_path in PAPERS:
        test_paper(paper_name, pdf_path)

    print(DIVIDER)
    print("  ALL DONE")
    print(DIVIDER)

from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.section_detector import (
    build_candidates,
    confirm_headings_with_llm,
    assemble_sections
)
from ingestion.chunker import chunk_sections

PDF_PATH = r"data\Attention is all you need.pdf"

result = extract_text_from_pdf(PDF_PATH)
candidates = build_candidates(result["pages"])
confirmed = confirm_headings_with_llm(candidates)
sections = assemble_sections(result["pages"], confirmed, candidates)

# Test all six combinations
combinations = [
    (256, 50), (256, 100),
    (512, 50), (512, 100),
    (1024, 50), (1024, 100)
]

for chunk_size, overlap in combinations:
    chunks = chunk_sections(sections, chunk_size, overlap)
    print(f"\nChunk size: {chunk_size} | Overlap: {overlap}")
    print(f"Total chunks produced: {len(chunks)}")
    print(f"Sample chunk from Introduction:")
    intro_chunks = [c for c in chunks if "Introduction" in c["section"]]
    if intro_chunks:
        print(f"  Tokens: {intro_chunks[0]['token_count']}")
        print(f"  Text preview: {intro_chunks[0]['text'][:200]}")
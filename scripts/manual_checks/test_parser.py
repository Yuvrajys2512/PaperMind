from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.section_detector import (
    build_candidates,
    confirm_headings_with_llm,
    assemble_sections
)
from ingestion.chunker import chunk_sections
from ingestion.embedder import embed_and_store
from ingestion.retriever import retrieve

PDF_PATH = r"data\Attention is all you need.pdf"
PAPER_NAME = "Attention Is All You Need"

# Step 1 — Extract
print("Extracting text...")
result = extract_text_from_pdf(PDF_PATH)

# Step 2 — Detect sections
print("Detecting sections...")
candidates = build_candidates(result["pages"])
confirmed = confirm_headings_with_llm(candidates)
sections = assemble_sections(result["pages"], confirmed, candidates)
print(f"Found {len(sections)} sections")

# Step 3 — Chunk
print("Chunking...")
chunks = chunk_sections(sections, chunk_size=512, overlap=100)
print(f"Created {len(chunks)} chunks")

# Step 4 — Embed and store
embed_and_store(chunks, PAPER_NAME)

# Step 5 — Retrieve
print("\n--- RETRIEVAL TEST ---")
questions = [
    "What is the attention mechanism?",
    "How many layers does the encoder have?",
    "What optimizer was used for training?",
    "What BLEU score did the model achieve?"
]

for question in questions:
    print(f"\nQ: {question}")
    results = retrieve(question, PAPER_NAME, top_k=3)
    for i, r in enumerate(results):
        print(f"  Result {i+1} [{r['metadata']['section']}] "
              f"(page {r['metadata']['page_num']}, "
              f"distance: {r['distance']:.3f})")
        print(f"  {r['text'][:150]}...")
import sys
import os
import json
sys.path.insert(0, r"c:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind")

from ingestion.pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section
from ingestion.section_detector import build_candidates, confirm_headings_with_llm, assemble_sections
from ingestion.chunker import chunk_sections

paper_name = "rag"
pdf_path = "data/Retrieval-Augmented Generation for.pdf"

print("Extracting...")
result = extract_text_from_pdf(pdf_path)
full_text = result["full_text"]
cleaned = remove_references_section(remove_credits_block(full_text))

print("Detecting sections...")
candidates = build_candidates(result["pages"])
confirmed = confirm_headings_with_llm(candidates)
sections = assemble_sections(result["pages"], confirmed, candidates)

print("Chunking...")
chunks = chunk_sections(sections, chunk_size=256, overlap=50)

with open("scratch/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Saved {len(chunks)} chunks to chunks.json")

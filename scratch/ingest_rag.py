import sys
import os

sys.path.insert(0, r"c:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind")

from ingestion.pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section
from ingestion.section_detector import build_candidates, confirm_headings_with_llm, assemble_sections
from ingestion.chunker import chunk_sections
from ingestion.embedder import embed_and_store

paper_name = "rag"
pdf_path = "data/Retrieval-Augmented Generation for.pdf"

print("Parsing...")
result = extract_text_from_pdf(pdf_path)
full_text = result["full_text"]
cleaned_text = remove_credits_block(full_text)
cleaned_text = remove_references_section(cleaned_text)

print("Detecting sections...")
pages = result["pages"]
candidates = build_candidates(pages)
confirmed = confirm_headings_with_llm(candidates)
sections = assemble_sections(pages, confirmed, candidates)

print("Chunking...")
chunks = chunk_sections(sections, chunk_size=256, overlap=50)

print(f"Embedding {len(chunks)} chunks...")
embed_and_store(chunks, paper_name)

print("Done!")

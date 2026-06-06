import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
INPUT_PDF = os.path.join(ROOT_DIR, "data", "Retrieval-Augmented Generation for.pdf")
OUTPUT_TXT = os.path.join(ROOT_DIR, "output", "rag_extracted.txt")
SECTIONS_OUTPUT = os.path.join(ROOT_DIR, "output", "rag_sections.txt")

from ingestion.pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section
from ingestion.section_detector import build_candidates, confirm_headings_with_llm, assemble_sections

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"[ERROR] PDF not found: {INPUT_PDF}")
        return

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    print("Step 1: Extracting text...")
    result = extract_text_from_pdf(INPUT_PDF)
    print(f"Total Pages Extracted: {result['total_pages']}")
    
    text = result["full_text"]
    text = remove_credits_block(text)
    text = remove_references_section(text)
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted text saved to: {OUTPUT_TXT}")

    print("\nStep 2: Detecting Sections...")
    candidates = build_candidates(result["pages"])
    confirmed = confirm_headings_with_llm(candidates)
    sections = assemble_sections(result["pages"], confirmed, candidates)
    
    print(f"Found {len(sections)} sections.")
    
    with open(SECTIONS_OUTPUT, "w", encoding="utf-8") as f:
        for section in sections:
            f.write(f"--- {section['heading']} (Page {section['page_num']}) ---\n")
            f.write(section['body'][:300] + "...\n\n")
            
    print(f"Section preview saved to: {SECTIONS_OUTPUT}")
    print("\nPlease verify the extracted text and sections output files before we proceed to Chunking and Embedding.")

if __name__ == "__main__":
    main()

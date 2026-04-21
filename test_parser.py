from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.section_detector import (
    build_candidates,
    confirm_headings_with_llm,
    assemble_sections
)

PDF_PATH = r"data\Attention is all you need.pdf"

result = extract_text_from_pdf(PDF_PATH)
candidates = build_candidates(result["pages"])
confirmed = confirm_headings_with_llm(candidates)
sections = assemble_sections(result["pages"], confirmed, candidates)

for section in sections:
    print(f"\n{'='*50}")
    print(f"[{section['type']}] {section['heading']} (page {section['page_num']})")
    print(f"{'-'*50}")
    print(section['body'][:300])
    print("...")
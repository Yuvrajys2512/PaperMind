from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.section_detector import build_candidates

PDF_PATH = r"data\Attention is all you need.pdf"

result = extract_text_from_pdf(PDF_PATH)
candidates = build_candidates(result["pages"])

print(f"Total candidates found: {len(candidates)}\n")

for c in candidates:
    print(f"[Page {c['page_num']}] Score:{c['score']} — {c['candidate_line']}")
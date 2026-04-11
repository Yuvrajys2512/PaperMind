from ingestion.pdf_parser import extract_text_from_pdf, remove_references_section

PDF_PATH = r"data\Attention is all you need.pdf"

result = extract_text_from_pdf(PDF_PATH)

print(f"Total pages found: {result['total_pages']}")
print(f"Pages with extractable text: {len(result['pages'])}")
print(f"Total characters extracted: {len(result['full_text'])}")

print("\n--- FIRST 800 CHARACTERS ---")
print(result['full_text'][:800])

print("\n--- AFTER REMOVING REFERENCES (last 500 chars) ---")
clean_text = remove_references_section(result['full_text'])
print(clean_text[-500:])
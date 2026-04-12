from ingestion.pdf_parser import (
    extract_text_from_pdf,
    remove_credits_block,
    remove_references_section
)

PDF_PATH = r"data\Attention is all you need.pdf"

result = extract_text_from_pdf(PDF_PATH)

print(f"Pages extracted: {result['total_pages']}")
print(f"Total characters: {len(result['full_text'])}")

# Apply filters
clean = remove_credits_block(result['full_text'])
clean = remove_references_section(clean)

print("\n--- FIRST 1000 CHARACTERS ---")
print(clean[:1000])

print("\n--- LAST 300 CHARACTERS ---")
print(clean[-300:])
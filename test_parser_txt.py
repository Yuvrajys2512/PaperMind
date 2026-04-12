from ingestion.pdf_parser import (
    extract_text_from_pdf,
    remove_credits_block,
    remove_references_section
)

PDF_PATH = r"data\Attention is all you need.pdf"
OUTPUT_PATH = r"data\extracted_output.txt"

result = extract_text_from_pdf(PDF_PATH)

clean = remove_credits_block(result['full_text'])
clean = remove_references_section(clean)

# Write full output to a text file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(clean)

print(f"Done. {result['total_pages']} pages processed.")
print(f"Total characters after cleaning: {len(clean)}")
print(f"Output written to {OUTPUT_PATH}")
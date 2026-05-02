from ingestion.ingest_document import ingest_document

result = ingest_document(
    pdf_path="data/Attention is all you need.pdf",   # swap in a real PDF you've used before
    paper_name="Attention is all you need"
)
print(result)
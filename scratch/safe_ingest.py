import sys
import traceback
sys.path.insert(0, r"c:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind")
print("Starting script...")
try:
    from ingestion.pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section
    from ingestion.section_detector import build_candidates, confirm_headings_with_llm, assemble_sections
    from ingestion.chunker import chunk_sections
    from ingestion.embedder import embed_and_store

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

    print("Embedding...")
    embed_and_store(chunks, paper_name)

    print("Done!")
except Exception as e:
    print("Exception occurred:")
    traceback.print_exc()

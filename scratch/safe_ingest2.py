import sys
import traceback
sys.path.insert(0, r"c:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind")

log_file = open("safe_log.txt", "w")
def log(msg):
    log_file.write(msg + "\n")
    log_file.flush()

log("Starting script...")
try:
    from ingestion.pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section
    log("Imported pdf_parser")
    from ingestion.section_detector import build_candidates, confirm_headings_with_llm, assemble_sections
    log("Imported section_detector")
    from ingestion.chunker import chunk_sections
    from ingestion.embedder import embed_and_store
    log("Imported all")

    paper_name = "rag"
    pdf_path = "data/Retrieval-Augmented Generation for.pdf"

    log("Extracting...")
    result = extract_text_from_pdf(pdf_path)
    full_text = result["full_text"]
    cleaned = remove_references_section(remove_credits_block(full_text))

    log("Detecting sections...")
    candidates = build_candidates(result["pages"])
    confirmed = confirm_headings_with_llm(candidates)
    sections = assemble_sections(result["pages"], confirmed, candidates)

    log("Chunking...")
    chunks = chunk_sections(sections, chunk_size=256, overlap=50)
    log(f"Chunks: {len(chunks)}")

    log("Embedding...")
    embed_and_store(chunks, paper_name)

    log("Done!")
except Exception as e:
    log("Exception occurred:")
    log(traceback.format_exc())

import sys
import shutil
from pydantic import BaseModel

# Ensure Unicode LLM output never crashes the server on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from ingestion.pipeline import answer_query
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.storage import (
    create_paper_record,
    update_paper_status,
    get_paper,
    list_papers,
    get_paper_pdf_path,
)
from ingestion.ingest_document import ingest_document

app = FastAPI(
    title="PaperMind API",
    description="AI-powered research paper Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_ingestion(paper_id: str, pdf_path: str, paper_name: str):
    """
    Background task — runs after the HTTP response is already sent.
    Calls the full ingestion pipeline and updates the paper status.
    """
    result = ingest_document(pdf_path=pdf_path, paper_name=paper_id)
    if result["success"]:
        update_paper_status(paper_id, "ready")
    else:
        update_paper_status(paper_id, "failed", error=result["error"])


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Create the paper record (status: processing)
    paper_id = create_paper_record(file.filename)

    # Save the PDF to disk
    pdf_path = get_paper_pdf_path(paper_id)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Kick off ingestion in the background
    background_tasks.add_task(
        run_ingestion,
        paper_id=paper_id,
        pdf_path=str(pdf_path),
        paper_name=paper_id
    )

    return {
        "paper_id": paper_id,
        "filename": file.filename,
        "status": "processing"
    }

@app.get("/status/{paper_id}")
def get_status(paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@app.get("/papers")
def get_all_papers():
    return list_papers()

class QueryRequest(BaseModel):
    paper_id: str
    question: str


@app.post("/query")
def query_paper(request: QueryRequest):
    paper = get_paper(request.paper_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    if paper["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Paper is not ready yet. Current status: {paper['status']}"
        )

    result = answer_query(request.question, request.paper_id)
    return result

 
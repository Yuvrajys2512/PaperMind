import json
import os
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path

PAPERS_DIR = Path("data/papers")
REGISTRY_FILE = Path("data/papers.json")

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE.touch(exist_ok=True)

# Single lock for all registry mutations — prevents concurrent uploads
# from overwriting each other's records in data/papers.json.
_registry_lock = threading.Lock()


def _load_registry() -> dict:
    content = REGISTRY_FILE.read_text().strip()
    if not content:
        return {}
    return json.loads(content)


def _save_registry(registry: dict):
    # Atomic write: a crash mid-write would otherwise leave papers.json
    # half-flushed and unparseable on the next read.
    tmp = REGISTRY_FILE.with_suffix(REGISTRY_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    os.replace(tmp, REGISTRY_FILE)


def create_paper_record(original_filename: str, source_id: str = None) -> str:
    """Creates a new paper entry with status 'processing'. Returns the paper_id."""
    paper_id = str(uuid.uuid4())
    with _registry_lock:
        registry = _load_registry()
        record = {
            "paper_id": paper_id,
            "filename": original_filename,
            "status": "processing",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        }
        if source_id:
            record["source_id"] = source_id
        registry[paper_id] = record
        _save_registry(registry)
    return paper_id


def update_paper_status(paper_id: str, status: str, error: str = None):
    """Updates a paper's status to 'ready' or 'failed'."""
    with _registry_lock:
        registry = _load_registry()
        if paper_id not in registry:
            return
        registry[paper_id]["status"] = status
        if status == "ready":
            registry[paper_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        if error:
            registry[paper_id]["error"] = error
        _save_registry(registry)


def get_paper(paper_id: str) -> dict | None:
    """Returns the paper record, or None if not found."""
    return _load_registry().get(paper_id)


def list_papers() -> list:
    """Returns all paper records as a list, newest first."""
    papers = list(_load_registry().values())
    papers.sort(key=lambda p: p["uploaded_at"], reverse=True)
    return papers


def get_paper_pdf_path(paper_id: str) -> Path:
    """Returns the path where a paper's PDF is/should be stored."""
    return PAPERS_DIR / f"{paper_id}.pdf"


def delete_paper_record(paper_id: str) -> bool:
    """Removes a paper from the registry. Returns True if it existed."""
    with _registry_lock:
        registry = _load_registry()
        if paper_id not in registry:
            return False
        del registry[paper_id]
        _save_registry(registry)
    return True

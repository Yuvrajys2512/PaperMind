import os

from api.storage import (
    download_pdf_to_tempfile,
    get_paper,
    list_ready_paper_ids,
    update_paper_status,
)
from api.usage import record_usage
from ingestion.ingest_document import ingest_document
from ingestion.llm_client import get_stats, reset_stats


def run_ingestion_from_storage(paper_id: str, kind: str = "upload"):
    """Downloads paper_id's PDF from R2 into a temp file, ingests it, and
    updates its status. Shared by the direct-upload and discovery-import
    flows so neither duplicates the download/cleanup logic.

    `kind` is the usage-event label: 'upload' for a real user action,
    'regenerate' when rebuilding a lost Chroma collection on startup — so the
    §2 cost dashboard doesn't conflate a server-side rebuild with a new upload.
    """
    reset_stats()
    temp_path = download_pdf_to_tempfile(paper_id)
    try:
        result = ingest_document(pdf_path=temp_path, paper_name=paper_id)
        if result["success"]:
            update_paper_status(paper_id, "ready")
        else:
            update_paper_status(paper_id, "failed", error=result["error"])
    finally:
        os.remove(temp_path)

    stats = get_stats()
    paper = get_paper(paper_id)
    if paper:
        record_usage(
            user_id=paper["user_id"],
            kind=kind,
            paper_id=paper_id,
            llm_calls=stats["call_count"],
            tokens_in=stats["tokens_in"],
            tokens_out=stats["tokens_out"],
            cost_usd=stats["cost_usd"],
        )


def regenerate_missing_collections():
    """Rebuild any 'ready' paper whose Chroma collection is missing from local
    disk (e.g. after an ephemeral-disk host wiped data/chroma_db on redeploy).
    PDFs live in R2 and the registry in Neon, so the index is fully
    regenerable. Idempotent: papers that already have a collection are skipped,
    making this a near-instant no-op on a warm or persistent disk."""
    from ingestion.retriever import collection_exists

    ready = list_ready_paper_ids()
    missing = [pid for pid in ready if not collection_exists(pid)]
    print(
        f"[regenerate] {len(ready)} ready paper(s), "
        f"{len(missing)} missing collection(s) to rebuild."
    )
    for pid in missing:
        try:
            print(f"[regenerate] rebuilding {pid} from R2…")
            run_ingestion_from_storage(pid, kind="regenerate")
        except Exception as exc:
            print(f"[regenerate] failed to rebuild {pid}: {exc}")

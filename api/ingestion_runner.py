import os
import time

from api.storage import (
    download_pdf_to_tempfile,
    get_paper,
    list_ready_paper_ids,
    update_paper_status,
)
from api.usage import record_usage
from api.logger import log_operation
from api.concurrency import paper_locked
from ingestion.ingest_document import ingest_document
from ingestion.llm_client import get_stats, reset_stats


def run_ingestion_from_storage(paper_id: str, kind: str = "upload"):
    """Downloads paper_id's PDF from R2 into a temp file, ingests it, and
    updates its status. Shared by the direct-upload and discovery-import
    flows so neither duplicates the download/cleanup logic.

    `kind` is the usage-event label: 'upload' for a real user action,
    'regenerate' when rebuilding a lost Chroma collection on startup — so the
    §2 cost dashboard doesn't conflate a server-side rebuild with a new upload.

    Acquires exclusive lock on paper_id to prevent concurrent ingestion/queries.
    """
    t0 = time.monotonic()

    # Acquire lock to ensure no concurrent operations on this paper
    with paper_locked(paper_id):
        reset_stats()
        temp_path = None
        try:
            temp_path = download_pdf_to_tempfile(paper_id)
            result = ingest_document(pdf_path=temp_path, paper_name=paper_id)
            if result["success"]:
                update_paper_status(paper_id, "ready")
                log_operation(
                    "ingest_document",
                    "success",
                    context={"paper_id": paper_id, "kind": kind},
                    duration_ms=round((time.monotonic() - t0) * 1000),
                )
            else:
                update_paper_status(paper_id, "failed", error=result["error"])
                log_operation(
                    "ingest_document",
                    "error",
                    context={"paper_id": paper_id, "kind": kind},
                    error=result["error"],
                    duration_ms=round((time.monotonic() - t0) * 1000),
                )
        except Exception as exc:
            update_paper_status(paper_id, "failed", error=f"Ingestion pipeline error: {exc}")
            log_operation(
                "ingest_document",
                "error",
                context={"paper_id": paper_id, "kind": kind},
                error=exc,
                duration_ms=round((time.monotonic() - t0) * 1000),
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as exc:
                    log_operation(
                        "cleanup_temp_file",
                        "error",
                        context={"paper_id": paper_id, "temp_path": temp_path},
                        error=exc,
                    )

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
            log_operation(
                "regenerate_missing_collections",
                "error",
                context={"paper_id": pid},
                error=exc,
            )
            print(f"[regenerate] failed to rebuild {pid}: {exc}")

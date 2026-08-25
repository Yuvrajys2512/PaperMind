import os
import time

from api.storage import (
    download_pdf_to_tempfile,
    get_chroma_snapshot,
    get_paper,
    list_ready_paper_ids,
    update_paper_status,
    upload_chroma_snapshot,
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
                _snapshot_chroma_collection(paper_id)
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
        # A startup rebuild is server-side work, not something the user did —
        # billing it to their account (kind='regenerate') would misattribute
        # cost/usage to someone who took no action. Real uploads/re-imports
        # still get recorded normally.
        if paper and kind != "regenerate":
            record_usage(
                user_id=paper["user_id"],
                kind=kind,
                paper_id=paper_id,
                llm_calls=stats["call_count"],
                tokens_in=stats["tokens_in"],
                tokens_out=stats["tokens_out"],
                cost_usd=stats["cost_usd"],
            )


def _snapshot_chroma_collection(paper_id: str) -> None:
    """Exports paper_id's freshly-built Chroma collection to R2 so a later
    startup rebuild (see regenerate_missing_collections) can restore it
    without re-running the LLM ingestion pipeline. Best-effort: an upload
    failure here must not fail the ingestion that just succeeded — worst case,
    the next rebuild falls back to a full re-ingest, same as before this
    existed."""
    from ingestion.chroma_snapshot import export_collection

    try:
        snapshot = export_collection(paper_id)
        upload_chroma_snapshot(paper_id, snapshot)
    except Exception as exc:
        log_operation(
            "upload_chroma_snapshot",
            "error",
            context={"paper_id": paper_id},
            error=exc,
        )


def regenerate_missing_collections():
    """Rebuild any 'ready' paper whose Chroma collection is missing from local
    disk (e.g. after an ephemeral-disk host wiped data/chroma_db on redeploy).

    Tries the R2 snapshot first (Launch Checklist 1.4): restoring one is a
    handful of local Chroma writes with zero LLM/embedding-model calls, unlike
    a full re-ingest (which re-runs LLM-backed section detection for every
    paper on every redeploy). Only papers ingested before snapshotting shipped
    — or whose snapshot upload never succeeded — fall back to the old full
    re-ingest path.

    PDFs live in R2 and the registry in Neon, so the index is fully
    regenerable either way. Idempotent: papers that already have a collection
    are skipped, making this a near-instant no-op on a warm or persistent disk.
    """
    from ingestion.chroma_snapshot import restore_collection
    from ingestion.retriever import collection_exists

    ready = list_ready_paper_ids()
    missing = [pid for pid in ready if not collection_exists(pid)]
    print(
        f"[regenerate] {len(ready)} ready paper(s), "
        f"{len(missing)} missing collection(s) to rebuild."
    )
    for pid in missing:
        snapshot = get_chroma_snapshot(pid)
        if snapshot is not None:
            try:
                print(f"[regenerate] restoring {pid} from Chroma snapshot…")
                restore_collection(pid, snapshot)
                continue
            except Exception as exc:
                log_operation(
                    "restore_chroma_snapshot",
                    "error",
                    context={"paper_id": pid},
                    error=exc,
                )
                print(f"[regenerate] snapshot restore failed for {pid}, falling back to re-ingest: {exc}")

        try:
            print(f"[regenerate] rebuilding {pid} from R2 (no snapshot available)…")
            run_ingestion_from_storage(pid, kind="regenerate")
        except Exception as exc:
            log_operation(
                "regenerate_missing_collections",
                "error",
                context={"paper_id": pid},
                error=exc,
            )
            print(f"[regenerate] failed to rebuild {pid}: {exc}")

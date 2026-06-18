"""
scripts/seed_demo_papers.py

One-time seeder for the shared, read-only sample papers that every signed-in
user sees in their library (Launch Checklist §5). The papers are owned by the
fixed system user `storage.DEMO_USER_ID`; api/main.py surfaces them via
list_demo_papers() with `is_demo: True`, shows them read-only, and excludes
them from per-user quotas.

This runs the *real* ingestion pipeline (PDF parse → chunk → embed → index →
R2 upload), so it makes live LLM calls and needs the same env as the server:
DATABASE_URL, the R2_* vars, and at least one LLM provider key. Run it locally
or wherever those are configured — not in CI.

Usage
-----
    # Seed one or more local PDFs into the shared demo set
    python scripts/seed_demo_papers.py path/to/attention.pdf path/to/bert.pdf

    # Show what's already in the demo set
    python scripts/seed_demo_papers.py --list

    # Remove a demo paper (PDF from R2 + registry row + Chroma collection)
    python scripts/seed_demo_papers.py --delete <paper_id>

Note: ingested demo papers get a normal Chroma collection, so the server's
startup regeneration (regenerate_missing_collections) rebuilds them from R2
on an ephemeral-disk host exactly like any user paper — no special-casing.
"""

import argparse
import os
import sys

# Windows' default cp1252 stdout can't encode non-Latin-1 glyphs; force UTF-8
# so progress/status lines never crash the run (same guard as api/main.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow `python scripts/seed_demo_papers.py` from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.storage import (  # noqa: E402
    DEMO_USER_ID,
    create_paper_record,
    delete_paper_record,
    delete_pdf,
    get_paper,
    list_demo_papers,
    upload_pdf,
)
from api.ingestion_runner import run_ingestion_from_storage  # noqa: E402
from api.uploads import PDF_MAGIC  # noqa: E402


def _is_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(PDF_MAGIC)).startswith(PDF_MAGIC)
    except OSError:
        return False


def cmd_list() -> int:
    papers = list_demo_papers()
    if not papers:
        print(f"No demo papers yet (owner user_id = {DEMO_USER_ID!r}).")
        return 0
    print(f"{len(papers)} demo paper(s) owned by {DEMO_USER_ID!r}:\n")
    for p in papers:
        print(f"  {p['paper_id']}  [{p['status']:<10}] {p['filename']}")
    return 0


def cmd_delete(paper_id: str) -> int:
    paper = get_paper(paper_id)
    if not paper:
        print(f"[error] No paper with id {paper_id}.")
        return 1
    if paper.get("user_id") != DEMO_USER_ID:
        # Guardrail: this script only ever touches the demo set, never a real
        # user's paper — deletion of those goes through the authenticated API.
        print(f"[error] {paper_id} is not a demo paper (owner = {paper.get('user_id')!r}). Refusing.")
        return 1

    try:
        delete_pdf(paper_id)
    except Exception as exc:
        print(f"[warn] R2 delete failed (continuing): {exc}")
    delete_paper_record(paper_id)
    try:
        import chromadb
        from ingestion.retriever import collection_name
        chromadb.PersistentClient(path="data/chroma_db").delete_collection(
            name=collection_name(paper_id)
        )
    except Exception as exc:
        print(f"[warn] Chroma collection drop skipped: {exc}")
    print(f"Deleted demo paper {paper_id}.")
    return 0


def cmd_seed(pdf_paths: list[str]) -> int:
    failures = 0
    for path in pdf_paths:
        name = os.path.basename(path)
        if not os.path.isfile(path):
            print(f"[skip] {path}: file not found.")
            failures += 1
            continue
        if not _is_pdf(path):
            print(f"[skip] {path}: not a PDF (missing %PDF- header).")
            failures += 1
            continue

        print(f"\n=== Seeding {name} ===")
        paper_id = create_paper_record(name, DEMO_USER_ID)
        print(f"  paper_id = {paper_id}")
        try:
            upload_pdf(paper_id, path)
            print("  uploaded to R2, running ingestion (LLM calls)…")
            # kind='seed' keeps these out of the §2 per-user upload cost view.
            run_ingestion_from_storage(paper_id, kind="seed")
        except Exception as exc:
            print(f"  [error] ingestion failed: {exc}")
            failures += 1
            continue

        status = (get_paper(paper_id) or {}).get("status")
        if status == "ready":
            print(f"  [ok] ready - {name} is now in the shared demo set.")
        else:
            print(f"  [error] finished with status {status!r}; check the registry.")
            failures += 1

    print(f"\nDone. {len(pdf_paths) - failures} ok, {failures} failed.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed/manage PaperMind's shared demo papers.")
    parser.add_argument("pdfs", nargs="*", help="PDF file paths to ingest into the demo set.")
    parser.add_argument("--list", action="store_true", help="List the current demo papers and exit.")
    parser.add_argument("--delete", metavar="PAPER_ID", help="Delete a demo paper by id and exit.")
    args = parser.parse_args()

    if args.list:
        return cmd_list()
    if args.delete:
        return cmd_delete(args.delete)
    if not args.pdfs:
        parser.error("provide one or more PDF paths, or use --list / --delete.")
    return cmd_seed(args.pdfs)


if __name__ == "__main__":
    raise SystemExit(main())

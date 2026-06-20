"""
scripts/manual_checks/audit_smoke.py

Manual smoke test for the claim-audit pipeline. Drives ingestion.claim_auditor
.audit_paper() directly against a real local Chroma collection — no HTTP, auth,
or DB required — so the extract → retrieve → verdict path can be validated end
to end with live LLM providers.

Usage:
    venv/Scripts/python.exe scripts/manual_checks/audit_smoke.py [paper_id]

With no argument it picks the local collection with the most chunks.
"""

import sys
from pathlib import Path

# The Windows console is cp1252; paper text carries ligatures/unicode. Match the
# server (api/main.py) and force utf-8 so printing the report can't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingestion.retriever import client, get_all_chunks
from ingestion.claim_auditor import audit_paper


def _pick_paper_id() -> str | None:
    """Return the collection name (== paper_id, the transform is idempotent for
    UUIDs) with the most chunks, so we audit a paper that actually has content."""
    best, best_n = None, -1
    for col in client.list_collections():
        try:
            n = col.count()
        except Exception:
            n = 0
        print(f"  collection {col.name}  ({n} chunks)")
        if n > best_n:
            best, best_n = col.name, n
    return best


def main():
    paper_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not paper_id:
        print("No paper_id given — scanning local collections:")
        paper_id = _pick_paper_id()
    if not paper_id:
        print("No collections found on disk. Ingest a paper first.")
        return

    print(f"\nAuditing paper_id = {paper_id}\n" + "=" * 60)

    def on_progress(evt):
        print(f"  [{evt['stage']:>11}] {evt['message']}")

    report = audit_paper(paper_id, on_progress=on_progress)

    print("=" * 60)
    if report.get("audit_failed"):
        print(f"AUDIT FAILED: {report.get('reason')}")
        return

    print(
        f"claims_checked={report['claims_checked']}  "
        f"grounded={report['grounded']}  flagged={report['flagged']}  "
        f"trust_score={report['trust_score']}"
    )
    print("-" * 60)
    for i, c in enumerate(report["claims"], 1):
        print(f"\n[{i}] {c['verdict']} ({c['severity']}) — {c['type']} / {c['source_section']}")
        print(f"    CLAIM: {c['claim']}")
        if c["why"]:
            print(f"    WHY:   {c['why']}")
        if c["evidence"]:
            ev = c["evidence"]
            print(f"    EVID:  [{ev['section_type']}] {ev['section']} p.{ev['page']}")
            print(f"           {ev['quote'][:160].strip()}…")


if __name__ == "__main__":
    main()

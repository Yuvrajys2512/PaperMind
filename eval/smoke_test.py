"""
eval/smoke_test.py — Phase 1 end-to-end smoke test for the QASPER harness.

Proves the full path works before scoring is built (Phase B):

    download dev split  ->  ingest N papers  ->  run K questions each through
    the REAL PaperMind pipeline (answer_query)  ->  print gold vs predicted.

This makes real LLM calls, so defaults are intentionally tiny. Run from the
project root with the venv interpreter so the relative ``data/chroma_db`` path
resolves for both ingest and retrieval:

    venv/Scripts/python.exe -m eval.smoke_test --papers 1 --qs 0   # ingest only
    venv/Scripts/python.exe -m eval.smoke_test --papers 2 --qs 2   # full path
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# QASPER answers and paper text contain non-cp1252 characters; keep Windows
# stdout from crashing on them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval.qasper_loader import load_papers, iter_questions  # noqa: E402
from eval.qasper_adapter import ingest_qasper_paper  # noqa: E402
from ingestion.pipeline import answer_query  # noqa: E402


def _clip(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + " …"


def main() -> None:
    ap = argparse.ArgumentParser(description="QASPER Phase 1 smoke test")
    ap.add_argument("--papers", type=int, default=2, help="papers to ingest")
    ap.add_argument("--qs", type=int, default=2, help="questions per paper to run")
    args = ap.parse_args()

    print("[smoke] loading QASPER dev split ...")
    papers = load_papers("dev")

    picked = []
    for pid, paper in papers.items():
        if paper.get("qas") and paper.get("full_text"):
            picked.append((pid, paper))
        if len(picked) >= args.papers:
            break
    print(f"[smoke] selected {len(picked)} papers\n")

    total = answered = 0
    for pid, paper in picked:
        print("=" * 80)
        print(f"PAPER {pid}  —  {_clip(paper.get('title', ''), 80)}")

        summary = ingest_qasper_paper(pid, paper)
        line = (f"  ingest: success={summary['success']} "
                f"sections={summary['num_sections']} chunks={summary['num_chunks']}")
        if summary.get("error"):
            line += f"  ERROR={summary['error']}"
        print(line)
        if not summary["success"]:
            continue

        for qi, (question, _qid, answers) in enumerate(iter_questions(paper)):
            if qi >= args.qs:
                break
            gold = answers[0] if answers else {"text": "(none)", "answerable": None, "type": "?"}
            total += 1
            print("-" * 80)
            print(f"  Q: {_clip(question, 200)}")
            print(f"  GOLD [{gold['type']}/answerable={gold['answerable']}]: {_clip(gold['text'])}")
            t0 = time.time()
            try:
                res = answer_query(question, pid)
                dt = (time.time() - t0) * 1000
                print(f"  PRED ({dt:.0f}ms, conf={res.get('confidence')}): "
                      f"{_clip(res.get('answer', ''))}")
                answered += 1
            except Exception as e:  # smoke test must never abort mid-run
                print(f"  PRED ERROR: {type(e).__name__}: {e}")

    print("=" * 80)
    print(f"[smoke] done — {answered}/{total} questions answered without error")


if __name__ == "__main__":
    main()

"""
eval/run_eval.py — QASPER evaluation runner (Phase B).

Ingests papers, runs each question through the real PaperMind pipeline, scores
the result with eval/metrics.py (+ optional LLM-as-judge), and writes one JSONL
row per question plus an aggregate summary.

Usage (project root, venv interpreter):
  venv/Scripts/python.exe -m eval.run_eval --papers 3 --qs 5
  venv/Scripts/python.exe -m eval.run_eval --papers 5 --qs 0 --judge      # all qs, with judge
  venv/Scripts/python.exe -m eval.run_eval --papers 1 --qs 3 --no-evidence

Metrics
  Answer-F1            SQuAD token-F1 vs. annotator references (max over them).
  Answerable accuracy  Did the system correctly answer vs. abstain?
  Evidence recall/F1   Overlap of dense-retrieved top-k context with gold
                       evidence paragraphs (retrieval-quality probe).
  Judge (optional)     CORRECT/PARTIAL/INCORRECT on answerable questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval.qasper_loader import load_papers, iter_questions
from eval.qasper_adapter import ingest_qasper_paper
from eval import metrics
from ingestion.pipeline import answer_query
from ingestion.retriever import retrieve

_RESULTS_DIR = _ROOT / "eval" / "results"


def _gold_type(answers: list[dict]) -> str:
    """One label per question for per-type breakdown."""
    if not metrics.gold_is_answerable(answers):
        return "unanswerable"
    types = [a["type"] for a in answers if a.get("answerable")]
    return Counter(types).most_common(1)[0][0] if types else "unanswerable"


def _gold_evidence(answers: list[dict]) -> list[str]:
    """Union of evidence paragraphs across all annotators (deduplicated)."""
    seen, union = set(), []
    for a in answers:
        for ev in a.get("evidence", []):
            if ev and ev.strip() and ev not in seen:
                seen.add(ev)
                union.append(ev)
    return union


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else ("—" if x is None else str(x))


def evaluate_question(paper_id, question, qid, answers, *, use_evidence, use_judge, topk):
    """Run + score a single question. Returns a JSONL-ready dict."""
    gold_evidence = _gold_evidence(answers)
    gold_type = _gold_type(answers)

    t0 = time.time()
    res = answer_query(question, paper_id)
    wall_ms = round((time.time() - t0) * 1000)

    pred = res.get("answer", "")
    passed = bool(res.get("passed"))
    pred_no_answer = metrics.is_no_answer(pred, passed)

    row = {
        "paper_id": paper_id,
        "question_id": qid,
        "question": question,
        "gold_type": gold_type,
        "gold_answerable": metrics.gold_is_answerable(answers),
        "gold_answers": [{"type": a["type"], "answerable": a["answerable"], "text": a["text"]}
                         for a in answers],
        "n_gold_evidence": len(gold_evidence),
        "pred_answer": pred,
        "pred_original_answer": res.get("original_answer", ""),
        "predicted_no_answer": pred_no_answer,
        "passed": passed,
        "confidence": res.get("confidence"),
        "faithfulness": res.get("faithfulness"),
        "answer_relevancy": res.get("answer_relevancy"),
        "attempts": res.get("attempts"),
        "grading_removed": res.get("grading", {}).get("removed_count"),
        "duration_ms": res.get("duration_ms", wall_ms),
        "llm_calls": res.get("llm_calls"),
        "sources": res.get("sources", []),
        # scores
        "answer_f1": metrics.answer_f1(pred, pred_no_answer, answers),
        "answerable_correct": metrics.answerable_correct(pred_no_answer, answers),
        "evidence_recall": None,
        "evidence_f1": None,
        "judge_verdict": None,
        "judge_score": None,
    }

    if use_evidence and gold_evidence:
        retrieved = retrieve(question, paper_id, top_k=topk)
        texts = [r["text"] for r in retrieved]
        row["evidence_recall"] = metrics.evidence_recall(texts, gold_evidence)
        row["evidence_f1"] = metrics.evidence_token_f1(texts, gold_evidence)

    if use_judge and metrics.gold_is_answerable(answers) and not pred_no_answer:
        from eval.judge import judge_answer
        gold_texts = [a["text"] for a in answers if a.get("answerable")]
        verdict = judge_answer(question, gold_texts, pred)
        row["judge_verdict"] = verdict["verdict"]
        row["judge_score"] = verdict["score"]

    return row


def summarize(rows: list[dict]) -> dict:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["gold_type"]].append(r)

    summary = {
        "n_questions": len(rows),
        "answer_f1": _mean([r["answer_f1"] for r in rows]),
        "answerable_accuracy": _mean([1.0 if r["answerable_correct"] else 0.0 for r in rows]),
        "evidence_recall": _mean([r["evidence_recall"] for r in rows]),
        "evidence_f1": _mean([r["evidence_f1"] for r in rows]),
        "mean_confidence": _mean([r["confidence"] for r in rows]),
        "mean_duration_ms": _mean([float(r["duration_ms"]) for r in rows if r["duration_ms"]]),
        "by_type": {},
    }

    judged = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    if judged:
        summary["judge_accuracy"] = _mean(judged)
        summary["judge_dist"] = dict(Counter(
            r["judge_verdict"] for r in rows if r["judge_verdict"]))

    for gtype, group in by_type.items():
        summary["by_type"][gtype] = {
            "n": len(group),
            "answer_f1": _mean([r["answer_f1"] for r in group]),
            "answerable_accuracy": _mean([1.0 if r["answerable_correct"] else 0.0 for r in group]),
            "evidence_recall": _mean([r["evidence_recall"] for r in group]),
        }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("QASPER EVAL SUMMARY")
    print("=" * 72)
    print(f"  questions            : {summary['n_questions']}")
    print(f"  Answer-F1            : {_fmt(summary['answer_f1'])}")
    print(f"  Answerable accuracy  : {_fmt(summary['answerable_accuracy'])}")
    print(f"  Evidence recall@k    : {_fmt(summary['evidence_recall'])}")
    print(f"  Evidence token-F1    : {_fmt(summary['evidence_f1'])}")
    if "judge_accuracy" in summary:
        print(f"  Judge accuracy       : {_fmt(summary['judge_accuracy'])}  {summary.get('judge_dist', {})}")
    print(f"  Mean confidence      : {_fmt(summary['mean_confidence'])}")
    print(f"  Mean latency (ms)    : {_fmt(summary['mean_duration_ms'])}")
    print("-" * 72)
    print(f"  {'type':<14}{'n':>4}{'ans-F1':>10}{'answerable':>12}{'ev-recall':>11}")
    for gtype, s in sorted(summary["by_type"].items()):
        print(f"  {gtype:<14}{s['n']:>4}{_fmt(s['answer_f1']):>10}"
              f"{_fmt(s['answerable_accuracy']):>12}{_fmt(s['evidence_recall']):>11}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="QASPER evaluation runner")
    ap.add_argument("--papers", type=int, default=3, help="papers to evaluate")
    ap.add_argument("--qs", type=int, default=5, help="questions per paper (0 = all)")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--topk", type=int, default=10, help="evidence retrieval depth")
    ap.add_argument("--judge", action="store_true", help="enable LLM-as-judge")
    ap.add_argument("--no-evidence", action="store_true", help="skip evidence scoring")
    ap.add_argument("--out", default=None, help="JSONL output path")
    args = ap.parse_args()

    print(f"[eval] loading QASPER {args.split} split ...")
    papers = load_papers(args.split)
    picked = []
    for pid, paper in papers.items():
        if paper.get("qas") and paper.get("full_text"):
            picked.append((pid, paper))
        if len(picked) >= args.papers:
            break

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else _RESULTS_DIR / f"qasper_{args.split}_{ts}.jsonl"

    rows: list[dict] = []
    print(f"[eval] {len(picked)} papers | judge={args.judge} | evidence={not args.no_evidence}")
    with open(out_path, "w", encoding="utf-8") as fout:
        for pid, paper in picked:
            print(f"\n[eval] ingesting {pid} — {(paper.get('title') or '')[:70]}")
            summary = ingest_qasper_paper(pid, paper)
            if not summary["success"]:
                print(f"[eval]   SKIP (ingest failed): {summary.get('error')}")
                continue

            for qi, (question, qid, answers) in enumerate(iter_questions(paper)):
                if args.qs and qi >= args.qs:
                    break
                if not answers:
                    continue
                try:
                    row = evaluate_question(
                        pid, question, qid, answers,
                        use_evidence=not args.no_evidence,
                        use_judge=args.judge,
                        topk=args.topk,
                    )
                except Exception as e:
                    print(f"[eval]   Q ERROR ({qid}): {type(e).__name__}: {e}")
                    continue

                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                jv = f" judge={row['judge_verdict']}" if row["judge_verdict"] else ""
                print(f"[eval]   [{row['gold_type'][:6]:<6}] F1={row['answer_f1']:.2f} "
                      f"ans_ok={int(row['answerable_correct'])} "
                      f"ev_rec={_fmt(row['evidence_recall'])}{jv}  "
                      f"Q: {question[:60]}")

    if not rows:
        print("[eval] no questions scored.")
        return

    summary = summarize(rows)
    print_summary(summary)

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[eval] rows  -> {out_path}")
    print(f"[eval] summary -> {summary_path}")


if __name__ == "__main__":
    main()

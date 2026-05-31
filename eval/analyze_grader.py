"""
eval/analyze_grader.py — Does the evidence grader help or hurt answer quality?

Paired design. For each question the pipeline returns BOTH an original
(pre-grader) answer and a cleaned (post-grader) answer from the *same*
generation call. We judge both against the gold reference and compare:

    helped   judge(cleaned) > judge(original)   grading removed a bad sentence
    hurt     judge(cleaned) < judge(original)   grading removed a GOOD sentence
    neutral  same verdict                        (incl. nothing removed)

Because both answers come from one generation, this isolates the grader's
effect with no between-config variance and no model confound — far more
sensitive than comparing the `full` vs `no_grader` ablation configs.

For every HURT case we print the sentences the grader removed: the direct
evidence of whether it is too aggressive.

Usage:
  venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval.qasper_loader import load_papers, select_papers, iter_questions
from eval.qasper_adapter import ingest_qasper_paper
from eval import metrics
from eval.judge import judge_answer
from ingestion.pipeline import answer_query

_RESULTS_DIR = _ROOT / "eval" / "results"


def _clip(s, n=150):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " …"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evidence-grader helped/hurt analysis")
    ap.add_argument("--papers", type=int, default=5)
    ap.add_argument("--qs", type=int, default=6, help="questions per paper (0 = all)")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--gen-model", default=None,
                    help="pin the generation model (e.g. llama-3.1-8b-instant). "
                         "Sets PAPERMIND_GEN_MODEL so only generation is affected.")
    args = ap.parse_args()

    if args.gen_model:
        import os
        os.environ["PAPERMIND_GEN_MODEL"] = args.gen_model
        print(f"[grader] generation pinned to: {args.gen_model}")

    print(f"[grader] loading QASPER {args.split}; ingesting {args.papers} papers ...")
    papers = load_papers(args.split)
    picked = select_papers(papers, args.papers)
    for pid, paper in picked:
        ing = ingest_qasper_paper(pid, paper)
        if not ing["success"]:
            print(f"[grader]   {pid}: ingest FAILED ({ing.get('error')})")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.gen_model.split('/')[-1]}" if args.gen_model else ""
    out_path = _RESULTS_DIR / f"grader_analysis_{args.split}{tag}_{ts}.jsonl"

    rows: list[dict] = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for pid, paper in picked:
            for qi, (question, qid, answers) in enumerate(iter_questions(paper)):
                if args.qs and qi >= args.qs:
                    break
                # Grader effect is only meaningful on answerable questions.
                if not answers or not metrics.gold_is_answerable(answers):
                    continue

                res = answer_query(question, pid)
                cleaned = res.get("answer", "")
                original = res.get("original_answer", "") or cleaned
                grades = res.get("grading", {}).get("grades", [])
                removed = [g["sentence"] for g in grades if not g.get("kept", True)]

                gold_texts = [a["text"] for a in answers if a.get("answerable")]

                j_clean = judge_answer(question, gold_texts, cleaned)
                j_orig = judge_answer(question, gold_texts, original)
                sc_clean, sc_orig = j_clean["score"], j_orig["score"]

                if sc_clean is None or sc_orig is None:
                    verdict = "unjudged"
                elif sc_clean > sc_orig:
                    verdict = "helped"
                elif sc_clean < sc_orig:
                    verdict = "hurt"
                else:
                    verdict = "neutral"

                row = {
                    "paper_id": pid, "question_id": qid, "question": question,
                    "gold": gold_texts,
                    "removed_count": len(removed),
                    "removed_sentences": removed,
                    "judge_original": j_orig["verdict"], "score_original": sc_orig,
                    "judge_cleaned": j_clean["verdict"], "score_cleaned": sc_clean,
                    "effect": verdict,
                }
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                tag = {"helped": "+", "hurt": "!!", "neutral": "·", "unjudged": "?"}[verdict]
                print(f"[grader] {tag:<2} removed={len(removed):<2} "
                      f"orig={j_orig['verdict']:<9} -> clean={j_clean['verdict']:<9} "
                      f"Q: {_clip(question, 55)}")

    _summarize(rows, out_path)


def _summarize(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("[grader] no answerable questions scored.")
        return

    effects = Counter(r["effect"] for r in rows)
    removed_any = [r for r in rows if r["removed_count"] > 0]

    def _mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "=" * 72)
    print("EVIDENCE GRADER — HELPED / HURT ANALYSIS")
    print("=" * 72)
    print(f"  questions (answerable)      : {len(rows)}")
    print(f"  questions w/ ≥1 removal     : {len(removed_any)}")
    print(f"  mean judge score (original) : {_mean('score_original')}")
    print(f"  mean judge score (cleaned)  : {_mean('score_cleaned')}")
    print("-" * 72)
    print(f"  HELPED  (grading improved)  : {effects.get('helped', 0)}")
    print(f"  HURT    (grading worsened)  : {effects.get('hurt', 0)}")
    print(f"  NEUTRAL (no change)         : {effects.get('neutral', 0)}")
    print("=" * 72)

    hurt = [r for r in rows if r["effect"] == "hurt"]
    if hurt:
        print("\nHURT CASES — sentences the grader removed from a better answer:")
        for r in hurt:
            print(f"\n  Q: {_clip(r['question'], 80)}")
            print(f"     {r['judge_original']} -> {r['judge_cleaned']}")
            for s in r["removed_sentences"]:
                print(f"     - removed: {_clip(s, 110)}")

    print(f"\n[grader] rows -> {out_path}")


if __name__ == "__main__":
    main()

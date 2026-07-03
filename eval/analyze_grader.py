"""
eval/analyze_grader.py — Does the evidence grader help or hurt answer quality?

Paired design. For each question the pipeline returns BOTH an original
(pre-grader) answer and a cleaned (post-grader) answer from the *same*
generation call. We judge both against the gold reference and compare:

    helped   judge(cleaned) > judge(original)   grading removed a bad sentence
    hurt     judge(cleaned) < judge(original)   grading removed a GOOD sentence
    neutral  same verdict, OR grader changed nothing (cleaned == original)

Because both answers come from one generation, this isolates the grader's
effect with no between-config variance and no model confound — far more
sensitive than comparing the `full` vs `no_grader` ablation configs.

IMPORTANT — only rows where the grader actually changed the text can be
helped/hurt. When nothing is removed, cleaned == original, so judging the two
separately would only sample judge variance and spuriously flip some rows.
Those rows are forced to neutral and judged once. See the text_changed guard.

For every HURT case we print the sentences the grader removed: the direct
evidence of whether it is too aggressive.

Usage:
  venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _process_question(pid: str, question: str, qid: str, answers: list) -> dict:
    """Run the pipeline + paired judge for one question.

    Pure worker: no shared state, no I/O — safe to run on a thread pool. Returns
    a dict that is either a finished result row (``kind="row"``) or a skip notice
    (``kind="skip"``) so the caller can log it. The pipeline is read-only here
    (all ingestion happens before the pool starts), so concurrent calls only
    share Chroma reads, which are safe.
    """
    try:
        res = answer_query(question, pid)
    except Exception as exc:
        return {"kind": "skip", "msg": f"SKIP  {_clip(question, 55)} ({exc})"}

    # Pipeline swallows generation errors and returns a graceful degradation
    # answer (confidence=0, warning set). Skip these — scoring a fallback string
    # against gold is meaningless.
    if res.get("confidence", 100) == 0.0 and res.get("warning"):
        return {"kind": "skip",
                "msg": f"SKIP  {_clip(question, 55)} (pipeline degradation: {res['warning'][:60]})"}

    cleaned = res.get("answer", "")
    original = res.get("original_answer", "") or cleaned
    grades = res.get("grading", {}).get("grades", [])
    removed = [g["sentence"] for g in grades if not g.get("kept", True)]

    gold_texts = [a["text"] for a in answers if a.get("answerable")]

    # When the grader changed nothing — no sentences removed, or grading failed
    # and returned the answer untouched — `cleaned` and `original` are the SAME
    # text. Judging them in two separate calls and comparing scores measures
    # judge variance, not the grader: a stochastic judge will flip some
    # identical-text rows to helped/hurt. Majority-vote only shrinks that noise;
    # it does not remove it. So short-circuit on text equality — identical text
    # is neutral by definition, and we judge only once.
    text_changed = cleaned.strip() != original.strip()
    if not text_changed:
        j_clean = j_orig = judge_answer(question, gold_texts, cleaned)
        sc_clean = sc_orig = j_clean["score"]
        verdict = "unjudged" if sc_clean is None else "neutral"
    else:
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
    sym = {"helped": "+", "hurt": "!!", "neutral": "·", "unjudged": "?"}[verdict]
    row_log = (f"{sym:<2} removed={len(removed):<2} "
               f"orig={j_orig['verdict']:<9} -> clean={j_clean['verdict']:<9} "
               f"Q: {_clip(question, 55)}")
    return {"kind": "row", "row": row, "msg": row_log}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evidence-grader helped/hurt analysis")
    ap.add_argument("--papers", type=int, default=5)
    ap.add_argument("--qs", type=int, default=6, help="questions per paper (0 = all)")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--gen-model", default=None,
                    help="pin the generation model (e.g. llama-3.1-8b-instant). "
                         "Sets PAPERMIND_GEN_MODEL so only generation is affected.")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent questions (default 4). Each question is "
                         "independent and read-only against Chroma, so the loop "
                         "is embarrassingly parallel — this multiplies throughput "
                         "on the latency-bound (Mistral) portion. Higher values "
                         "trip per-minute provider rate limits sooner; 1 = the "
                         "old fully-sequential behaviour.")
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

    # Flatten to a task list up front so the work is a flat pool of independent
    # units rather than nested loops — honours --qs per paper and the answerable
    # filter (the grader effect is only meaningful on answerable questions).
    tasks: list[tuple[str, str, str, list]] = []
    for pid, paper in picked:
        for qi, (question, qid, answers) in enumerate(iter_questions(paper)):
            if args.qs and qi >= args.qs:
                break
            if not answers or not metrics.gold_is_answerable(answers):
                continue
            tasks.append((pid, question, qid, answers))

    workers = max(1, args.workers)
    print(f"[grader] {len(tasks)} answerable questions; {workers} concurrent worker(s)")

    rows: list[dict] = []
    io_lock = threading.Lock()  # guards the output file, rows list, and stdout
    with open(out_path, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_question, *t) for t in tasks]
            for fut in as_completed(futures):
                out = fut.result()
                with io_lock:
                    if out["kind"] == "skip":
                        print(f"[grader] {out['msg']}")
                        continue
                    rows.append(out["row"])
                    fout.write(json.dumps(out["row"], ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"[grader] {out['msg']}")

    _summarize(rows, out_path)


def _summarize(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("[grader] no answerable questions scored.")
        return

    effects = Counter(r["effect"] for r in rows)
    # Rows where the grader actually altered the text — the only rows that can
    # be helped/hurt. If this is 0, helped/hurt are both 0 by construction.
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

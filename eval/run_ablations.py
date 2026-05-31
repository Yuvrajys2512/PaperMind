"""
eval/run_ablations.py — Phase 3 ablation matrix.

Runs the QASPER eval under several pipeline configurations and tabulates them
side by side. PaperMind's ablation flags are read at MODULE IMPORT time, so each
config must run in its own subprocess with the env set *before* import — that's
exactly what this orchestrator does.

Papers are ingested ONCE up front (chunking/embedding don't depend on the
flags); every config then queries the same vector store with --skip-ingest, and
evidence scoring is skipped (the dense-retrieval probe is flag-independent, so
it would be identical across configs).

Configs
  full       all features on (no flags)
  baseline   vanilla RAG — grader, rerank, HyDE, retry all OFF
  no_grader  evidence grading off        (PAPERMIND_DISABLE_GRADER=1)
  no_rerank  reranker off                (PAPERMIND_DISABLE_RERANK=1)
  no_hyde    HyDE query expansion off    (PAPERMIND_DISABLE_HYDE=1)
  no_retry   confidence-gated retry off  (PAPERMIND_MAX_ATTEMPTS=1)

Usage (project root, venv interpreter)
  venv/Scripts/python.exe -m eval.run_ablations --papers 3 --qs 5 --judge
  venv/Scripts/python.exe -m eval.run_ablations --papers 1 --qs 1 --configs full,baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval.qasper_loader import load_papers, select_papers
from eval.qasper_adapter import ingest_qasper_paper

_RESULTS_DIR = _ROOT / "eval" / "results"

# config name -> ablation env-flag overrides
CONFIGS: dict[str, dict] = {
    "full":      {},
    "baseline":  {"PAPERMIND_DISABLE_GRADER": "1", "PAPERMIND_DISABLE_RERANK": "1",
                  "PAPERMIND_DISABLE_HYDE": "1", "PAPERMIND_MAX_ATTEMPTS": "1"},
    "no_grader": {"PAPERMIND_DISABLE_GRADER": "1"},
    "no_rerank": {"PAPERMIND_DISABLE_RERANK": "1"},
    "no_hyde":   {"PAPERMIND_DISABLE_HYDE": "1"},
    "no_retry":  {"PAPERMIND_MAX_ATTEMPTS": "1"},
}


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else ("—" if x is None else str(x))


def run_config(name, env_overrides, *, papers, qs, judge, split, out_path) -> dict | None:
    """Run one config in a subprocess; return its parsed summary (or None)."""
    env = {**os.environ, **env_overrides}
    cmd = [sys.executable, "-m", "eval.run_eval",
           "--papers", str(papers), "--qs", str(qs), "--split", split,
           "--skip-ingest", "--no-evidence", "--out", str(out_path)]
    if judge:
        cmd.append("--judge")

    flags = env_overrides or "(none)"
    print(f"\n{'#' * 72}\n# CONFIG: {name}    flags={flags}\n{'#' * 72}")
    proc = subprocess.run(cmd, env=env, cwd=str(_ROOT))

    summary_path = out_path.with_suffix(".summary.json")
    if proc.returncode != 0 or not summary_path.exists():
        print(f"[ablate] config '{name}' produced no summary (exit {proc.returncode})")
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def print_matrix(results: dict[str, dict | None]) -> None:
    print("\n" + "=" * 92)
    print("ABLATION MATRIX")
    print("=" * 92)
    print(f"  {'config':<12}{'n':>4}{'ans-F1':>9}{'judge':>8}{'faith':>8}"
          f"{'answerbl':>10}{'lat(ms)':>10}{'llm':>7}")
    print("-" * 92)
    for name, s in results.items():
        if not s:
            print(f"  {name:<12}  (no data)")
            continue
        print(f"  {name:<12}{s['n_questions']:>4}"
              f"{_fmt(s['answer_f1']):>9}"
              f"{_fmt(s.get('judge_accuracy')):>8}"
              f"{_fmt(s.get('mean_faithfulness')):>8}"
              f"{_fmt(s['answerable_accuracy']):>10}"
              f"{_fmt(s.get('mean_duration_ms')):>10}"
              f"{_fmt(s.get('mean_llm_calls')):>7}")
    print("=" * 92)


def main() -> None:
    ap = argparse.ArgumentParser(description="QASPER ablation matrix")
    ap.add_argument("--papers", type=int, default=3)
    ap.add_argument("--qs", type=int, default=5, help="questions per paper (0 = all)")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--judge", action="store_true", help="enable LLM-as-judge per config")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS))
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        ap.error(f"unknown configs {unknown}; valid: {list(CONFIGS)}")

    # 1) Ingest the shared paper set ONCE — config-independent.
    print(f"[ablate] loading QASPER {args.split}; ingesting {args.papers} papers once ...")
    papers = load_papers(args.split)
    picked = select_papers(papers, args.papers)
    for pid, paper in picked:
        summ = ingest_qasper_paper(pid, paper)
        status = "ok" if summ["success"] else f"FAIL: {summ.get('error')}"
        print(f"[ablate]   {pid}: {status} ({summ['num_chunks']} chunks)")

    # 2) Run each config in its own subprocess (env set before import).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _RESULTS_DIR / f"ablation_{args.split}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict | None] = {}
    for name in names:
        results[name] = run_config(
            name, CONFIGS[name],
            papers=args.papers, qs=args.qs, judge=args.judge,
            split=args.split, out_path=run_dir / f"{name}.jsonl",
        )

    print_matrix(results)
    (run_dir / "matrix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[ablate] matrix -> {run_dir / 'matrix.json'}")


if __name__ == "__main__":
    main()

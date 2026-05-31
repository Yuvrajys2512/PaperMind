# Phase 3 — Ablation Matrix

**Date:** 2026-05-31
**Goal:** Run the QASPER eval under multiple pipeline configurations and tabulate them side by side — the experiment that turns the harness into the paper's results.

---

## Key design constraint

PaperMind's ablation flags are read at **module-import time** (`pipeline.py:47`, `query_router.py:31/35`, `retry_engine.py:25`). Setting env vars in-process after import has no effect. Therefore **each config runs in its own subprocess**, with the env set before import. Papers are ingested **once** up front (chunking/embedding are flag-independent); every config then queries the same vector store with `--skip-ingest`.

## What was built

| File | Change |
|------|--------|
| `eval/run_ablations.py` | New orchestrator: ingest the shared paper set once → run each config as a subprocess with its ablation flags → collect each `summary.json` → print + save a comparison matrix. |
| `eval/run_eval.py` | Added `--skip-ingest`; added `mean_faithfulness` + `mean_llm_calls` to the summary. |
| `eval/qasper_loader.py` | Added `select_papers()` — shared deterministic selection so the runner and orchestrator score the **identical** question set. |

### Configs

| Config | Flags |
|--------|-------|
| `full` | none (all features on) |
| `baseline` | grader + rerank + HyDE + retry all OFF (vanilla RAG) |
| `no_grader` | `PAPERMIND_DISABLE_GRADER=1` |
| `no_rerank` | `PAPERMIND_DISABLE_RERANK=1` |
| `no_hyde` | `PAPERMIND_DISABLE_HYDE=1` |
| `no_retry` | `PAPERMIND_MAX_ATTEMPTS=1` |

```
venv/Scripts/python.exe -m eval.run_ablations --papers 3 --qs 5 --judge
venv/Scripts/python.exe -m eval.run_ablations --papers 1 --qs 1 --configs full,baseline
```

Output lands in `eval/results/ablation_<split>_<ts>/` (per-config JSONL + summaries + `matrix.json`), gitignored.

## Validation (1 paper, 1 question, 3 configs)

```
  config         n   ans-F1   judge   faith  answerbl   lat(ms)    llm
  full           1    0.011       —   1.000     1.000   43764      5
  baseline       1    0.031       —   1.000     1.000   20995      3
  no_grader      1    0.027       —   1.000     1.000   31821      4
```

The numbers are meaningless at n=1, but the **machinery is proven**: the `llm` (LLM-call count) and `lat(ms)` columns differ per config exactly as the flags predict — `baseline` skips HyDE + grader + retry (3 calls), `no_grader` drops only the grader (4 calls), `full` runs everything (5 calls). The env flags are being read correctly in each subprocess.

---

## Status & next

**Phase 3: ✅ machinery complete and validated.** A meaningful matrix just needs to be *run* at scale (more papers/questions, `--judge` on).

**The blocker before a real run is throughput** (carried from Phases 1–2):
- Gemini is free-tier rate-limited.
- **Cerebras returns `NotFoundError` on every call** (wrong model name) — so it never contributes.
- Everything lands on Mistral at ~20–44 s/question.

A full matrix = `configs × papers × questions × (pipeline calls + judge)`. At 6 configs and even 3 papers × 5 questions that's ~90 pipeline runs × tens of seconds each — hours on a single rate-limited provider. **Fixing the provider chain (at minimum Cerebras) is the highest-leverage next step**: it widens throughput and de-risks rate-limit failures mid-run.

**Recommended order from here:**
1. Fix the Cerebras model name (and confirm Gemini limits) → restore real multi-provider throughput.
2. Run the ablation matrix at scale with `--judge`.
3. Analyze: does grading lift faithfulness / judged-correctness without dropping Answer-F1, and at what latency cost? → the paper's headline table.

# Phase 2 — QASPER Scoring Harness

**Date:** 2026-05-31
**Goal:** Turn the Phase 1 pipeline (papers flowing end-to-end) into real, logged numbers — the scoring layer the workshop paper is built on.

---

## What was built — added to the `eval/` package

| File | Role |
|------|------|
| `eval/metrics.py` | Pure scoring functions (no I/O, no LLM). SQuAD-style `normalize_text` + `token_f1`; `answer_f1` (max over annotator references, with unanswerable special-casing); answerable/abstention logic (`is_no_answer`, `gold_is_answerable`, `answerable_correct`); evidence `evidence_recall` + `evidence_token_f1`. Unit-validated against known cases. |
| `eval/judge.py` | LLM-as-judge via the unified rotating client. Grades a system answer vs. the human reference(s) as CORRECT / PARTIAL / INCORRECT. Catches semantically-correct prose that token-F1 misses. |
| `eval/run_eval.py` | The runner: ingest papers → run each question through `answer_query` → score → write one JSONL row per question + an aggregate summary (overall and per answer-type). |

Run outputs land in `eval/results/` (gitignored).

```
venv/Scripts/python.exe -m eval.run_eval --papers 3 --qs 5
venv/Scripts/python.exe -m eval.run_eval --papers 5 --qs 0 --judge   # all questions, with judge
```

## Metrics implemented

- **Answer-F1** — SQuAD token-F1 vs. annotator references (max over them). Unanswerable references score 1.0 iff the system also abstained.
- **Answerable accuracy** — did the system correctly decide to answer vs. abstain? (gold = annotator majority vote; system abstention = failed confidence gate or an explicit refusal phrase.)
- **Evidence recall@k / token-F1** — overlap of the dense-retrieved top-k context with the gold evidence paragraphs. *Adaptation:* PaperMind retrieves chunks, not QASPER's original paragraphs, and the API exposes only section-level cited sources — so evidence is scored on text overlap of an independent retrieval probe, clearly labeled, rather than exact paragraph-set membership.
- **LLM-as-judge (optional)** — CORRECT/PARTIAL/INCORRECT on answerable questions.

## First real numbers (smoke validation, 1 paper, 3 extractive questions)

| Metric | Value |
|--------|-------|
| Answer-F1 | 0.147 |
| Evidence recall@k | **0.917** |
| Answerable accuracy | 1.000 |
| Mean confidence | 93.1 |
| Judge (2 qs) | PARTIAL / PARTIAL → 0.50 |

**Reading:** the low Answer-F1 alongside 0.92 evidence recall is the token-F1-vs-prose gap in the data — these extractive questions have terse gold (`BIBREF19 ; BIBREF20`, `pivoting ; pivoting_m`) while PaperMind returns correct descriptive prose. Token-F1 collapses; evidence recall shows retrieval is finding the right paragraphs. This is the empirical justification for pairing Answer-F1 with the LLM-as-judge, and the judge is appropriately discriminating (PARTIAL, not a rubber-stamp, where the system got one of two terms right).

---

## Bug fixed during Phase 2

`eval/judge.py` verdict parsing: `"INCORRECT"` contains `"CORRECT"` as a substring, so a naive in-order check misread INCORRECT as correct. Fixed by testing `INCORRECT` before `CORRECT`.

---

## Status & next

**Phase 2: ✅ complete and validated.** Answer-F1, Answerable accuracy, Evidence recall/F1, and LLM-as-judge all run; JSONL + summary are written per run.

**Open items before a full run:**
- **Throughput** (carried from Phase 1): Gemini rate-limited → Cerebras broken (`NotFoundError`) → effectively Mistral-only at ~20–44 s/question. A few-hundred-question run needs a rate-limit plan first.
- **Evidence metric** currently uses a dense-retrieval probe; could be upgraded to mirror the pipeline's hybrid+rerank retrieval for a closer-to-system number.

**Phase 3 (next):** the ablation matrix — baseline (all flags off) vs. full PaperMind vs. single-ablation runs, driven by the existing env flags `PAPERMIND_DISABLE_GRADER`, `PAPERMIND_DISABLE_RERANK`, `PAPERMIND_DISABLE_HYDE`, `PAPERMIND_MAX_ATTEMPTS=1`. The headline test: does grading lift faithfulness/judged-correctness without dropping Answer-F1, and at what latency cost?

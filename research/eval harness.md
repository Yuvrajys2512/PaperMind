# QASPER Eval Harness — Overview

_Last updated: 2026-06-11._

The harness is QASPER-based — Question Answering on Scientific Papers (Dasigi et al., NAACL 2021).

## Why it exists

PaperMind's core build is done, and the goal now is a workshop paper (targeting ACL/EMNLP 2026). The harness exists to turn "we think the evidence-grading pipeline works" into numbers. QASPER was picked over alternatives because it provides gold  per question, *evidence paragraphs*which lines up exactly with PaperMind's evidence-grading contribution. The dev split is used for development; the test split is reserved for final paper numbers.

## What's in `eval/`

The harness was built in three phases (all committed) plus a newer analysis tool:

- **`qasper_loader.py`** — downloads and caches the official v0.3 dataset to `data/qasper/`, normalizes the four QASPER answer types (extractive / abstractive / yes-no / unanswerable) into one shape, and has `select_papers()` for deterministic paper selection so every ablation config scores the identical question set.
- **`qasper_adapter.py`** — feeds QASPER papers into PaperMind's real pipeline, but skips PDF parsing and LLM section detection since QASPER's `full_text` is already structured sections. It chunks (512/100) and embeds through the existing embedder worker.
- **`metrics.py`** — pure scoring functions: SQuAD-style token F1, Answer-F1 (max over annotator references, with unanswerable handled), answerability accuracy, and evidence recall/F1.
- **`judge.py`** — LLM-as-judge (CORRECT/PARTIAL/INCORRECT) with majority-vote over N=3 calls, added because single-call judge variance was polluting results. The judge exists because token-F1 underrates correct prose answers — gold answers are often terse like "BIBREF19" while the system gives a correct descriptive answer.
- **`run_eval.py`** — the runner: ingest → `answer_query` → score → JSONL per question plus an aggregate summary, output to `eval/results/` (gitignored). Run as `python -m eval.run_eval --papers N --qs K [--judge] [--skip-ingest]`.
- **`run_ablations.py`** — the ablation orchestrator. Six configs: full, baseline (vanilla RAG), no_grader, no_rerank, no_hyde, no_retry, controlled via `PAPERMIND_DISABLE_GRADER/RERANK/HYDE` and `PAPERMIND_MAX_ATTEMPTS=1`. Crucially, those flags are read at module import time, so each config runs in its own subprocess; papers are ingested once up front and each config queries with `--skip-ingest`.
- **`analyze_grader.py`** — paired helped/hurt analysis of the grader (the same question with and without grading), the most recent addition, with `--gen-model` support for weak-vs-strong generator comparisons.
- **`smoke_test.py`** — quick end-to-end sanity check.

## Where the science stands

The headline question is: *does the evidence grader improve faithfulness and/or judged correctness without hurting Answer-F1, and at what latency cost?* Right now the central hypothesis is **unsupported**. At n=14, grading did not improve judged correctness (full 0.615 vs baseline 0.654 vs no_grader 0.692, all within 1 standard error). The one consistent signal is that grading lifts **faithfulness** (0.912 → 0.878 → 0.841 for full → no_grader → baseline) — pointing at a faithfulness-vs-correctness tradeoff rather than a free win. Two confounds from that run were since fixed (prompt-scaffolding leaking into answers, and the generation model drifting mid-run; generation can now be pinned via `PAPERMIND_GEN_MODEL`).

## Next steps (see `research/to_do.md` for the living roadmap)

1. Re-run the paired grader analysis with the majority-vote judge and a pinned generation model: `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6`
2. The weak-vs-strong generator study (llama-3.3-70b vs llama-3.1-8b).
3. Scale up n once a single config is clean, then write up whatever verdict the data gives — including a negative result if that's what it shows.

The standing blocker for any large run is **throughput**: Gemini 2.5-flash-lite is the working #1 provider but rate-limited, Cerebras intermittently returns null content, and the Mistral fallback runs ~20–44s per question. Full write-ups live in `research/phase_1.md` through `phase_3.md`.

# Phase 1 — QASPER Evaluation Harness (Foundation)

**Date:** 2026-05-31
**Goal:** Get real QASPER papers flowing through the full PaperMind pipeline end-to-end, before any scoring is built. This is the foundation every later phase depends on.

---

## Context

PaperMind's build is essentially complete (the 7-session `PLAN.md`, plus live research discovery and the off-topic guard). Both `research.md` and the future-work notes in `PLAN.md` converge on the same conclusion: the missing 30% of the project is an **end-to-end evaluation** that turns *"we think this works"* into a number.

**Benchmark chosen: QASPER** (Dasigi et al., NAACL 2021) — information-seeking questions over NLP papers, with gold **evidence paragraphs** per question. It was picked over FinanceBench / a hand-built set because its evidence annotations line up exactly with PaperMind's central contribution (sentence-level evidence grading).

- **Dev split** → development.
- **Test split** → reserved for final numbers.
- **Planned metrics:** Answer-F1 + Evidence-F1 + LLM-as-judge.
- **End goal:** a workshop paper (ACL / EMNLP / NAACL 2026).

---

## What was built — the `eval/` package

| File | Role |
|------|------|
| `eval/qasper_loader.py` | Downloads + caches the official QASPER v0.3 train/dev tarball to `data/qasper/`. `load_papers(split)`; `normalize_answer()` collapses extractive / abstractive / yes-no / unanswerable answers into one flat record `{answerable, type, text, evidence}`; `iter_questions()` iterates a paper's questions with normalized gold answers. |
| `eval/qasper_adapter.py` | Converts a QASPER paper's structured `full_text` → section list → `chunk_sections(512/100)` → embeds via the existing `embedder_worker.py` subprocess. **Skips PDF parsing and LLM section detection entirely**, because QASPER is already sectioned. The QASPER `paper_id` is used as `paper_name` for both ingest and `answer_query`. |
| `eval/smoke_test.py` | `python -m eval.smoke_test --papers N --qs K` — runs the real pipeline end-to-end and prints gold vs. predicted side by side. |

### Why the adapter is clean

The normal ingestion path is heavily PDF-coupled: PDF text extraction → LLM section detection → chunk → table extraction → embed. QASPER hands us clean sections (`section_name` + `paragraphs`) directly, so the adapter builds the section list the chunker expects and reuses the project's standard chunker (512/100) and embedder unchanged. No PDF, no LLM section-detection cost.

---

## What works (verified)

Verified live on paper `1912.01214` ("Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine Translation"):

```
download → adapt → ingest (18 sections → 23 chunks) → retrieve → answer
```

| Question | Gold | PaperMind | Conf |
|----------|------|-----------|------|
| which multilingual approaches do they compare with? | BIBREF19 ; BIBREF20 | pivoting, MNMT, cross-lingual transfer… | 91.8 |
| what are the pivot-based baselines? | pivoting ; pivoting$_{\rm m}$ | "pivoting and pivot-synthetic" | 96.1 |

Both questions answered with high-confidence, paper-grounded responses.

---

## Bug found and fixed during Phase 1

**`ingestion/bm25_retriever.py`** looked up the ChromaDB collection with the **raw** `paper_name`, while `embedder.py` and `retriever.py` clean it first (non-alphanumerics → hyphen, lowercased — e.g. `1912.01214` → `1912-01214`).

- UUID-style paper IDs (the main app) clean to themselves, so the bug stayed hidden.
- QASPER's dotted arXiv IDs exposed it: `Collection [1912.01214] does not exist`.

**Fix:** clean the name for the lookup, mirroring `retriever.py` exactly; the in-memory cache key stays the raw name (consistent on read/write/invalidate). No effect on the main app (no-op for already-clean names); unblocks dotted IDs.

> Note: the name-cleaning logic is now triplicated inline across `embedder.py`, `retriever.py`, and `bm25_retriever.py`. A single shared helper would be a worthwhile cleanup later.

---

## Observations to carry into Phase B

1. **Throughput is the real constraint.** Every LLM call currently tries Gemini (free-tier, rate-limited) → Cerebras (returns `NotFoundError` on *every* call — the model name is still wrong) → lands on Mistral. The system is effectively Mistral-only at **20–44 s/question**. A few-hundred-question QASPER run needs a rate-limit / throughput plan first (and probably a dedicated eval API key).

2. **LLM-as-judge is already justified.** Q1's gold answer was a citation (`BIBREF19`) but the system gave a *correct descriptive* answer — token-F1 alone would score that near zero. This confirms Answer-F1 must be paired with an LLM-as-judge.

3. **Keep eval data out of the app DB.** The smoke test writes QASPER collections into the same `data/chroma_db` the app uses. Harmless now (isolated collections), but the full harness should use a separate Chroma path.

---

## Status & next

**Phase 1: ✅ complete.** The QASPER → PaperMind path runs end-to-end on real data.

**Phase B (next):** the scoring harness —
- official QASPER **Answer-F1** + **Evidence-F1**,
- **LLM-as-judge** for abstractive answers,
- per-question **JSONL logging**.

**Phase C (after):** the ablation matrix, driven entirely by existing env flags —
`PAPERMIND_DISABLE_GRADER`, `PAPERMIND_DISABLE_RERANK`, `PAPERMIND_DISABLE_HYDE`, `PAPERMIND_MAX_ATTEMPTS=1` (the last four together = the vanilla-RAG baseline).

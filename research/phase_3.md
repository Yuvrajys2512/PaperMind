# Phase 3 — Ablation Matrix

**Date:** 2026-05-31 (findings corrected 2026-06-01)
**Goal:** Run the QASPER eval under multiple pipeline configurations to test whether evidence grading improves answer quality.

---

## ⚠️ Headline correction — read this first

An initial **n=9** run suggested evidence grading gave a large judge-accuracy boost (+0.187 over `no_grader`, +0.25 over `baseline`). **This did not replicate.** A larger, cleaner **n=14** run showed the effect vanish and slightly reverse — well within noise. The n=9 result was driven by small-sample variance and an unstable provider mix (different questions answered by different models). **Do not build the paper on the n=9 numbers.**

---

## Setup

Each config runs in its own subprocess (ablation flags are read at import time). Papers ingested once; configs query the same store. Judge = LLM-as-judge (CORRECT/PARTIAL/INCORRECT) on answerable questions.

| Config | Flags |
|--------|-------|
| `full` | none — all features on |
| `baseline` | grader + rerank + HyDE + retry OFF (vanilla RAG) |
| `no_grader` | `PAPERMIND_DISABLE_GRADER=1` |

---

## Results

### Run A — preliminary (n=9, 3 papers, mixed providers) — SUPERSEDED

| Config | Judge acc | Faithfulness |
|--------|-----------|--------------|
| full | 0.750 | 0.889 |
| no_grader | 0.563 | 0.889 |
| baseline | 0.500 | 0.867 |

Caveat: the provider chain was unstable during this run (Gemini rate-limited, Cerebras broken), so different questions were effectively answered by different models. n=9. Treat as void.

### Run B — scaled & cleaner (n=14, 5 papers, Groq-first)

| Config | Judge acc | Faithfulness | Answerable | Latency | LLM calls |
|--------|-----------|--------------|------------|---------|-----------|
| full | **0.615** | **0.912** | 0.929 | 8.7s | 4.6 |
| baseline | 0.654 | 0.841 | 0.929 | 5.2s | 2.9 |
| no_grader | **0.692** | 0.878 | 0.929 | 7.4s | 3.8 |

---

## Honest findings

1. **No measurable judge-accuracy benefit from grading.** `full` (0.615) vs `no_grader` (0.692) is a ~0.08 gap — smaller than one standard error (~0.13 at n=14, ≈ one question). The earlier +0.187 was noise.

2. **Grading does improve faithfulness** (`full` 0.912 > `no_grader` 0.878 > `baseline` 0.841) — exactly what it's designed to do (remove unsupported sentences). But this does **not** convert into higher judged correctness; in at least one case the grader removed 3 sentences from an answer that was still correct. The likely real story is a **faithfulness-vs-correctness tradeoff**, not a free win.

3. **Confounds in Run B (now fixed):**
   - Two variables changed at once — scaled n *and* switched the generation model to Groq `llama-3.3-70b`. Bad hygiene; the reversal can't be cleanly attributed.
   - That model **leaked prompt scaffolding** (`## FINAL ANSWER FORMAT`, "Now writing the final answer.") into answers, degrading all three configs. Fixed in `generator.py` via `_strip_scaffolding()` (anchors the answer at the `ESSENCE` marker).

4. **Token-F1 stays uninformative** (0.087–0.114): terse gold spans vs. descriptive prose. The judge remains the right metric — this part still holds.

---

## Where this leaves the research

The central hypothesis — *"evidence grading improves faithfulness without sacrificing correctness"* — is **not yet supported by data**. Current evidence suggests grading helps faithfulness but may cost correctness. That is still a legitimate and publishable finding **if confirmed at scale** — but it's a more nuanced story than "grading is a clear win," and it would change the paper's framing.

---

## Next steps (toward a clean, trustworthy run)

1. ✅ **Fixed** the scaffolding leak (`generator._strip_scaffolding`).
2. **Standardize on one generation model** (Groq `llama-3.3-70b`) across all configs so runs are comparable. Pin it; don't let provider fallback silently swap models mid-run.
3. **Re-run at larger n** (more papers/questions) now that throughput is fixed (~5–9s/question).
4. **Report whatever it shows.** If grading trades correctness for faithfulness, measure that tradeoff explicitly (faithfulness gain vs. correctness loss, per answer type).
5. **Investigate the grader threshold** — it may be too aggressive, removing correct sentences. Tuning it could turn the tradeoff into a net win (or confirm it can't).

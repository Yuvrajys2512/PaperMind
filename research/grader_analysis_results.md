# Evidence Grader — Helped/Hurt Analysis Results

_Tracking all `eval.analyze_grader` runs. Each run uses the paired design:
both pre-grader (`original_answer`) and post-grader (`answer`) come from the
same generation call, so the delta is grader-only with no model confound._

---

## Run 0 — Contaminated baseline (VOID)

**Date:** 2026-06-01
**File:** `eval/results/grader_analysis_dev_20260601_015450.jsonl`
**Command:** `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6`
**Status:** VOID — do not use

**Why voided:** Judge was single-call (no majority vote), so the only "helped"
and "hurt" rows had `removed_count: 0` — pure judge non-determinism, not the
grader. Fixed by N=3 majority-vote judging (commit `6235832`).

---

## Run 1 — Clean baseline ✓

**Date:** 2026-06-02
**File:** `eval/results/grader_analysis_dev_20260602_005711.jsonl`
**Command:** `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6`
**Generation:** default provider chain (Groq-1 → Gemini → Mistral fallback)
**Judge:** N=3 majority vote
**Status:** VALID — first trustworthy run

| Metric | Value |
|--------|-------|
| Questions (answerable) | 14 |
| Questions with ≥1 removal | 4 |
| Mean judge score — original | 0.607 |
| Mean judge score — cleaned | **0.607** |
| HELPED | 3 |
| HURT | 3 |
| NEUTRAL | 8 |

**Finding:** Net grader effect on judged correctness is **zero** (helped = hurt = 3).
Faithfulness was not measured in this run (paired design, not ablation).

**HURT cases:**

1. Q: "did they use a crowdsourcing platform for manual annotations?"
   - `CORRECT → PARTIAL`
   - Removed: *"No, the paper does not mention using a crowdsourcing platform..."*
   - **Root cause:** Grader removed a correct negative answer — factual negations
     have no direct textual support so they always score low on the grader's
     faithfulness check. Classic false-positive removal.

2. Q: "What accuracy does the proposed system achieve?"
   - `INCORRECT → INCORRECT` (score dropped within tier — different numeric score)

3. Q: "On how many language pairs do they show that preordering..."
   - `PARTIAL → INCORRECT`

**Interpretation:** The grader's HURT pattern is driven by removing correct
negative answers and over-trimming partially correct answers. Not random noise.

---

## Run 2 — Strong generator (INVALID)

**Date:** 2026-06-02
**File:** `eval/results/grader_analysis_dev_llama-3.3-70b-versatile_20260602_011148.jsonl`
**Command:** `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6 --gen-model llama-3.3-70b-versatile`
**Status:** INVALID — do not use

**Why invalidated:** Both Groq keys share the same organisation's 100k TPD
quota for `llama-3.3-70b-versatile`. Run 1 exhausted most of that budget.
The pin to Groq-2 hit TPD immediately; the pipeline swallowed the errors
and returned graceful-degradation answers ("Unable to answer...") for all
14 questions. Scores (mean 0.154, all INCORRECT) reflect the fallback
string, not actual answers.

**Fix applied:** `analyze_grader.py` now detects graceful degradation
(`confidence == 0 AND warning set`) and skips those questions instead of
scoring them.

---

## Run 3 — Weak generator (PARTIAL — scaffolding leak)

**Date:** 2026-06-02
**File:** `eval/results/grader_analysis_dev_llama-3.1-8b-instant_20260602_*.jsonl`
**Command:** `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6 --gen-model llama-3.1-8b-instant`
**Generation:** Groq-2 pinned to `llama-3.1-8b-instant`
**Judge:** N=3 majority vote
**Status:** PARTIAL — 7/~14 questions answered; scaffolding leak in weak model

| Metric | Value |
|--------|-------|
| Questions (answerable, scored) | 7 |
| Questions with ≥1 removal | 2 |
| Mean judge score — original | 0.571 |
| Mean judge score — cleaned | **0.381** |
| HELPED | 0 |
| HURT | 2 |
| NEUTRAL | 5 |

**Finding:** The grader strongly hurt the weak model. But the data is partially invalid.

**Issues:**
1. **Scaffolding leak** — `llama-3.1-8b-instant` emits the system prompt template
   text verbatim ("2-3 sentences capturing the most important insight", "Sharp, direct,
   standalone", "Follow the answer_structure steps…") into the answer body.
   `_strip_scaffolding()` in `generator.py` was designed for the strong model's output
   format and doesn't catch the weak model's different leakage pattern.
   The grader correctly removes these non-answer sentences as UNSUPPORTED, but the
   answer is already broken at the generator level — it's the generator's bug, not the
   grader's.
2. **Only 7/~14 questions scored** — remaining questions hit Groq TPD on the pinned
   provider and returned graceful degradation (now correctly skipped by the new SKIP
   handler).

**Action required:** Fix `_strip_scaffolding()` to handle the weak model's output, then re-run.

---

## Run 4 — Strong generator retry (PENDING)

**Date:** TBD — blocked on Groq daily quota reset (~24h from 2026-06-02 00:00 IST)
**Command:** `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6 --gen-model llama-3.3-70b-versatile`
**Status:** PENDING

---

## Summary of fixes applied this session (2026-06-02)

| Fix | File | Reason |
|-----|------|--------|
| Mark generator.py commit done | `research/to_do.md` | Stale `[ ]` item; was committed in `6a9fd1a` |
| Log real error type on skip | `ingestion/llm_client.py:192` | Always printed "rate-limited (RateLimitError)" regardless of actual error |
| Pin generation to Groq-2 | `ingestion/generator.py:289` | Groq-1 shared TPM/TPD with all other calls; Groq-2 has separate quota |
| Skip graceful degradation | `eval/analyze_grader.py:91` | Pipeline swallows generation failures silently; returned fallback answers were being scored as real data |

---

## Open research question

> Does the evidence grader improve **faithfulness** and/or **judged correctness**
> without dropping Answer-F1, and at what latency cost?

Current evidence (Run 1, n=14):
- Net correctness effect: **zero** (helped = hurt = 3)
- Faithfulness improvement: not re-measured post majority-vote fix (Phase 3 Run B
  showed `full` 0.912 > `no_grader` 0.878, but that run had the scaffolding leak)
- The HURT pattern suggests a specific failure mode (correct negations) rather than
  random noise — tuning the grader threshold may resolve it

**Next:** Run weak generator (Run 3) to compare grader effect across model strengths.
Then re-run the ablation (run_ablations.py) at scale for final faithfulness numbers.

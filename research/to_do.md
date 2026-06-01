# PaperMind — QASPER Eval TODO

_Last updated: 2026-06-01. Goal: turn the QASPER ablation study into a workshop paper._

## Where we are

- **Phase 1–3 of the eval harness are built and committed** (`eval/`): loader/adapter,
  scoring (`metrics.py` + `judge.py` + `run_eval.py`), and the ablation matrix
  orchestrator (`run_ablations.py`). Write-ups: `research/phase_1.md`, `phase_2.md`,
  `phase_3.md`.
- **Central hypothesis is currently UNSUPPORTED by data.** At n=14 (5 papers, Groq-first)
  the evidence grader did **not** improve judged correctness (full 0.615 / baseline 0.654 /
  no_grader 0.692 — gap < 1 SE). The only consistent signal: grading lifts **faithfulness**
  (full 0.912 > no_grader 0.878 > baseline 0.841), not correctness. Likely a
  faithfulness-vs-correctness tradeoff, not a free win.
- **Two confounds from that run are now fixed:** (a) prompt-scaffolding leak →
  `generator._strip_scaffolding()`; (b) generation model drift → can now be pinned.
- **Judge noise was the dominant artifact** in the last paired run
  (`grader_analysis_dev_20260601_015450.jsonl`): the only "helped"/"hurt" rows had
  `removed_count: 0`, i.e. cleaned == original, so the flips were pure judge variance.
  Fixed by majority-vote (N=3) judging in `judge.py` (committed `6235832`).

## Pending (uncommitted)

- [x] **Commit `ingestion/generator.py`** — the `PAPERMIND_GEN_MODEL` wiring that pins
      generation only (`pin=("Groq-1", model)`). Done in commit `6a9fd1a`.

## Immediate next steps

- [ ] **Re-run the paired grader analysis with the fixes in place** — majority-vote judge +
      a pinned generation model — to get a clean helped/hurt signal:
      `venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6`
- [ ] **Weak-vs-strong generator study** — does generator strength change the grader effect?
      - `... analyze_grader --papers 5 --qs 6 --gen-model llama-3.3-70b-versatile` (strong)
      - `... analyze_grader --papers 5 --qs 6 --gen-model llama-3.1-8b-instant` (weak)
- [ ] **Scale up the run** once a single config is clean: larger n (more papers/qs) so the
      faithfulness-vs-correctness effect clears 1 SE in one direction or the other.

## Core research question to settle

> Does the evidence grader improve **faithfulness** and/or **judged correctness** without
> dropping **Answer-F1**, and at what **latency cost**?  ← this is the paper headline.

Decide the verdict honestly from the data:
- If grading lifts faithfulness but not correctness → frame as an explicit tradeoff.
- If neither holds at scale → report the negative result; it's still a valid contribution.

## Known issues / risks

- [ ] **Throughput is the bottleneck for any large run.** Mistral fallback is ~20–44s/q.
      - Cerebras was updated to `gpt-oss-120b` but returns null content intermittently
        (treated as a skip in `llm_client.py`, falls through to Mistral).
      - Gemini `2.5-flash-lite` is the working #1 provider (`2.0-flash` had free-quota 0).
- [ ] **Verify provider names in `llm_client.py` (lines ~20–26).** Saw what looked like
      duplicate/mislabeled `"Groq-1"` entries while tracing the `pin` path — confirm the
      provider chain is what we think before trusting pinned-model runs. (Pin currently
      matches the first `Groq-1`, which is genuine Groq, so weak/strong runs are valid.)
- [ ] **Clean up the misleading log** in `llm_client.py`: every skipped provider is labeled
      "rate-limited (RateLimitError)", which hid the real `limit:0` cause. Print the actual
      error on skip.

## Paper write-up tasks (after numbers are final)

- [ ] Consolidate `phase_1/2/3.md` into a single methods + results narrative.
- [ ] Results table: per-config Answer-F1, faithfulness, judged accuracy, mean LLM calls,
      mean latency.
- [ ] Ablation figure (full vs baseline vs no_grader / no_rerank / no_hyde / no_retry).
- [ ] Grader helped/hurt case study (the paired analysis — qualitative examples of
      sentences the grader removed).
- [ ] Limitations: small n, free-tier provider variance, judge model = grader model family.

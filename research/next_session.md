# NEXT SESSION — start here

_Handoff written 2026-06-01. Read this, then `research/to_do.md` for the full roadmap._

## TL;DR — what to do first

The tree is **clean and committed**. Start the **clean grader re-run**:

```
venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6
```

This is the first run with BOTH recent fixes in place (majority-vote judge +
pinnable generation), so it's the first trustworthy helped/hurt signal. Everything
before this was contaminated by judge noise.

## What just happened (this session)

- Recovered context after the app closed — no prior chat in memory; reconstructed from
  memory files + git.
- Committed `6a9fd1a` "pin generation model via PAPERMIND_GEN_MODEL" — bundled the
  `ingestion/generator.py` wiring AND `research/to_do.md` (plus settings + chroma binaries).
- Working tree is clean apart from chroma_db binary drift (expected) and this handoff file.

## The experiment you're running

`eval/analyze_grader.py` — **paired** helped/hurt analysis of the evidence grader.
For each answerable question the pipeline returns the pre-grader answer
(`original_answer`) AND the post-grader answer (`answer`) from the *same* generation;
both are judged vs gold, and the delta is attributed purely to the grader (no model
confound). HURT cases print the sentences the grader removed.

## Why the last run didn't count

`grader_analysis_dev_20260601_015450.jsonl` (14 Qs): the grader only removed sentences
in 2/14 cases, and the lone "helped"/"hurt" rows had `removed_count: 0` (cleaned ==
original) — i.e. pure judge non-determinism, not the grader. That noise is exactly what
the N=3 majority-vote judge (commit `6235832`) was built to kill. So re-run now.

## Then: weak-vs-strong generator study

Does generator strength change the grader's effect?

```
venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6 --gen-model llama-3.3-70b-versatile   # strong
venv/Scripts/python.exe -m eval.analyze_grader --papers 5 --qs 6 --gen-model llama-3.1-8b-instant       # weak
```

Output files are tagged by model. `--gen-model` sets `PAPERMIND_GEN_MODEL`, which the
generator (now committed) reads to pin generation ONLY — grading and judging stay on the
normal provider chain.

## Open question to settle (paper headline)

Does the grader improve **faithfulness** and/or **judged correctness** without dropping
**Answer-F1**, and at what **latency cost**? Current data (n=14): grader lifts
faithfulness (0.912 vs 0.841 baseline) but NOT judged correctness (gap < 1 SE) — a likely
tradeoff, not a free win. Re-run at clean signal + larger n to confirm or overturn.

## Watch out for

- **Throughput** is the bottleneck. Mistral fallback ~20–44s/q. Cerebras (`gpt-oss-120b`)
  returns null content intermittently → skips to Mistral. Gemini `2.5-flash-lite` is the
  working #1. A 5-paper × 6-q run with N=3 judging is a fair number of LLM calls — expect
  minutes, not seconds. Consider running it in the background.
- **`llm_client.py` provider names (~lines 20–26):** there appeared to be duplicate-looking
  `"Groq-1"` entries. The pin matches the first one (genuine Groq), so weak/strong runs are
  valid — but verify the chain before trusting it.

## Full roadmap

`research/to_do.md` — pending commits (none now), next steps, known risks, and the paper
write-up task list.

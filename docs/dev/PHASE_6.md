# Phase 6 — End-to-End Evaluation Set

> Fills the "Gap: End-to-end eval set" from PLAN.md.
> Goal: turn PaperMind from "interesting demo" into "measured engineering result" with one defensible headline number.

---

## 1. What we're really trying achieve

Right now, every claim about PaperMind is theoretical. We *say* hybrid retrieval beats BM25-only. We *say* the evidence grader fixes hallucination. We *say* the retry engine recovers failed answers. **None of that is measured.** We have anecdotes ("attention paper question worked"), not numbers.

The goal of this phase is to produce **one defensible headline number** like:

> "PaperMind scores 0.78 ± 0.03 on PaperMind-Eval-v1 (300 questions), vs 0.51 for vanilla RAG and 0.34 for the raw LLM."

That number is what turns the project from "look at this cool pipeline" into "look at this pipeline that demonstrably outperforms the baseline by 27 points." Without it, anything we built is unverifiable.

---

## 2. What an eval set actually is (the rigorous definition)

A Q&A pair is **not** just a question and a string answer. To be useful for evaluation, each pair must contain:

| Field | Why it exists |
|---|---|
| `question` | What the user asks |
| `ground_truth` | The correct answer (verbatim string for factoid, rubric for open-ended) |
| `source` | `{section, page, exact_quote}` — where in the paper the answer lives |
| `paper_id` | Which paper this is asked against |
| `type` | factual / mechanism / comparison / multi_hop / summarization / critique / out_of_domain |
| `difficulty` | easy / medium / hard |
| `out_of_domain` | bool — some questions must be unanswerable (tests refusal calibration) |

**Why all this metadata?** Because aggregate accuracy is a useless number. "70% accurate" tells you nothing. "70% on factual, 45% on multi-hop, 92% on refusing out-of-domain" tells you exactly where the system breaks. Breakdowns are how you find the next bug to fix.

---

## 3. Dataset construction — the 80% of the work

### 3.1 How many questions, and why

**Target: 300 questions across 10-12 papers.** Reasoning:

- **N=300 gives statistical power.** Binomial 95% CI is roughly ±3-4 percentage points at that N. A 5-point lift over baseline is statistically significant. At N=50, only differences ≥14 points are detectable — anything subtler is noise.
- **10-12 papers** covers variety without losing depth — too few and you over-fit to one paper's quirks; too many and per-paper question count drops too low to compare papers fairly.

### 3.2 The mix (this matters a lot)

Per paper, ~25-30 questions broken down as:

| Type | Count per paper | Why include it |
|---|---|---|
| Factual | 8-10 | The easy baseline. Numbers, names, dates. Tests basic retrieval. |
| Mechanism | 4-5 | "How does X work?" Tests CoT and detailed retrieval. |
| Multi-hop | 3-4 | Requires connecting evidence from 2+ sections. Tests the multi-hop path. |
| Comparison | 2-3 | Tests comparison-type answer structure (in-paper, not cross-paper). |
| Critique | 2-3 | Tests whether the system correctly identifies stated limitations. |
| Summarization | 2 | Broad coverage. Tests retrieval breadth. |
| **Out-of-domain** | 3-4 | **Mandatory.** Tests refusal — "Did the author discuss climate change?" on an ML paper. Ground truth = "not in paper." |

The out-of-domain bucket is non-negotiable. A system that confidently answers everything is broken even if its in-domain accuracy is 100%, because it would hallucinate on adversarial input. You need to *measure* refusal.

### 3.3 How to actually write 300 questions

**Hybrid: ~50 human-written, ~250 LLM-generated then human-verified.** Not 300 hand-written (too slow), not 300 LLM-only (too template-y, low quality).

**Step-by-step:**

1. **Pick 10-12 papers.** Mix of: short methods papers (~10 pages), long survey papers (~30 pages), math-heavy (Transformer), empirical (BERT), recent papers, classic papers. This protects against domain-specific over-fitting.

2. **Human-write 5 anchor questions per paper.** These are gold-standard examples — perfectly worded, clearly cited, varied types. They serve two purposes: (a) high-quality data points, (b) few-shot examples for the LLM generator.

3. **LLM generation, paper-by-paper.** For each paper, feed Claude Opus (or GPT-4) the cleaned full text + 5 anchor questions, prompt: "Generate 25 more Q&A pairs in this format, covering these types, with exact source quotes." Save to draft file.

4. **Human verification pass — the slow part.** Every single generated question gets reviewed:
   - Is it unambiguous?
   - Is the ground-truth answer actually in the paper (verbatim or paraphrased)?
   - Is the cited section/page correct?
   - Is the difficulty tag honest?
   - Reject anything where you'd be unsure of the "correct" answer yourself.

   **Expected rejection rate: 30-40%.** That's normal. LLM-generated questions tend to be either too obvious ("What is the title?") or too vague ("What is the paper about?"). The rejection is the quality gate.

5. **Diversity check.** Sort by question type and difficulty. If one bucket is empty (e.g., zero comparison questions for a given paper), write them by hand to fill the gap.

**Time estimate: 15-20 hours of focused human work.** This is the bulk of the phase, and there is genuinely no shortcut. If you generate 300 questions with an LLM and don't verify them, you're testing the LLM's question-generation prior, not your pipeline.

### 3.4 Schema (the file format)

```json
{
  "id": "q001",
  "paper_id": "2d4db181-...",
  "paper_title": "Attention Is All You Need",
  "question": "What hardware was used to train the base Transformer model?",
  "ground_truth": "8 NVIDIA P100 GPUs",
  "ground_truth_aliases": ["eight P100 GPUs", "8 P100s"],
  "source": {
    "section": "5.1 Training Data and Batching",
    "page": 7,
    "quote": "We trained on one machine with 8 NVIDIA P100 GPUs."
  },
  "type": "factual",
  "difficulty": "easy",
  "out_of_domain": false
}
```

`ground_truth_aliases` matters — "8" vs "eight" should both count. Exact-match scoring is brittle without alias support.

---

## 4. The harness — `eval/harness.py`

A single script that runs any pipeline against the dataset and produces raw results.

```python
harness.run(
    dataset = "eval/dataset.json",
    pipeline = papermind.answer_query,   # or any other callable
    output = "eval/runs/papermind_2026-05-12.json",
)
```

For each question:
1. Call `pipeline(question, paper_id)`.
2. Capture: answer text, confidence, sources, duration, attempts, evidence grades.
3. Store raw — **do not score yet.**

Why separate "run" from "score": scoring is the expensive/iterative part. You'll change scorers over time (better LLM judge prompt, new metric). Running the pipeline once and re-scoring repeatedly is much cheaper than re-running every time.

---

## 5. The scorers — multiple, because no single metric is sufficient

### 5.1 Exact match
Normalize whitespace + casing, strip punctuation, check against `ground_truth` and `ground_truth_aliases`. Robust for factual questions ("8 GPUs", "P100", "BLEU 28.4").

### 5.2 Token F1
Token overlap between answer and ground truth. Catches near-matches that exact match misses. Standard SQuAD metric.

### 5.3 LLM-as-judge
Feed a strong LLM (Claude Opus): question + ground truth + pipeline answer. Prompt:
> Rate 0-5 how completely the pipeline's answer covers the ground truth. 0 = wrong/missing. 5 = fully covered. Be strict.

Why this is needed: open-ended questions (mechanism, critique) don't have one right string. A pipeline answer can be 100% correct and have 5% token overlap with ground truth because both express the same idea in different words. Only an LLM judge handles this.

**Why this is risky and how to mitigate:** LLM judges have biases — they prefer fluent answers, they're soft graders, they're inconsistent. Mitigations:
- Run each rating twice and average.
- Use Claude Opus or GPT-4 only — cheaper models are too noisy.
- Calibrate on the 50 hand-written anchor questions: if the judge gives a known-correct answer a 5, we trust it.

### 5.4 Citation correctness
Did the pipeline's `[Section: X, Page: N]` citation match the ground-truth `source.section` and `source.page`? Binary: correct or not. Tests retrieval grounding separately from answer text.

### 5.5 Refusal check (for out-of-domain questions)
For questions tagged `out_of_domain: true`, did the pipeline:
- Return a warning?
- Trigger the out-of-domain shortcut (relevancy < 0.05)?
- Decline to answer?

Score: refused vs hallucinated. **A system that confidently answers OOD questions fails this even with 100% in-domain accuracy.**

---

## 6. The baselines — what we measure ourselves against

A pipeline score with no baseline is meaningless. We run the same dataset through three other systems:

### 6.1 Vanilla RAG (the primary baseline)
Bare-bones implementation, no PaperMind features:
- Top-5 dense vector retrieval (no BM25, no RRF).
- No reranking.
- Generic prompt: "Answer based on this context."
- No CoT, no plan, no grader, no retry.

This is "what someone would build in an afternoon with LangChain defaults." If PaperMind doesn't beat this, the project's premise is wrong.

### 6.2 Raw LLM (the floor)
Just `chat_completion("Paper: <title>. Question: <Q>")`. No retrieval at all. Tests how much the LLM already knows from training data. Usually scores ~20-40% — useful as a floor.

### 6.3 BM25-only and vector-only (optional, isolates retrieval contribution)
Tests whether hybrid retrieval was worth building vs picking one.

---

## 7. Ablations on PaperMind — which features earn their complexity

Run the dataset through *modified* PaperMind versions, each with one feature disabled:

| Variant | What's disabled |
|---|---|
| No-grader | Skip evidence_grader entirely |
| No-retry | `MAX_ATTEMPTS = 1` |
| No-CoT | Generator uses simple prompt, no scratchpad |
| No-rerank | Skip CrossEncoder reranking |
| No-planner | Skip query_planner, use fixed retrieval config |
| No-multi-hop | Always single-pass retrieval |

**Why this matters:** If "PaperMind full" scores 0.78 and "no-grader" scores 0.77, the grader added 1 point — it might not be worth its complexity and LLM cost. If "no-grader" scores 0.65, the grader is load-bearing.

**This is how you justify every architectural decision in interviews.** "I built the evidence grader because the ablation shows it lifts accuracy 13 points on the multi-hop subset" is a defensible engineering statement. "I built it because hallucinations are bad" is hand-waving.

---

## 8. The final report — what success looks like

`eval/reports/report_2026-05-XX.md`:

```
## PaperMind Eval v1 — 2026-05-12

Dataset: 300 questions, 11 papers
Pipelines compared: 5 (PaperMind + 4 baselines/ablations)

| Pipeline          | Overall | Factual | Mechanism | Multi-hop | OOD refusal |
|-------------------|---------|---------|-----------|-----------|-------------|
| PaperMind (full)  | 0.78    | 0.91    | 0.74      | 0.62      | 0.94        |
| – grader          | 0.71    | 0.90    | 0.65      | 0.55      | 0.91        |
| – retry           | 0.74    | 0.91    | 0.71      | 0.58      | 0.93        |
| Vanilla RAG       | 0.51    | 0.78    | 0.41      | 0.22      | 0.35        |
| Raw LLM           | 0.34    | 0.52    | 0.30      | 0.12      | 0.10        |

Cost per question (median):
  PaperMind: $0.008, 7.2s, 3 LLM calls
  Vanilla:   $0.001, 1.4s, 1 LLM call
```

That single table is the deliverable. Everything else is supporting analysis.

---

## 9. Why this approach actually works (the science defense)

- **N=300 → statistical significance.** Differences ≥4 points are real, not noise. We can say "PaperMind lifts accuracy by X points (p < 0.05)" with confidence.
- **Multi-scorer → no single metric blindspot.** Exact-match catches factoids; LLM judge catches paraphrase; citation check catches grounding; refusal check catches calibration. A system has to score well across all four to be genuinely good.
- **Baselines + ablations → causal claims.** We aren't just measuring PaperMind; we're measuring *which features cause its advantage*. That's the difference between "PaperMind works" and "PaperMind works because of X, Y, Z."
- **Out-of-domain bucket → tests safety.** Most RAG benchmarks ignore refusal. Including it surfaces hallucination-on-adversarial-input, which is the real production failure mode.
- **Human verification → trust.** LLM-generated questions verified by hand are the only way to ensure ground truth is actually true. Skipping this step means measuring the wrong thing.

---

## 10. What "no shortcuts" means concretely

These are the corners that will tempt you and that you must not cut:

1. **Don't skip human verification.** Every LLM-generated question gets reviewed. Yes, this is slow. No, there is no faster way to get trustworthy ground truth.
2. **Don't run ablations on a subset.** If you ablate on 50 questions instead of 300, your conclusions are noise. Full dataset every time.
3. **Don't omit out-of-domain.** "My system scores 90% on questions it can answer" is meaningless without "and refuses 95% of questions it can't."
4. **Don't write only questions PaperMind can answer.** That's evaluating against a curve. Include questions you suspect will fail — those are the most informative.
5. **Don't use Mistral-Small or Groq Llama as the LLM judge.** Use Claude Opus or GPT-4. Cheap judges produce noisy ratings; the noise will swamp your real signal.
6. **Don't compare PaperMind to "no RAG at all" and call it a win.** Vanilla RAG is the honest baseline — anything beating raw-LLM is trivial.
7. **Don't tune the pipeline on the eval set.** Once the eval set is built, freeze it. If you change thresholds based on eval results, you've contaminated your test set. Keep a separate dev set (~30 questions) for tuning.
8. **Report failures.** If the retry engine lifts accuracy by 1 point and costs 3x latency, say so. Honest negative results build credibility.

---

## 11. Phase deliverables — done = all six exist

1. `eval/dataset.json` — 300 verified Q&A pairs with full metadata, frozen.
2. `eval/harness.py` — runs any pipeline against the dataset.
3. `eval/scorers/` — exact_match, f1, llm_judge, citation_check, refusal_check.
4. `eval/baselines/vanilla_rag.py` — minimum viable RAG implementation.
5. `eval/ablations/` — five PaperMind variants with one feature disabled each.
6. `eval/reports/report_v1.md` — the table from §8, plus per-type breakdowns, per-paper breakdowns, cost analysis, and a written conclusion identifying which features earned their complexity.

When all six exist and the report shows a defensible lift over baseline, the phase is done — and the project has its headline number.

---

## 12. Time budget

| Stage | Hours |
|---|---|
| Picking papers + writing 50 anchor questions | 3-4 |
| LLM generation of 300 candidate questions | 1-2 |
| Human verification (the slow part) | 12-15 |
| Harness + scorers | 4-6 |
| Baselines + ablations | 3-4 |
| Running all 5 pipelines × 300 questions | 2-3 (mostly wall-clock waiting) |
| Writing the report | 2-3 |
| **Total** | **~30 hours** |

This is the largest single phase in the project. It's also the one that turns "interesting demo" into "measured engineering result." That's what the gap is asking for, and that's what no shortcuts looks like.

---

## 13. Suggested directory layout (for when implementation starts)

```
eval/
├── dataset.json                # the 300 Q&A pairs (frozen after construction)
├── dev_set.json                # 30 separate questions for tuning — never use this for final scoring
├── papers/                     # the 10-12 source PDFs (or pointers to data/papers/)
├── harness.py                  # run any pipeline against the dataset
├── scorers/
│   ├── __init__.py
│   ├── exact_match.py
│   ├── f1.py
│   ├── llm_judge.py
│   ├── citation_check.py
│   └── refusal_check.py
├── baselines/
│   ├── vanilla_rag.py          # top-5 vector retrieval + simple prompt
│   ├── raw_llm.py              # no retrieval at all
│   └── bm25_only.py            # optional retrieval-isolation baseline
├── ablations/
│   ├── no_grader.py
│   ├── no_retry.py
│   ├── no_cot.py
│   ├── no_rerank.py
│   ├── no_planner.py
│   └── no_multi_hop.py
├── runs/                       # raw pipeline outputs, dated
│   └── papermind_2026-05-12.json
├── scores/                     # scored outputs (after applying scorers)
│   └── papermind_2026-05-12_scored.json
└── reports/
    └── report_v1.md            # the headline deliverable
```

Keeping `runs/` separate from `scores/` is the discipline that lets you re-score without re-running. Adding a new metric next month should not require firing up the pipeline again.

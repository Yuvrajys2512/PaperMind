# PaperMind — Master Development To-Do

> Tick off items as you complete them. Order reflects priority and dependencies.
> Sessions 1–7 are already done and listed for reference only.

---

## Completed (Reference)

- [x] **Session 1** — Unified LLM Layer: Gemini as provider #1, exception handling fixed, section_detector/multi_hop migrated, intent_detector deleted
- [x] **Session 2** — Dependencies + Storage: requirements.txt rewritten, thread-safe concurrent uploads, chunk params corrected to 512/100
- [x] **Session 3** — Pipeline Efficiency: shared embedding model singleton, eliminated duplicate evaluate_answer calls, Unicode crash fix
- [x] **Session 4** — Evidence Grading UI: per-sentence DIRECT/INFERRED/REMOVED chips in frontend
- [x] **Session 5** — Table Extraction: pdfplumber table → markdown chunks, violet table chips in sources
- [x] **Session 6** — Structured Logging: request IDs, timing breakdown, 60s timeout, 504 on hang
- [x] **Session 7** — Multi-paper Comparison: compare_retriever.py, /query accepts paper_ids list, A/B label chips in UI
- [x] **PaperMind 2.0 Phase 1** — Live Research Discovery: arXiv + Semantic Scholar search, PDF import, temporary session collections, RAG pipeline integration

---

## Active Bug Fixes

- [ ] **Fix Semantic Scholar results** — every search currently returns only arXiv papers; S2 has citation counts, venue info, and broader non-CS coverage. Debug the rate limit / API response issue so both sources return results (BUGS.txt)

---

## Phase 6 — End-to-End Evaluation Set

> Goal: one defensible headline number that turns PaperMind from "interesting demo" into "measured engineering result."
> Reference: `PHASE_6.md`

### 6.1 Dataset Construction (~15–20 hours)

- [ ] Pick 10–12 source papers — mix of short methods papers, long surveys, math-heavy, empirical, recent + classic
- [ ] Write 5 human anchor questions per paper (gold-standard: clear citation, varied types, correct difficulty tag)
- [ ] Generate ~250 candidate Q&A pairs via LLM (Claude Opus / GPT-4) using anchor questions as few-shot examples
- [ ] Human verification pass — review every generated question: unambiguous? ground truth in paper? citation correct? difficulty honest? (expect 30–40% rejection)
- [ ] Diversity check — sort by type + difficulty; hand-write any missing buckets (must have factual, mechanism, multi-hop, comparison, critique, summarization, out-of-domain per paper)
- [ ] Build 30-question dev set (`eval/dev_set.json`) — separate from the eval set, used only for tuning thresholds
- [ ] Freeze `eval/dataset.json` — 300 verified Q&A pairs with full schema (id, paper_id, question, ground_truth, ground_truth_aliases, source.section/page/quote, type, difficulty, out_of_domain)

### 6.2 Eval Harness

- [ ] Create `eval/harness.py` — `harness.run(dataset, pipeline, output)` loops through questions, calls pipeline, saves raw results to `eval/runs/` without scoring
- [ ] Verify harness saves: answer text, confidence, sources, duration, attempts, evidence grades

### 6.3 Scorers (`eval/scorers/`)

- [ ] `eval/scorers/exact_match.py` — normalize whitespace/casing/punctuation, check against ground_truth + aliases
- [ ] `eval/scorers/f1.py` — token F1 overlap (SQuAD-style)
- [ ] `eval/scorers/llm_judge.py` — Claude Opus or GPT-4 rates 0–5; run each question twice and average; calibrate on anchor questions
- [ ] `eval/scorers/citation_check.py` — binary match of pipeline source.section + source.page vs ground-truth source
- [ ] `eval/scorers/refusal_check.py` — for out_of_domain=true questions: did pipeline refuse / return warning? (binary: refused vs hallucinated)
- [ ] `eval/scorers/__init__.py`

### 6.4 Baselines (`eval/baselines/`)

- [ ] `eval/baselines/vanilla_rag.py` — top-5 dense vector retrieval only, generic prompt, no CoT/plan/grader/retry
- [ ] `eval/baselines/raw_llm.py` — no retrieval at all, just `chat_completion("Paper: <title>. Question: <Q>")`
- [ ] `eval/baselines/bm25_only.py` — optional: isolates retrieval contribution of hybrid vs BM25-only

### 6.5 Ablations (`eval/ablations/`)

- [ ] `eval/ablations/no_grader.py` — PaperMind with evidence_grader disabled
- [ ] `eval/ablations/no_retry.py` — PaperMind with MAX_ATTEMPTS = 1
- [ ] `eval/ablations/no_cot.py` — PaperMind with simple prompt, no scratchpad
- [ ] `eval/ablations/no_rerank.py` — PaperMind skipping CrossEncoder reranking
- [ ] `eval/ablations/no_planner.py` — PaperMind skipping query_planner, fixed retrieval config
- [ ] `eval/ablations/no_multi_hop.py` — PaperMind always single-pass retrieval

### 6.6 Run + Score

- [ ] Run all 5 pipelines (PaperMind full + 4 baselines/ablations) × 300 questions — save raw outputs to `eval/runs/`
- [ ] Apply all scorers to each run — save to `eval/scores/`
- [ ] Re-run any ablations on full 300-question set (no subsets)

### 6.7 Report

- [ ] Write `eval/reports/report_v1.md` — overall + per-type (factual/mechanism/multi-hop/OOD) breakdown table for all pipelines, cost per question (median), written conclusion identifying which features earned their complexity
- [ ] Verify headline number is statistically meaningful (N=300 → differences ≥ 4 points are real)

---

## PaperMind 2.0 — Phase 2: Writing Assistant

> Reference: `PaperMind_2.0.md` — build after Phase 6 eval is complete.

### 2a — Humanize / Rewrite (Easy Win)

- [ ] Add `/rewrite` endpoint in `api/main.py` accepting text + mode (academic / plain / concise)
- [ ] Write rewrite prompt in `ingestion/generator.py` or new `services/humanizer.py`
- [ ] Add Rewrite panel in frontend (text input → mode selector → output with copy button)
- [ ] Write UI copy positioning it as "writing improvement tool" not "AI bypass"

### 2b — AI Detection

- [ ] Create `services/ai_detector.py` with swappable backend interface (GPTZero first)
- [ ] Integrate GPTZero free tier (10,000 words/month) — `POST /ai-detect` endpoint
- [ ] Add Sapling AI as fallback backend
- [ ] Frontend: AI Detection panel — paste text → show confidence score (not binary verdict), surface uncertainty to user

### 2c — Plagiarism Check

- [ ] Create `services/plag_checker.py` — cosine similarity search against all indexed ChromaDB papers
- [ ] Add `POST /plag-check` endpoint — returns matched passages with similarity scores + source citations
- [ ] Frontend: Plagiarism Check panel — paste text → expandable matched passages with similarity %
- [ ] (Later / paid) Hook in Copyleaks or iThenticate API for broader corpus check

---

## PaperMind 2.0 — Phase 3: Research Workspace

> Requires Phase 1 library feature (saving search results) as a prerequisite.

- [ ] **Personal Library** — "Save to library" button on search results moves paper from temp session collection to user's permanent ChromaDB collection; persists across sessions
- [ ] **Draft Editor** — rich text editor with inline citation insertion (pull from saved papers), AI-assisted paragraph expansion via `/rewrite`
- [ ] **Citation Formatter** — output saved papers' metadata as APA, MLA, Chicago, BibTeX
- [ ] **Research Timeline** — track papers read, papers cited, papers in queue (UI sidebar or dedicated page)

---

## Notes

- **Never tune thresholds against `eval/dataset.json`** — use `eval/dev_set.json` for that. The eval set is frozen once built.
- **LLM judge must be Claude Opus or GPT-4** — cheaper models produce noisy ratings that swamp real signal.
- **Ablations must run on the full 300 questions** — running on subsets makes conclusions noise.
- **Report failures honestly** — if a feature lifts accuracy 1 point and costs 3× latency, say so.

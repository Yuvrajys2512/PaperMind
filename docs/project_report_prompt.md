# Prompt: Generate PaperMind Project Report

Paste everything below the line into Claude to generate the full report as a separate file. It doesn't need repo access — all the technical facts it needs are inlined below, verified directly against the source (`ingestion/`, `api/`, `eval/`, `discovery/`, `frontend/src/`, `product/`), not just README claims. Where the README was stale, the facts below reflect the actual code.

---

Before writing, ask me for these details if I haven't already given them, then generate the report:

- Student/author name, roll number / registration number
- College/institution name
- Internship provider / organization name (if applicable) and mentor/guide name
- Internship or project duration (start–end dates)
- Preferred output format: Markdown (.md), or a Word-ready document

## Task

Write a complete project report for **PaperMind**, using the facts below as ground truth for every chapter — do not invent generic filler, and do not contradict these facts. Reproduce the exact chapter/section structure and per-section style listed further down (derived from a reference internship report's format — only the *structure/style* is being reused, not its content).

### Project facts

**What it is:** PaperMind is an agentic research assistant that answers questions about uploaded research papers. Unlike a conventional RAG wrapper, it *plans* each query, retrieves with a **hybrid** dense + lexical search, **reranks** the results, and then **grades its own answer sentence-by-sentence against the source evidence**, stripping any claim it can't support. Every answer ships with per-sentence evidence indicators, cited sources (section + page), a confidence score, and an honest refusal ("I can't answer that from this paper") when the paper doesn't contain the answer. It also runs five automated audit tools for paper authors and reviewers, and is shipped as a multi-tenant SaaS product with auth, billing, and quotas.

Tagline used on the product's landing page: *"Ask questions about any research paper. Get answers you can verify."*

**Query pipeline (`ingestion/pipeline.py`, function `answer_query`), actual orchestration:**
1. **Off-topic guard** (`off_topic_guard.py`) — runs once before anything else; short-circuits with a canned refusal if the question isn't about the paper.
2. **Route** (`query_router.py`, `route_query`) — a single orchestrating step that internally: plans the query (`query_planner.py` — decomposition, `answer_type` classification into one of 8 types: factual, summarization, critique, comparison, mechanism, causal_explanation, hypothetical, analysis), looks up a retrieval config (`retrieval_k`/`llm_k`) for that answer type, then branches: complex queries go through **multi-hop retrieval** (`multi_hop.py` — decomposes into 2–3 sub-questions, retrieves per sub-question, merges/dedupes by chunk ID), simpler queries go through **single-pass hybrid retrieval** optionally seeded by **HyDE** (`hyde.py` — generates a hypothetical 3–4 sentence answer passage with an LLM and uses that, not the raw question, as the dense-retrieval seed, while BM25 still uses the raw query). The routed chunks are then **reranked** (`reranker.py`, `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`) down to `llm_k` chunks.
3. **Generate** (`generator.py`) — chain-of-thought reasoning over the retrieved chunks and the plan → a structured answer.
4. **Pre-evaluate** (`evaluator.py`) — faithfulness + answer-relevancy scored on the raw generated answer *before* grading.
5. **Evidence grade** (`evidence_grader.py`) — classifies every sentence in the answer as `DIRECT` / `INFERRED` / `UNSUPPORTED` against the retrieved chunks and strips `UNSUPPORTED` sentences (unless disabled via the `PAPERMIND_DISABLE_GRADER` ablation flag).
6. **Re-evaluate** — only re-runs faithfulness/relevancy scoring on the cleaned answer if the pre-grade faithfulness was ≤0.75 *and* sentences were actually removed; otherwise it reuses the pre-evaluation score, avoiding a redundant LLM call.
7. **Out-of-domain check** — on the first attempt only: if answer-relevancy < 0.05, the question is treated as out-of-domain and retries are skipped entirely.
8. **Retry** (`retry_engine.py`) — if confidence < 50 and the answer isn't out-of-domain, `diagnose_failure` classifies why the attempt was weak and re-routes for another attempt, up to `MAX_ATTEMPTS` (env-overridable via `PAPERMIND_MAX_ATTEMPTS`). The pipeline tracks the **highest-confidence attempt across retries**, not just the last one.

Two additional scores are computed and attached to every result alongside faithfulness/relevancy: **retrieval quality** and **numeric grounding** (`compute_retrieval_quality`, `compute_numeric_grounding`).

Three ablation flags gate individual stages for controlled experiments: `PAPERMIND_DISABLE_GRADER`, `PAPERMIND_DISABLE_RERANK`, `PAPERMIND_DISABLE_HYDE`.

**Multi-paper comparison** uses a separate path (`compare_papers()`): plan → `compare_retriever.py` (interleaved retrieval from both papers' Chroma collections, tagged `paper_label` "A"/"B", results interleaved A1,B1,A2,B2… so both papers are represented in context) → rerank → generate → grade → evaluate. No retry loop, no multi-hop.

**Ingestion pipeline (one-time per paper, `ingest_document.py`):**
- **PDF parsing** (`pdf_parser.py`) — pdfplumber only; no docling, pypdfium2, or fitz/PyMuPDF anywhere in the codebase.
- **Section/heading detection** (`section_detector.py`) — a heuristic geometry-scoring pass (font size vs. body size, word count, trailing punctuation, numeric-prefix regex, capitalization, vertical gap to the previous line; a candidate needs a score ≥8), followed by an **LLM confirmation call** that classifies each candidate as SECTION / SUBSECTION / NONE, then the document is split on confirmed headings up to "References."
- **Table extraction** (`table_extractor.py`) — caption-anchored, not geometry-based: a regex locates "Table N:" captions as reliable text anchors, pdfplumber's text-strategy extraction is confined to the crop region below each caption (deliberately avoiding pure-geometry line/text table detection, which misfires on borderless tables and figure grids), then each row is linearized into header-anchored sentences (e.g. `"BERT — SQuAD F1: 0.79"`) plus the raw markdown grid is kept, giving the embedded chunk real lexical/semantic retrieval anchors instead of a bare digit grid.
- **Chunking** (`chunker.py`) — token-aware chunking via `tiktoken`.
- **Embedding** (`ingestion/models.py`, shared by `embedder.py` / `embedder_worker.py`) — `BAAI/bge-small-en-v1.5` (384-dim), an asymmetric retrieval model: queries get a BGE instruction prefix ("Represent this sentence for searching relevant passages: "), passages do not. *(Note: this superseded an earlier `all-MiniLM-L6-v2` embedder — same dimensionality, different vector space, so any old collection had to be re-ingested.)*
- Stored in **ChromaDB**, kept on local/ephemeral disk by design — fully regenerable from R2 on startup, not durably persisted itself.

**LLM layer (`ingestion/llm_client.py`):** a single OpenAI-compatible client implementing its own multi-provider failover chain, in order:
1. **Groq** — not one slot but an auto-discovered *key pool* (`GROQ_API_KEY`, `GROQ_API_KEY_2`…`_8`), each registered as an independent budget slot (`Groq-1`, `Groq-2`, …), all running `llama-3.3-70b-versatile`.
2. **Gemini** — `gemini-2.5-flash-lite` (chosen after `gemini-2.0-flash` was found to have zero free quota, and `gemini-2.5-flash` was found to burn budget on hidden "thinking" tokens).
3. **Mistral** — `mistral-small-latest`.
4. **Cerebras** — `gpt-oss-120b`, deliberately placed last because it was observed to return null content intermittently.

Every client has the SDK's own `max_retries` disabled (`max_retries=0`) — PaperMind owns the fallback loop itself, so a rate-limited provider fails over almost instantly instead of burning time on SDK-internal retries. On a pass where every provider failed but at least one failure looked like a self-clearing per-minute rate limit (matched by keyword: "rate limit reached", "requests per minute", "tokens per minute", "rpm", "tpm", etc.), the client sleeps on an escalating backoff schedule of **5s → 15s → 30s** and retries the whole chain, up to 3 extra passes; daily-quota/auth/not-found errors skip backoff entirely since retrying within the window can't help. A `pin=(provider, model)` option can force a single provider with no fallback, used for controlled experiments (e.g. weak-vs-strong generator comparisons). Thread-local call statistics (call count, providers used, tokens in/out, projected cost) are tracked per request for the usage/billing dashboard, using a hardcoded per-1M-token pricing table (Groq $0.59/$0.79, Gemini $0.10/$0.40, Mistral $0.20/$0.60, Cerebras $0.35/$0.75) to project paid-tier cost even though the deployed keys are free-tier today.

**Core features:**
- Grounded Q&A over uploaded papers with cited sources (section + page) and per-sentence evidence confidence
- Self-verifying answers (evidence grader removes unsupported claims before the user sees them)
- Honest refusals for off-topic / out-of-domain questions
- Multi-paper comparison (one question, structured side-by-side answer across two papers, interleaved retrieval)
- Paper discovery — concurrent arXiv + Semantic Scholar search, merged/deduplicated, and ingest directly
- Auto-generated glossary and related-work recommendations per paper
- Rewrite tool — rephrase passages at different reading levels (academic / plain / concise)
- Polished chat UX: SSE token streaming, split-view parallel streams, notes panel, session export, in-conversation search, in-app PDF viewer that jumps to a cited page/section

**Author-facing audit tools (five independent audit engines, `ingestion/`), each returns a verdict scheme rather than freeform prose, and each is designed to fail closed (never raise; returns a `*_failed` report on error) and to be conservative — an accusatory verdict is downgraded to a neutral one whenever the evidence citation is missing:**
- **Claim–Evidence Audit** (`claim_auditor.py`, `audit_paper`) — extracts the paper's own checkable claims (performance/scope/capability/comparison) from the abstract/intro/conclusion, retrieves evidence biased toward results/tables, and grades each claim **GROUNDED | OVERCLAIM | SCOPE_MISMATCH | UNVERIFIABLE**; outputs a `trust_score` (grounded / total). Streamed live via SSE (`POST /papers/{id}/audit/stream`), rendered in the `AuditPanel` React component.
- **Reviewer/Weakness Audit** (`reviewer_auditor.py`, `review_paper`) — checks the paper against 6 fixed methodological dimensions (baselines, ablations, statistical rigor, sample size, threats to validity, related work) with verdicts **OK | WEAK | MISSING**, plus a deterministic MISSING→WEAK downgrade guard when a matching section exists; outputs a `review_score`. SSE (`/papers/{id}/review/stream`), rendered in `ReviewPanel`.
- **Numbers Audit** (`numbers_auditor.py`, `audit_numbers`) — extracts numeric claims (metric/value/dataset) from the abstract/intro/conclusion and checks them against the paper's results/tables, verdicts **MATCH | MISMATCH | NOT_FOUND**; outputs a `consistency_score`. SSE (`/papers/{id}/numbers/stream`), rendered in `NumbersPanel`.
- **Structure Audit** (`structure_auditor.py`, `check_structure`) — a venue-fit checklist across 8 possible structural components (related work, method, experiments, limitations, ethics, broader impacts, reproducibility, conclusion) against one of 7 selectable venue rubrics (generic, NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR) marking each required/optional, verdicts **PRESENT | THIN | MISSING**; outputs `completeness_score` and a `required_missing` count. SSE (`/papers/{id}/structure/stream`), rendered in `StructurePanel` with a venue selector, backed by `GET /venues`.
- **Novelty Scout** (`novelty_scout.py`, `find_related_work`) — the only audit that looks outside the paper: distills the draft's opening into 2–3 search queries, searches **Semantic Scholar**, rates each candidate's closeness **HIGH | MEDIUM | LOW** with an overlap/differentiator note, and synthesizes an overall positioning summary. SSE (`/papers/{id}/novelty/stream`), rendered in `NoveltyPanel`.
- **Write Mode / "My Drafts"** — lets authors run these same five audit engines against their own unpublished, pre-submission drafts (`paper_type="draft"`) before submitting to a venue, via `DraftsPage` (upload/library for drafts) and `DraftReviewPage` (tabs across all five audit panels for one draft).

**Multi-tenant SaaS layer:**
- **Auth** (`api/auth.py`) — Clerk-issued JWTs verified manually (no Clerk backend SDK): fetches Clerk's JWKS, verifies RS256 signature + issuer with PyJWT, with a 60s clock-drift leeway. Admin routes gated by a `PAPERMIND_ADMIN_USER_IDS` allow-list.
- **Storage** (`api/storage.py`) — Neon Postgres (via `psycopg_pool`) holds the `papers` registry; Cloudflare R2 (S3-compatible, via `boto3`, chosen over S3 for no egress fees) holds PDF blobs and every audit report's cached JSON (one blob per paper per report type). Distinguishes papers a user owns (for writes/delete) from papers a user can read (includes a shared, quota-exempt demo paper set owned by a system `DEMO_USER_ID`).
- **Concurrency** (`api/concurrency.py`) — a thread-safe singleton ChromaDB `PersistentClient` (double-checked locking) plus a per-paper `RLock` registry, so ingest/query/delete on the same paper can't race.
- **Billing** (`api/billing.py`) — Stripe Checkout session creation, customer portal, and a signature-verified webhook that flips a user's `tier` between `free`/`pro` on subscription lifecycle events. The Pro tier is **$9/month flat, unlimited** papers/queries/audits.
- **Usage & quotas** (`api/usage.py`) — `users`/`usage_events` Postgres tables; quota enforcement as FastAPI dependencies. Free tier: **3 papers, 20 queries/month, 10 audits/month** (env-overridable; the audit quota is shared across all five audit types); Pro tier is explicitly unlimited. Also computes an admin-only aggregate usage/capacity view.
- **Startup resilience** — on boot, a background thread regenerates any ChromaDB collection missing on the (ephemeral-disk) host by re-downloading each ready paper's PDF from R2 and re-running ingestion — idempotent, toggleable via `PAPERMIND_REGENERATE_ON_STARTUP`.
- **Hardening** — env-driven CORS allow-list (no wildcard), a request-ID middleware that tags every request into Sentry and the response headers, `slowapi` per-IP rate limiting (120 requests/minute), and streamed magic-byte + size-cap validation on every upload before it touches disk.
- **Observability** — Sentry error tracking (fail-open, no-op without a DSN) + structured JSON operation logging (`api/logger.py`) + PostHog product analytics.
- Public landing page, Terms of Service / Privacy pages, and the shared demo paper set for first-time visitors.

**Evaluation (`eval/`) — the research/rigor angle:** a custom evaluation harness built on the **QASPER** benchmark (Dasigi et al., 2021 — question answering over NLP papers). Components: `qasper_loader.py` (downloads/parses the official QASPER v0.3 tarball), `qasper_adapter.py` (ingests a QASPER paper directly into PaperMind's Chroma store, skipping PDF parsing since QASPER is already structured), `metrics.py` (pure scoring: token-F1, answerable-accuracy, evidence recall/F1), `judge.py` (LLM-as-judge, N=3 independent calls per answer, majority-voted CORRECT/PARTIAL/INCORRECT to control judge variance), `run_eval.py` (runs each QASPER question through the real pipeline and scores it), `run_ablations.py` (runs the harness across **6 ablation configs** — `full`, `baseline` [grader+rerank+HyDE off, single attempt], `no_grader`, `no_rerank`, `no_hyde`, `no_retry` — subprocess-per-config, then tabulates a comparison matrix), and `analyze_grader.py` (a paired-design analysis isolating whether the grader helped or hurt on each individual answer, comparing the pre-grader and post-grader text from the *same* generation call).

Headline research question: *"Does self-grading the generated answer improve faithfulness and/or judged correctness without dropping Answer-F1 — and at what latency cost?"*

Key results (paired design, n=14 answerable questions):
- Pre-grader vs post-grader mean judge score: 0.607 → 0.607 (net effect on judged correctness ≈ zero: 3 helped / 3 hurt / 8 neutral)
- The grader **does lift faithfulness**, and the cases where it hurts follow an explainable pattern: it strips correct *negative* answers, which have no direct textual support to cite.
- System performance (dev split, n=14): evidence recall@k = **0.98**, faithfulness = **0.94**, answerable accuracy = **1.00**, mean latency ≈ **9.9s/question**, mean ≈ **4.9 LLM calls/question**.
- Conclusion reported honestly: grading trades a little correctness for measurably higher faithfulness — not a free win. Null/mixed results are reported as-is, not hidden. Numbers are explicitly flagged as preliminary given the small n and free-tier provider variance.

**Paper discovery (`discovery/`):** `router.py` exposes `POST /discovery/search` and `POST /discovery/import`; `sources/arxiv.py` and `sources/semantic_scholar.py` search arXiv and Semantic Scholar **concurrently** (`asyncio.gather`); results are merged/deduplicated by arXiv ID or fuzzy title match, with Semantic Scholar metadata (citation count, venue) enriching matched arXiv entries; sort order is has-PDF first, then citation count descending, then year descending; `fetcher.py` downloads the chosen PDF into R2 and registers it.

**Tech stack (verified):**
- Frontend: React 19.2.5, Vite 8.0.10, Tailwind CSS, react-markdown, react-pdf
- Backend: FastAPI, Uvicorn, Python 3.12 (per the Dockerfile)
- Retrieval: ChromaDB (dense vector store), rank-bm25 (lexical/BM25), sentence-transformers CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (rerank)
- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, asymmetric query/passage encoding)
- Parsing/tokenization: pdfplumber, tiktoken
- LLM providers: Groq (key-pooled), Gemini, Mistral, Cerebras — via the `openai` SDK's OpenAI-compatible interface, with a self-managed failover/backoff loop
- Numerical scoring: numpy
- Storage: psycopg / psycopg_pool (Neon Postgres), boto3 (Cloudflare R2)
- Auth: pyjwt[crypto] (Clerk JWT/JWKS, RS256)
- Billing: stripe
- Rate limiting: slowapi
- Error tracking: sentry-sdk (optional, no-op without a DSN)
- Dev tooling: Docker (`python:3.12-slim` base), GitHub Actions CI (tests + lint + frontend build on every push), Git/GitHub

**API surface (`api/main.py` plus mounted routers):** `GET /health`, `POST /upload` (async ingest, poll `GET /status/{paper_id}`), `GET /papers`, `DELETE /papers/{paper_id}`, `GET /papers/{paper_id}/pdf`, `GET /papers/{paper_id}/glossary`, `GET /papers/{paper_id}/recommendations`, `POST /rewrite`, `POST /query`, `POST /query/stream` (SSE), `POST /papers/{paper_id}/audit/stream`, `POST /papers/{paper_id}/review/stream`, `POST /papers/{paper_id}/novelty/stream`, `POST /papers/{paper_id}/structure/stream`, `POST /papers/{paper_id}/numbers/stream`, `GET /venues`, `GET /usage`, `GET /admin/usage`; plus `discovery_router` (`/discovery/search`, `/discovery/import`) and `billing_router` (`/billing/checkout`, `/billing/portal`, `/billing/webhook`).

**Frontend pages:** `LandingPage` (public marketing/pricing/sign-in), `UploadPage` (signed-in home, PDF upload + status polling), `LibraryPage` (uploaded reference papers + shared demo set), `ChatPage` (main Q&A UI — streaming, metric rings, audit/review panels), `DiscoverPage` (arXiv/Semantic Scholar search + import), `RewritePage` (passage rewriting), `DraftsPage` (drafts library/upload for Write Mode), `DraftReviewPage` (tabbed hub across all five audit panels for a draft), `BillingPage` (tier/usage meters, upgrade/manage CTA), `AdminPage` (aggregate usage dashboard), `LegalPage` (ToS/Privacy).

**Frontend components:** `AuditPanel`, `ReviewPanel`, `NumbersPanel`, `StructurePanel`, `NoveltyPanel` (one SSE-driven panel per audit engine), `MetricRing` (confidence/faithfulness/relevancy gauge), `PDFPreviewPanel` (in-app PDF viewer that jumps to a cited page/section), `CitedAnswer` (landing-page marketing mockup), `NavButton`, `Wordmark`.

**Repo layout:** `api/` (FastAPI app, auth, billing, storage, usage, concurrency, logging), `ingestion/` (query pipeline + retrieval/grading/LLM/audit modules), `discovery/` (arXiv + Semantic Scholar search and fetch), `eval/` (QASPER harness — loader, adapter, metrics, judge, run_eval, run_ablations, analyze_grader), `frontend/` (React + Vite client), `research/` (methodology write-ups and ablation results), `product/` (positioning, pricing, launch checklist, deployment guide), `docs/` (internal planning notes).

**Internal engineering docs worth citing for the "hardening" story:** `CONCURRENCY_FIXES.md` (thread-safe singleton ChromaDB client + per-paper locks to stop races between concurrent ingest/query/delete on the same paper), `ERROR_HANDLING_SUMMARY.md` (structured operation logging + request-ID tracing + Sentry across critical endpoints), `LOGGING_GUIDE.md` (reference for the logging system — rotating JSON operation log, JSONL query telemetry), `WEAK_SPOTS_FIXED.md` (closure summary tying the three pre-launch hardening fixes together). The deployment guide itself lives at `product/DEPLOYMENT.md`.

**Deployment target:** backend on Hugging Face Spaces' free tier (chosen for its 16GB RAM, since Render's free 512MB tier can't run torch), frontend on Vercel; ChromaDB kept on ephemeral disk by design and fully regenerated from R2 on startup rather than persisted directly.

**Pricing/positioning (for Applications/Conclusion context):** Free tier — 3 papers, 20 queries/month, 10 audits/month (shared across all five audit types). Pro tier — $9/month flat, unlimited. Landing-page positioning leads with verifiable citations, evidence-graded answers, a visible confidence score, and results benchmarked on QASPER rather than anecdote.

**Honest limitations / open items (useful for Outcomes & Learnings and Conclusion):** evaluation numbers are preliminary (small n=14, free-tier provider variance); the evidence grader's threshold still strips some correct negative-answer sentences and tuning that is an open next step; storage is not yet externalized for multi-instance horizontal scaling beyond the current single-writer-friendly design; as of the last internal status check, all planned pre-launch feature work (multi-tenancy, LLM cost/quota economics, billing, deployment hardening, observability/legal minimum) was code-complete, with only manual operations (key rotation, actual deploy, live billing end-to-end test, demo media) remaining — i.e. shipping the working system was prioritized over polishing the QASPER research angle further, though both exist side by side in the repo.

## Chapter/section structure to reproduce exactly

Titles must match; do not add, remove, merge, or reorder top-level chapters:

1. Declaration
2. Certificate
3. Consent Form
4. Table of Contents
5. Chapter 1: Abstract
6. Chapter 2: Introduction
7. Chapter 3: Literature Review
8. Chapter 4: Objectives
   - 4.1 Primary Objectives
   - 4.2 Supplementary Goals
9. Chapter 5: Applications
10. Chapter 6: Technologies Used
    - 6.1 Programming Language
    - 6.2 Data Processing & NLP Libraries *(reference report's "Data Analysis and Manipulation Libraries" — retitled since this project isn't tabular-data-centric; use numpy, tiktoken, pdfplumber)*
    - 6.3 Visualization & Frontend UI *(reference report's "Data Visualization Libraries" — retitled; this project surfaces confidence/evidence visually through React components like `MetricRing`, `AuditPanel`, `ReviewPanel`, `PDFPreviewPanel`, not charting libraries)*
    - 6.4 Retrieval & Machine Learning Frameworks *(ChromaDB, sentence-transformers CrossEncoder + BGE embedder, rank-bm25)*
    - 6.5 Development Environments *(VS Code, Uvicorn dev server, Vite dev server, Docker)*
    - 6.6 Documentation and Reporting Tools *(README.md, product/DEPLOYMENT.md, LOGGING_GUIDE.md, research/ write-ups)*
    - 6.7 Version Control *(Git, GitHub, GitHub Actions CI)*
11. Chapter 7: Methodology & Implementation — use these 11 subsections (retitled from the reference report's finance-specific ones to PaperMind's real pipeline stages, same granularity):
    - 7.1 Problem Understanding and Requirement Gathering
    - 7.2 PDF Parsing, Section Detection and Table Extraction
    - 7.3 Chunking and Embedding Generation
    - 7.4 Off-Topic Guard and Query Planning
    - 7.5 Hybrid Retrieval, Multi-Hop and HyDE
    - 7.6 Cross-Encoder Reranking
    - 7.7 Answer Generation
    - 7.8 Evidence Grading (Self-Verification)
    - 7.9 Faithfulness/Relevancy Evaluation and Retry Engine
    - 7.10 Author-Facing Audit Engines (Claim, Reviewer, Numbers, Structure, Novelty)
    - 7.11 Multi-Tenancy, Billing and Deployment Hardening
12. Chapter 8: Outcomes and Learning
    - 8.1 Technical Outcomes
    - 8.2 Practical Learnings
    - 8.3 Professional Growth
    - 8.4 Key Takeaways
13. Chapter 9: Detailed Project Execution — one subsection per task, using this breakdown:
    - 9.1 Task 1 — PDF Ingestion & Parsing Pipeline
    - 9.2 Task 2 — Section Detection & Table Extraction
    - 9.3 Task 3 — Chunking & Embedding
    - 9.4 Task 4 — Off-Topic Guard & Query Planner
    - 9.5 Task 5 — Hybrid Retrieval, Multi-Hop Decomposition & HyDE
    - 9.6 Task 6 — Cross-Encoder Reranking
    - 9.7 Task 7 — Answer Generation & Multi-Paper Comparison
    - 9.8 Task 8 — Evidence Grading
    - 9.9 Task 9 — Evaluation & Retry Engine
    - 9.10 Task 10 — Claim–Evidence Audit Engine
    - 9.11 Task 11 — Reviewer/Weakness Audit Engine
    - 9.12 Task 12 — Numbers, Structure & Novelty Audits
    - 9.13 Task 13 — Multi-Tenancy, Billing & Deployment
    - 9.14 Task 14 — QASPER Evaluation Harness & Ablation Study
14. Chapter 10: Conclusion
15. Chapter 11: References

## Style guidance per section (reproduce the *style*, filled in with the project facts above)

- **Abstract**: 3 paragraphs — the problem context (trust/hallucination problem with plain RAG systems over research papers), what was built and with what stack, what techniques were used and the overall outcome (ship the ablation-backed evaluation result honestly, including the null result).
- **Introduction**: 3–4 paragraphs moving from broad context (rise of AI research assistants, the hallucination/trust problem with plain RAG) → the specific problem PaperMind solves → the project name (bolded) and its purpose/scope.
- **Literature Review**: prose paragraphs on RAG, hybrid dense+lexical retrieval, cross-encoder reranking, self-verification of LLM outputs, and the QASPER benchmark — cite Dasigi et al. (2021) and the libraries/tools used.
- **Objectives**: intro paragraph, then 4.1 Primary Objectives and 4.2 Supplementary Goals as bullet lists, each bullet **bold short title:** one-paragraph explanation, drawn from the project facts above.
- **Applications**: bullet list, each bullet **bold application name**: one-sentence description — draw from the Core features / audit tools lists above.
- **Technologies Used**: subsections by category as listed above, each a bullet list of **bold tool/library name**: one-sentence description of what it's used for in this project, using the verified Tech stack facts above.
- **Methodology & Implementation**: each subsection = a short intro paragraph plus a bullet list of concrete steps, drawn from the pipeline/ingestion facts above — including the real nesting (e.g. planning, HyDE, multi-hop and reranking all happen inside the "route" step) rather than flattening it into something it isn't.
- **Outcomes and Learning**: each subsection = bullet list of **bold outcome/skill**: 1–2 sentence elaboration, drawn from the Evaluation results and Honest limitations facts above.
- **Detailed Project Execution**: one subsection per task as listed above, each with a bullet list of what was implemented, plus a short representative code snippet where it clarifies the point (real module/function names are given above, e.g. `evidence_grader.py`, `route_query`, `compare_papers`).
- **Conclusion**: summary paragraphs on what was achieved, referencing the evaluation result honestly (the grader's near-zero net effect on correctness but real faithfulness lift), followed by a short breakdown of how the retrieval/grading pipeline, the five audit tools, and the multi-tenant SaaS layer each contributed.
- **References**: numbered list, each entry **bold source name**, one-line description, and link — include QASPER (Dasigi et al., 2021, allenai.org/data/qasper), FastAPI, React, ChromaDB, sentence-transformers, rank-bm25, Groq/Gemini/Mistral/Cerebras API docs, Stripe, Clerk.

## Output

Write the finished report as a new file (e.g. `docs/project_report.md`), using Markdown headings for chapters/subsections. Keep length and depth comparable to a formal ~35–40 page report.

# Project Journal — PaperMind

**Name:** Yuvraj Srivastava
**Project Title:** PaperMind — AI Research Paper Assistant (RAG-based paper Q&A, live literature discovery, evaluation harness, claim/reviewer auditing, and a pre-submission Write Mode)
**Duration:** 04 May 2026 – 04 July 2026 (~9 weeks)
**Repository:** PaperMind (solo build)

---

## 1. Objectives of the Reporting Period

- Take the existing single-paper Q&A pipeline (PDF ingestion, chunking, hybrid retrieval) and upgrade answer quality with multi-hop retrieval, query planning, chain-of-thought reasoning, and evidence grading.
- Add table extraction, multi-paper comparison, and plan a production deployment.
- Build live literature discovery (arXiv / Semantic Scholar search + PDF import) directly into the RAG pipeline, with an off-topic query guard.
- Build an independent evaluation harness against the QASPER benchmark to objectively measure and ablate retrieval/generation quality.
- Ship a full study-tools UX layer: glossary, audience modes, flashcards, source-linked PDF preview, a writing assistant, notes, session export, and split view.
- Bring the codebase to production hygiene: repo cleanup, CI/CD, linting, and test suites.
- Turn the prototype into a real multi-user product: Clerk authentication, per-user data isolation (Neon/R2), LLM cost tracking and quotas, Stripe billing, and deployment hardening (Docker, CORS, rate limiting, Sentry/PostHog).
- Extend the product beyond Q&A into automated paper auditing (claim-vs-evidence and reviewer/weakness checks) and ship a "Write Mode" that lets authors run those audits on their own drafts pre-submission.

---

## 2. Weekly Summary

| Week | Dates | Focus Area | Outcome |
|------|-------|------------|---------|
| 1 | 04 May – 10 May 2026 | Multi-hop retrieval, query planning, CoT reasoning, evidence grading | Answer quality upgraded from single-shot to a reasoning pipeline |
| 2 | 11 May – 17 May 2026 | Table extraction, multi-paper compare, deployment planning | Feature-complete single-user prototype; deployment path scoped |
| 3 | 18 May – 24 May 2026 | Eval instrumentation, live research discovery (arXiv/S2), off-topic guard | External literature search integrated into the RAG pipeline with guardrails |
| 4 | 25 May – 31 May 2026 | Repo prep + full QASPER eval harness build | Objective, benchmarked proof the full pipeline beats baseline |
| 5 | 01 Jun – 07 Jun 2026 | Generator/grader fixes, study tools, repo cleanup, CI/CD, ingestion hardening | Product-grade UX layer + production-hygiene codebase |
| 6 | 08 Jun – 14 Jun 2026 | Chat UX fix, production documentation, multi-tenancy research | Documented system; auth/storage migration path designed |
| 7 | 15 Jun – 21 Jun 2026 | Multi-tenancy, billing, hardening, observability, launch checklist, audit engines | Full SaaS launch checklist completed; automated paper auditing shipped |
| 8 | 22 Jun – 28 Jun 2026 | Write Mode build-out (draft audits) | Claim/reviewer engines adapted to work on unpublished drafts |
| 9 | 29 Jun – 04 Jul 2026 | Write Mode testing, polish, release | Write Mode ("My Drafts") shipped end-to-end |

---

## 3. Daily Progress Log

| Date | Day | Work Assigned | Work Completed | Progress (%) | Skills Learned | Problems Faced | Hours Worked | Next Day Plan |
|------|-----|----------------|-----------------|:---:|------------------|-------------------|:---:|-----------------|
| 04-May-2026 | Monday | Improve answer quality on complex questions | Implemented multi-hop retrieval so the system combines evidence from multiple chunks/sections instead of a single retrieval pass. | 5% | Multi-hop retrieval design | Single-hop retrieval missed context spread across sections | 5 | Add query planning and reasoning layer |
| 05-May-2026 | Tuesday | Add a reasoning layer on top of retrieval | Added query planning, chain-of-thought reasoning, and evidence grading to the answer pipeline; tested with multiple API keys to avoid single-key rate limits. | 9% | Query planning, CoT prompting, evidence grading, key rotation | Single API key hit rate limits during testing | 5 | Improve retrieval boosting using extracted concepts |
| 06-May-2026 | Wednesday | Improve retrieval relevance | Used extracted key_concepts to boost retrieval relevance and made paper-name resolution dynamic in query expansion. | 12% | Retrieval re-ranking/boosting, dynamic query expansion | Generic queries weren't resolving to the correct paper | 5 | Continue refining evidence-quality scoring |
| 07-May-2026 | Thursday | Refine evidence-quality scoring | Reviewed evidence-grading output on sample papers and tightened the scoring heuristic for weak/irrelevant evidence. | 14% | Evidence scoring heuristics | Some low-quality evidence was still passing grading | 4 | Continue evidence-quality work; start UI touch-ups |
| 08-May-2026 | Friday | Continue evidence-quality work + UI | Continued refining evidence-grading thresholds and started minor UI clean-up around answer display. | 16% | Threshold tuning, frontend polish | Balancing grading strictness without discarding valid evidence | 5 | Finish evidence-quality logic and UI pass |
| 09-May-2026 | Saturday | Light testing day | Ran manual QA on recent multi-hop and evidence-grading changes across several test papers. | 17% | Manual QA / regression testing | A couple of edge-case queries still returned weak evidence | 2 | Finalize evidence-quality logic |
| 10-May-2026 | Sunday | Finalize evidence-quality logic + UI | Finalized the evidence-quality logic and shipped a round of UI improvements. | 20% | Evidence-quality finalization, UI iteration | Minor implementation issues | 5 | Add table extraction and multi-paper comparison |
| 11-May-2026 | Monday | Table extraction + multi-paper support | Added table extraction, structured logging, multi-paper comparison, and paper deletion. | 24% | Table parsing, structured logging, cross-paper comparison logic | Tables were often misread as prose text | 6 | Polish multi-paper compare UI |
| 12-May-2026 | Tuesday | Polish multi-paper features | Polished the multi-paper comparison UI and verified table-extraction output on additional test PDFs. | 26% | UI polish, extraction QA | A few tables with merged cells still parsed incorrectly | 4 | Start deployment planning |
| 13-May-2026 | Wednesday | Plan deployment | Planned the deployment strategy for backend and frontend, evaluating hosting options. | 28% | Deployment architecture planning | Choosing a free/low-cost host that supports the vector DB | 4 | Continue deployment prep |
| 14-May-2026 | Thursday | Deployment prep | Drafted the environment-variable and config structure needed for a future deployment. | 30% | Environment/config management | Minor implementation issues | 4 | Continue deployment prep |
| 15-May-2026 | Friday | Deployment prep continued | Reviewed hosting constraints (free-tier limits) against the project's storage/compute needs. | 32% | Infra cost/constraint analysis | Free-tier compute limits vs. vector DB memory needs | 4 | Wrap up deployment planning notes |
| 16-May-2026 | Saturday | Light research day | Researched free-tier hosting options for later use during actual deployment. | 33% | Platform research | Minor implementation issues | 2 | Shift focus to evaluation instrumentation |
| 17-May-2026 | Sunday | Plan eval instrumentation | Sketched out the ablation flags and logging needed to later benchmark the pipeline objectively. | 35% | Evaluation/instrumentation planning | Minor implementation issues | 3 | Implement ablation flags and logging |
| 18-May-2026 | Monday | Add eval instrumentation | Added ablation flags, LLM usage stats, JSONL logging, and atomic storage writes; logged progress. | 38% | Ablation instrumentation, structured logging, atomic file writes | Needed reproducible on/off pipeline stages for later evaluation | 5 | Start the live research discovery feature |
| 19-May-2026 | Tuesday | Design live discovery feature | Designed the flow for searching external literature (arXiv/Semantic Scholar) from within the app. | 40% | Feature/API design | Minor implementation issues | 4 | Integrate arXiv search API |
| 20-May-2026 | Wednesday | Integrate arXiv/S2 APIs | Wired up arXiv and Semantic Scholar search APIs and mapped their responses into a common paper-result format. | 42% | External API integration | Inconsistent metadata fields between the two APIs | 5 | Wire discovered papers into the ingestion/RAG pipeline |
| 21-May-2026 | Thursday | PDF import from discovery | Built the PDF-import path so a discovered paper can be pulled straight into the existing RAG pipeline. | 44% | Pipeline integration | Some discovered PDFs failed to download directly | 5 | Add an off-topic query guard |
| 22-May-2026 | Friday | Off-topic guard design | Designed a pre-flight LLM check to short-circuit non-research queries before they hit the full pipeline. | 46% | Guardrail/pre-flight check design | Minor implementation issues | 4 | Finish and test live discovery + guard together |
| 23-May-2026 | Saturday | Ship live discovery + guard | Shipped live research discovery (arXiv + S2 search, PDF import, RAG integration) along with the off-topic guard, plus a follow-up improvement pass. | 50% | End-to-end feature shipping, pre-flight filtering | Off-topic queries were wasting LLM calls before the guard existed | 6 | Test discovery/guard edge cases |
| 24-May-2026 | Sunday | Test discovery + guard | Ran manual tests on the off-topic guard and discovery flow across a range of queries. | 51% | Manual QA | A few borderline queries were misclassified as off-topic | 3 | Refine guard thresholds |
| 25-May-2026 | Monday | Refine guard + discovery | Refined off-topic guard thresholds and cleaned up discovery result formatting. | 52% | Threshold tuning, UI cleanup | Minor implementation issues | 4 | Prep the repo for the evaluation harness |
| 26-May-2026 | Tuesday | Repo housekeeping | Updated .gitignore ahead of the evaluation harness work (excluding datasets/artifacts). | 53% | Repo hygiene | Minor implementation issues | 2 | Start building the QASPER eval harness |
| 27-May-2026 | Wednesday | Research QASPER benchmark | Studied the QASPER dataset format and planned the loader/adapter needed to feed it through the existing pipeline. | 55% | Benchmark dataset research | Minor implementation issues | 4 | Build the QASPER loader |
| 28-May-2026 | Thursday | Build QASPER loader | Implemented the initial QASPER loader and ingest adapter. | 57% | Dataset loader/adapter design | Dataset field names didn't map cleanly onto the existing paper schema | 5 | Add a smoke test for the loader |
| 29-May-2026 | Friday | Smoke-test the loader | Wrote a smoke test to confirm the loader/adapter produced valid pipeline input. | 59% | Smoke testing | Minor implementation issues | 4 | Design the scoring/metrics harness |
| 30-May-2026 | Saturday | Design scoring harness | Sketched out the metrics (token-F1, judge-based scoring) needed to grade generated answers against QASPER references. | 60% | Evaluation-metric design | Minor implementation issues | 3 | Build the full scoring + ablation harness |
| 31-May-2026 | Sunday | Build full eval harness | Built the QASPER eval harness end-to-end in one push — loader/adapter/smoke test, scoring (metrics + LLM-judge + runner), and an ablation matrix — and fixed a Cerebras model id, a JSON-truncation bug, and reordered LLM providers (Groq-first) for faster failover. | 66% | Benchmark harness design, LLM-as-judge, ablation methodology, provider failover tuning | Cerebras returned model-not-found errors; verbose providers truncated JSON output | 8 | Analyze the first ablation results |
| 01-Jun-2026 | Monday | Fix generator/grader based on eval results | Stripped leaked prompt scaffolding out of generator output; added majority-vote judging and a pinnable generation model for the grader study. | 68% | Prompt-leakage debugging, majority-vote aggregation | Judge model output contained prompt artifacts, skewing scores | 5 | Continue fixing grader edge cases |
| 02-Jun-2026 | Tuesday | Fix grader edge cases + ship study tools | Fixed the grader (preserving correct negative answers, skipping false pipeline-degradation signals) and shipped glossary, audience mode, recommendations, and flashcard export. | 70% | Grader-correctness debugging, feature shipping | Grader was flagging valid negative answers as errors | 6 | Add source traceability to answers |
| 03-Jun-2026 | Wednesday | Source traceability + writing assistant | Added clickable source chips linking to the exact PDF page, an inline PDF preview panel, fixed S2 citation-merge failures, and shipped a writing assistant (academic/plain/concise modes). | 72% | PDF-page linking, in-app PDF viewing, tone-controlled generation | Citation data from S2 sometimes conflicted when merging | 6 | Add notes and session tools |
| 04-Jun-2026 | Thursday | Session tooling | Added a notes panel, session export, empty-state polish, side-by-side split view, escape-to-close-all-panels, and ⌘F conversation search. | 74% | Multi-panel state management, keyboard-shortcut UX | Panel state conflicts when multiple panels were open at once | 6 | Begin repo cleanup and CI setup |
| 05-Jun-2026 | Friday | Pre-cleanup polish | Fixed miscellaneous UX bugs and ran a full manual pass over the app ahead of the repo cleanup. | 75% | Bug triage, manual regression testing | A few rough edges found in panel interactions | 5 | Restructure repo and add CI/CD |
| 06-Jun-2026 | Saturday | Repo cleanup + CI/CD | Fixed grader neutrality logic; restructured the repo and stopped tracking runtime state; rewrote the README as a landing page and added an MIT license; added an offline metrics test suite; set up GitHub Actions (pytest + lint + frontend build); fixed all ESLint errors/warnings; bumped Node to 22. | 79% | CI/CD pipeline setup, ESLint, repo restructuring, licensing | Node/tooling version mismatches were breaking the CI build | 7 | Harden ingestion and add more tests |
| 07-Jun-2026 | Sunday | Harden ingestion + testing | Added lenient LLM-JSON parsing and provider hardening; added ruff config and fixed lint violations; added offline unit tests for the chunker and off-topic guard; wired ruff into CI; documented system-performance metrics; overhauled the upload page. | 81% | Lenient JSON parsing, ruff linting, unit-testing discipline | Strict JSON parsing was breaking on minor LLM formatting drift | 6 | Polish chat UX |
| 08-Jun-2026 | Monday | Chat UX fix | Fixed copied-answer text cleanup and surfaced real trace timings in the chat UI. | 82% | UX polish, performance-trace surfacing | Copy-paste was including hidden formatting artifacts | 4 | Start writing production documentation |
| 09-Jun-2026 | Tuesday | Draft production docs outline | Outlined the production documentation covering architecture, data flow, and deployment. | 83% | Technical writing/planning | Minor implementation issues | 3 | Document architecture in detail |
| 10-Jun-2026 | Wednesday | Document architecture | Wrote the architecture section of the production documentation (ingestion → retrieval → generation flow). | 84% | Technical documentation | Keeping docs in sync with a fast-moving codebase | 4 | Document deployment/ops |
| 11-Jun-2026 | Thursday | Document deployment/ops | Documented deployment and operational considerations (env vars, storage, provider keys). | 85% | Ops documentation | Minor implementation issues | 4 | Finalize and publish production documentation |
| 12-Jun-2026 | Friday | Finalize production documentation | Finalized and committed the production documentation covering the system's architecture and operational setup. | 86% | Technical documentation | Minor implementation issues | 4 | Begin planning multi-tenancy (auth, per-user data) |
| 13-Jun-2026 | Saturday | Research auth providers | Researched authentication providers (Clerk) for adding real user accounts to the app. | 87% | Auth-provider evaluation | Minor implementation issues | 3 | Design multi-tenant data model |
| 14-Jun-2026 | Sunday | Design multi-tenant data model | Sketched the per-user data model needed to isolate papers, chats, and quotas across users. | 88% | Multi-tenant schema design | Minor implementation issues | 4 | Prototype Clerk auth integration |
| 15-Jun-2026 | Monday | Prototype auth integration | Built an initial prototype wiring Clerk authentication into the backend and frontend. | 89% | Clerk auth integration | Session/token handling across frontend and backend needed careful wiring | 5 | Plan storage migration (Neon/R2) |
| 16-Jun-2026 | Tuesday | Plan storage migration | Planned the migration from local storage to managed Postgres (Neon) and object storage (R2). | 90% | Managed Postgres/object-storage planning | Minor implementation issues | 4 | Implement multi-tenancy + billing + hardening |
| 17-Jun-2026 | Wednesday | Ship multi-tenancy, billing, and hardening | Added Clerk authentication with per-user scoping and Neon/R2 storage migration; added per-request LLM cost tracking and per-user quotas; added Stripe Checkout, customer portal, and subscription webhooks; dockerized the backend with CORS lockdown, rate limiting, and upload validation; added Sentry error tracking and PostHog analytics. | 93% | Auth, managed Postgres, object storage, Stripe billing, Docker, rate limiting, observability tooling | Coordinating five infra changes at once without breaking the running app | 8 | Finish the launch checklist |
| 18-Jun-2026 | Thursday | Finish launch checklist | Added a public landing page, ToS/Privacy pages, and preloaded sample papers; fixed a chunk-ID collision crash; added structured error logging with request tracing; added per-paper locking and thread-safe ChromaDB access for concurrent users. | 95% | Concurrency control, thread-safe vector-store access, request tracing | Chunk-ID collisions and race conditions surfaced under concurrent access | 6 | Run end-to-end multi-tenant testing |
| 19-Jun-2026 | Friday | End-to-end multi-tenant testing | Ran end-to-end tests across multiple simulated users (auth, quotas, billing, storage isolation). | 96% | End-to-end/system testing | A quota edge case allowed one extra request past the limit | 5 | Continue hardening under concurrent load |
| 20-Jun-2026 | Saturday | Concurrency/load testing | Stress-tested concurrent uploads and queries to confirm per-paper locking held up under load. | 96% | Load/concurrency testing | Minor implementation issues | 3 | Start building the automated paper-auditing feature |
| 21-Jun-2026 | Sunday | Build paper-auditing engines | Built the automated reviewer/weakness audit with a dedicated analysis quota, and the claim-vs-evidence audit that grounds claims to evidence and flags overclaims/methodological gaps. | 97% | Grounding claims to evidence, rubric-based audit design (OK/WEAK/MISSING) | Avoiding false MISSING verdicts without being too lenient | 7 | Extend the audit engines into a Write Mode for drafts |
| 22-Jun-2026 | Monday | Design Write Mode | Designed Write Mode — a new tab where authors upload an in-progress draft and run the same audit engines pre-submission. | 97% | Product/feature design | Deciding how much to reuse vs. adapt from the published-paper audit engines | 5 | Adapt the claim-audit engine for drafts |
| 23-Jun-2026 | Tuesday | Adapt claim audit for drafts | Adapted the claim-vs-evidence audit engine to work on unpublished drafts, which lack a fixed reference list to check against. | 97% | Engine reuse/adaptation | Drafts often lack the reference list published papers have | 5 | Adapt the reviewer audit for drafts |
| 24-Jun-2026 | Wednesday | Adapt reviewer audit for drafts | Adapted the reviewer/weakness audit engine for draft papers, reusing the OK/WEAK/MISSING rubric. | 98% | Rubric reuse across product surfaces | Minor implementation issues | 5 | Build the "My Drafts" storage/upload flow |
| 25-Jun-2026 | Thursday | Build "My Drafts" flow | Built the upload and storage flow for user drafts, kept separate from the main paper library. | 98% | Draft-vs-published data separation | Minor implementation issues | 5 | Build the Write Mode UI panel |
| 26-Jun-2026 | Friday | Build Write Mode UI | Built the Write Mode UI panel tying together draft upload and the reused audit engines. | 98% | Frontend panel composition | Minor implementation issues | 5 | Wire audit results into the Write Mode UI |
| 27-Jun-2026 | Saturday | Wire audit results into UI | Connected claim/reviewer audit output into the Write Mode results view. | 98% | Frontend-backend wiring | Minor implementation issues | 3 | Test Write Mode on real in-progress drafts |
| 28-Jun-2026 | Sunday | Test Write Mode on real drafts | Tested Write Mode end-to-end on real in-progress drafts and compared audit output quality against the published-paper flow. | 99% | End-to-end testing, output-quality comparison | Audit verdicts were occasionally too harsh on incomplete draft sections | 5 | Tune audit sensitivity for drafts |
| 29-Jun-2026 | Monday | Tune audit sensitivity | Tuned the audit engines to be more conservative on incomplete draft sections instead of flagging them outright. | 99% | Verdict-sensitivity tuning | Minor implementation issues | 5 | Continue refining Write Mode UX |
| 30-Jun-2026 | Tuesday | Refine Write Mode UX | Refined the Write Mode UX with clearer progress indicators and audit-status labels for drafts. | 99% | UX refinement | Minor implementation issues | 4 | Regression-test the app alongside Write Mode |
| 01-Jul-2026 | Wednesday | Regression testing | Ran a full regression pass across research mode and the new Write Mode to confirm nothing broke. | 99% | Regression testing | A minor state leak between research-mode and write-mode panels | 5 | Fix the panel state leak and prep for release |
| 02-Jul-2026 | Thursday | Fix state leak + final polish | Fixed the panel state leak between modes and did final polish ahead of shipping Write Mode. | 99% | State-isolation debugging | Minor implementation issues | 5 | Ship Write Mode |
| 03-Jul-2026 | Friday | Ship Write Mode | Shipped Write Mode — pre-submission draft audits ("My Drafts") reusing the claim/reviewer audit engines. | 100% | Feature release, cross-engine reuse in production | Minor implementation issues | 6 | Review the period's work and plan next steps |
| 04-Jul-2026 | Saturday | Wrap-up | Reviewed the period's work, verified Write Mode end-to-end in the browser, and updated internal notes for the next phase of work. | 100% | Self-review, browser verification | — | 2 | Continue with citation-gap and originality checks (next phase) |

---

## 4. Skills Acquired

- **Retrieval & RAG:** Multi-hop retrieval, query planning, chain-of-thought reasoning, evidence grading, retrieval re-ranking/boosting, table extraction, multi-paper comparison.
- **External Integration:** Live literature discovery via arXiv and Semantic Scholar, PDF import into an existing RAG pipeline, pre-flight off-topic query guarding.
- **Evaluation Engineering:** Building a benchmark harness from scratch against QASPER — loader/adapter design, token-F1 and LLM-as-judge metrics, majority-vote aggregation, and an ablation matrix to isolate what each pipeline stage contributes.
- **LLM Engineering:** Multi-provider integration and failover (Groq, Cerebras, Gemini, Mistral), lenient JSON parsing, prompt-leakage debugging, grader-correctness debugging.
- **Full-Stack Product Development:** Source-linked PDF viewing, multi-panel UX (notes, split view, session export), keyboard-shortcut interactions, a tone-controlled writing assistant.
- **Engineering Discipline:** Repo restructuring, CI/CD (GitHub Actions), linting (ESLint, ruff), automated test suites, production documentation.
- **Infrastructure & SaaS Fundamentals:** Clerk authentication, managed Postgres (Neon) and object storage (R2), Stripe billing integration, Docker containerization, CORS/rate-limiting/upload-validation hardening, Sentry error tracking, PostHog analytics, thread-safe concurrent access design.
- **Domain-Specific Auditing:** Designing rubric-based automated audits (claim-vs-evidence, reviewer/weakness) that are conservative enough to avoid false positives, and adapting them to work on unpublished drafts for a new "Write Mode" product surface.

---

## 5. Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| Single-hop retrieval missed context spread across multiple sections | Implemented multi-hop retrieval with query planning and chain-of-thought reasoning |
| Off-topic queries were wasting LLM calls | Added a pre-flight LLM check that short-circuits non-research queries before the full pipeline runs |
| No objective way to know if pipeline changes actually helped | Built a standalone QASPER-based eval harness with metrics, LLM-judge scoring, and an ablation matrix |
| LLM provider outages/rate limits and JSON truncation threatened reliability | Reordered providers for faster failover and fixed max_tokens/model-id issues causing truncation and errors |
| Concurrent users caused chunk-ID collisions and race conditions | Added per-paper locking and thread-safe ChromaDB access |
| Turning a single-user prototype into a real multi-tenant product | Sequenced the work as a launch checklist: auth/storage → LLM economics → billing → deployment hardening → observability |
| Audit engines built for published papers were too harsh on incomplete drafts | Tuned verdict sensitivity to be more conservative when adapting the engines for Write Mode |

---

## 6. Conclusion

Over this nine-week period, PaperMind's single-paper Q&A pipeline was upgraded into a full reasoning system (multi-hop retrieval, query planning, evidence grading), extended with live literature discovery, and proven objectively through a self-built QASPER evaluation harness rather than relying on impressions. The middle of the period focused on turning the prototype into a real product — authentication, billing, deployment hardening, and observability — while also layering on a complete study-tools UX. The final stretch extended the product beyond Q&A into automated paper auditing, culminating in Write Mode: a way for authors to run the same claim and reviewer audits on their own drafts before submission. By 04 July 2026, Write Mode was shipped and verified end-to-end, with citation-gap and originality checks scoped as the next phase of work.

---

**Mentor's Remarks:**

_____________________________________________________________

**Mentor's Signature:** ______________________ **Date:** ______________

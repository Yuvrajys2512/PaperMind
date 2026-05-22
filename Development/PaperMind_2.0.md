# PaperMind 2.0 — Expansion Scope

> Expanding PaperMind from a "read research papers" tool into a full research writing and discovery platform.
> Target: portfolio project + real users. Build order: Phase 1 → 2 → 3.

---

## Vision

PaperMind 1.0 helps users understand papers they already have.
PaperMind 2.0 helps users **find**, **understand**, and **write** research — end to end.

---

## Phase 1: Live Research Discovery (build first — highest value)

Multi-source paper search that feeds directly into the existing RAG pipeline. This is the killer feature — it compounds on everything already built.

### Flow
User types a topic → search across sources in parallel → results displayed → user selects papers → app fetches + chunks + embeds on the fly → user chats across them exactly like uploaded PDFs.

### Free APIs to aggregate

| Source | Coverage | Notes |
|--------|----------|-------|
| OpenAlex | Everything — 250M+ works | Best general coverage, free, no API key needed |
| Semantic Scholar | CS, Medicine, Biology, etc. | Free tier, abstracts + citation graph |
| arXiv | CS, Physics, Math, Biology, Finance, Economics | Free, full PDFs available |
| PubMed / NCBI | Biomedical | Free, E-utilities API |
| CORE | Open access only | Free tier, full text where available |

### Architecture

- `services/search_aggregator.py` — fans out queries to all sources in parallel, deduplicates by DOI/title, ranks by relevance score
- `services/paper_fetcher.py` — downloads PDFs for open-access papers, extracts text on the fly
- Reuses existing chunking + ChromaDB embedding pipeline
- Results stored in a temporary "search session" collection, separate from the user's saved library

### Key decisions
- Search results are embedded into a **temporary collection** (not polluting the user's library) until they explicitly save a paper
- Deduplication by DOI prevents the same paper appearing from multiple sources
- Fallback to abstract-only RAG when full PDF is not open-access

---

## Phase 2: Writing Assistant

### Feature 1: Humanize / Rewrite
**Approach:** LLM prompt — already have the infrastructure.
**Modes:** academic tone, plain language, concise summary.
**Cost:** negligible, uses existing LLM layer.
**Free forever** — no third-party API needed.

### Feature 2: AI Detection
**Approach:** Third-party APIs (probabilistic, no self-hosted alternative is reliable).

| Service | Free Tier | Notes |
|---------|-----------|-------|
| GPTZero | 10,000 words/month | Most widely known |
| Sapling AI | Limited free tier | Good accuracy |
| Winston AI | Free trial | Strong on academic text |

**Integration:** single `services/ai_detector.py` abstraction, swappable backend.
**Note:** AI detection is probabilistic by nature — surface confidence scores, not binary verdicts.

### Feature 3: Plagiarism Check
**Free approach:** cosine similarity against the user's indexed corpus + OpenAlex/Semantic Scholar for broader comparison. Not as comprehensive as commercial tools but meaningful for portfolio use.

**Paid later:** Copyleaks API or iThenticate once the product goes live.

**Architecture:** `services/plag_checker.py` — runs similarity search against all indexed papers, returns matched passages with similarity scores and source citations.

---

## Phase 3: Research Workspace

Full end-to-end research environment once Phase 1 and 2 are solid.

- **Personal library:** save papers from live search into the user's permanent collection
- **Draft editor:** write with inline citations pulled from saved papers, AI-assisted paragraph expansion
- **Citation formatter:** output in APA, MLA, Chicago, BibTeX
- **Research timeline:** track what papers you've read, what you've cited, what's in queue

---

## Build Order & Rationale

| Phase | Why first |
|-------|-----------|
| 1 — Live Search | Builds directly on existing RAG. Highest user value. Clean extension. |
| 2a — Humanize | Trivial to add (LLM prompt). Easy win, makes the platform feel complete. |
| 2b — AI Detection | Third-party API integration, limited scope. |
| 2c — Plag Check | Most complex, needs broader corpus. Do after detection is live. |
| 3 — Workspace | Requires Phase 1 library feature as prerequisite. |

---

## API Strategy

**Development/testing:** all free tiers. OpenAlex, Semantic Scholar, arXiv, PubMed are free indefinitely. GPTZero/Sapling free tiers cover testing.

**Production:** upgrade to paid tiers for AI detection and plagiarism check once the product is ready for real users. Existing LLM costs (Gemini) scale with usage.

---

## What We're Not Building (and Why)

- **Our own plagiarism database** — needs petabytes of indexed content. Not feasible. Use APIs.
- **AI detection from scratch** — state-of-the-art models (GPT-4, Gemini) have collapsed perplexity distributions, making statistical detection near-impossible without a massive training set.
- **Humanize as a standalone SaaS** — this is a support feature for research writing, not the core product.

---

## Notes on the Humanize / AI-Removal Feature

This feature helps users rewrite AI-assisted drafts into a more natural academic voice. It is academically gray in contexts where AI use is prohibited. Position it clearly as a **writing improvement tool**, not an "AI bypass" tool. Surface this distinction in the UI copy.

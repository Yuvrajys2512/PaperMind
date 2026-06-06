# PaperMind

[![CI](https://github.com/Yuvrajys2512/PaperMind/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuvrajys2512/PaperMind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An agentic research assistant that reasons about *whether to retrieve, what to retrieve, and whether to trust what it found* — before it answers.**

PaperMind goes beyond conventional RAG. It plans each query, retrieves with a hybrid dense + lexical search, reranks, and then **grades its own answer sentence-by-sentence against the source**, stripping any claim it can't support. Every answer ships with per-sentence evidence indicators, cited sources, a confidence score, and an honest "I can't answer that from this paper" when the paper doesn't contain the answer.

It's also a **measured** system: the retrieval and grading choices are backed by an ablation study on the [QASPER](https://allenai.org/data/qasper) benchmark, not vibes. See [Evaluation](#evaluation).

<!-- TODO: replace with your live demo URL once deployed (see DEPLOYMENT.md) -->
🔗 **Live demo:** _coming soon_ &nbsp;·&nbsp; <!-- TODO: drop a chat GIF here --> 🎥 **Demo video:** _coming soon_

---

## Why this is more than a RAG wrapper

| Most RAG demos | PaperMind |
|---|---|
| Embed → retrieve top-k → stuff into prompt → answer | **Plans** the query (decomposition, answer type, structure) before retrieving |
| Dense retrieval only | **Hybrid** dense (ChromaDB) + lexical (BM25) with cross-encoder **reranking** |
| Trusts the LLM's output | **Grades every sentence** against the retrieved evidence and removes unsupported claims |
| Fails silently or hallucinates off-topic | **Off-topic guard** + **out-of-domain detection** → refuses instead of confabulating |
| One model, one point of failure | **Multi-provider failover** (Groq → Gemini → Mistral → Cerebras) with escalating backoff |
| "Looks good to me" | **Ablation study on QASPER** quantifying what each component actually buys |

---

## Architecture

```
                            ┌──────────────────────────┐
                            │   React + Vite frontend   │
                            │  chat · discover · library│
                            │  upload · rewrite · split │
                            └────────────┬──────────────┘
                                         │  REST + SSE streaming
                            ┌────────────▼──────────────┐
                            │   FastAPI backend (api/)   │
                            └────────────┬──────────────┘
                                         │
        ┌────────────────────────────────▼─────────────────────────────────┐
        │                  Query pipeline (ingestion/)                      │
        │                                                                   │
        │   1. Off-topic guard ──────────── refuse non-paper questions      │
        │   2. Query planner ────────────── decompose · answer type/shape   │
        │   3. Hybrid retrieve ──────────── BM25 + dense (ChromaDB)          │
        │   4. Rerank ───────────────────── cross-encoder ms-marco-MiniLM    │
        │   5. Generate ─────────────────── CoT reasoning → structured answer│
        │   6. Evidence grade ───────────── drop UNSUPPORTED sentences  ★    │
        │   7. Evaluate ─────────────────── faithfulness + answer-relevancy  │
        │   8. Retry (if low confidence) ── diagnose failure, re-route       │
        └────────────────────────────────┬─────────────────────────────────┘
                                         │
                            ┌────────────▼──────────────┐
                            │  Unified LLM client        │
                            │  Groq · Gemini · Mistral · │
                            │  Cerebras  (auto-failover) │
                            └────────────────────────────┘
```

**Ingestion side** (one-time per paper): PDF parsing (`pdfplumber`) → section detection → table extraction → chunking (`tiktoken`) → embedding (`all-MiniLM-L6-v2`) → ChromaDB.

---

## Features

- **Grounded Q&A** over uploaded papers, with cited sources (section + page) and per-sentence evidence confidence.
- **Self-verifying answers** — the evidence grader removes claims the source doesn't support before you ever see them.
- **Honest refusals** — off-topic and out-of-domain questions are declined instead of hallucinated.
- **Multi-paper comparison** — ask one question across two papers, get a structured side-by-side.
- **Paper discovery** — search arXiv + Semantic Scholar and ingest directly.
- **Glossary & recommendations** generated per paper.
- **Rewrite tool** — rephrase passages at different reading levels.
- **Polished chat UX** — streaming answers (SSE), split-view parallel streams, notes panel, session export, and ⌘F in-conversation search.

---

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS, react-markdown |
| Backend | FastAPI, Uvicorn |
| Retrieval | ChromaDB (dense), rank-bm25 (lexical), sentence-transformers cross-encoder (rerank) |
| Embeddings | `all-MiniLM-L6-v2` |
| Parsing | pdfplumber, tiktoken |
| LLMs | Groq, Gemini, Mistral, Cerebras — via one OpenAI-compatible client with failover |
| Eval | Custom harness on QASPER; LLM-judge with N=3 majority vote |

---

## Evaluation

The headline research question PaperMind is built to answer:

> **Does self-grading the generated answer improve faithfulness and/or judged correctness without dropping Answer-F1 — and at what latency cost?**

The harness (`eval/`) loads QASPER, runs PaperMind in configurable ablation modes (full / no-grader / no-rerank / no-hyde / no-retry), and scores each answer with token-F1, a faithfulness check, and an LLM judge using **N=3 majority vote** to control judge variance.

**Honest current finding (paired design, n=14 answerable questions):**

| | Mean judge score |
|---|---|
| Pre-grader answer | 0.607 |
| Post-grader (cleaned) answer | **0.607** |

The evidence grader's **net effect on judged correctness is ~zero** (3 helped / 3 hurt / 8 neutral) at this scale — but it does **lift faithfulness** in the full ablation run, and the cases where it *hurts* follow a specific, explainable pattern (it strips correct *negative* answers, which have no direct textual support). In other words: grading trades a little correctness for measurably higher faithfulness, rather than being a free win.

I'm reporting that straight because **measuring honestly — including null results — is the point.** Scaling n and tuning the grader threshold are the open next steps. Full methodology and run-by-run logs live in [`research/`](research/).

> ⚠️ Numbers are preliminary (small n, free-tier provider variance). Treat as directional.

---

## Getting started

### Backend

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Create a `.env` with at least one provider key (more = better failover):

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
CEREBRAS_API_KEY=...
```

Run it:

```bash
uvicorn api.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, upload a PDF, and ask away.

---

## API surface (selected)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/upload` | Ingest a PDF (async; poll `/status/{id}`) |
| `GET` | `/papers` | List ingested papers |
| `POST` | `/query` | Ask a question about a paper |
| `POST` | `/query/stream` | Same, streamed token-by-token (SSE) |
| `GET` | `/papers/{id}/glossary` | Generated glossary |
| `GET` | `/papers/{id}/recommendations` | Related-work suggestions |
| `POST` | `/rewrite` | Rephrase a passage |

---

## Deployment

A complete, beginner-friendly deployment guide (Render + Vercel, Dockerfile, persistent disk, CORS hardening, key rotation) is in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Project layout

```
api/          FastAPI app, routes, logging, storage
ingestion/    The query pipeline + retrieval/grading/LLM modules
discovery/    arXiv + Semantic Scholar search and PDF fetching
eval/         QASPER evaluation harness (loader, metrics, judge, ablations)
frontend/     React + Vite client
research/     Methodology write-ups and ablation results
```

---

## Roadmap

- [ ] Public live demo (see DEPLOYMENT.md)
- [ ] Scale the QASPER eval past n=14 so the faithfulness/correctness tradeoff clears 1 SE
- [ ] Tune the grader threshold to stop stripping correct negations
- [x] CI (tests + lint + frontend build on every push)
- [ ] Externalize storage (Postgres / S3 / hosted vector DB) for multi-instance scaling

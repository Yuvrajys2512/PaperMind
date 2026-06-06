# PaperMind — Complete Mastery Document

> One-stop reference to defend every line, every decision, every concept in this project. After reading this you should be able to answer any structural, architectural, theoretical, or practical question without notes.

---

## 0. Elevator Pitch (memorize this verbatim)

**PaperMind is an agentic, evidence-grounded Q&A system for academic research papers. Unlike conventional RAG which is "retrieve then generate," PaperMind reasons about *whether to retrieve, what to retrieve, and whether to trust what it generated* — before showing it to the user.** It runs a 9-stage pipeline that combines hybrid retrieval (BM25 + dense vectors fused with Reciprocal Rank Fusion), cross-encoder reranking, chain-of-thought answer generation, per-sentence evidence grading that physically removes unsupported sentences, and a self-evaluation + retry engine that diagnoses whether a failure was a retrieval or a generation problem and retries with a different strategy. The whole pipeline is provider-agnostic — it rotates across Gemini, Cerebras, Mistral, and Groq through one OpenAI-compatible client so no single LLM outage can crash it.

If asked "what makes this different from LangChain RAG?": **Evidence grading, agentic retry, query planning, and provider failover. Those four are the non-table-stakes parts.**

---

## 1. Problem & Motivation

### 1.1 The problem
Research papers are long, dense, and structured. Naive ChatGPT-style Q&A on them fails in three ways:
1. **Hallucination** — the LLM invents facts that look plausible but don't appear in the paper.
2. **Shallow retrieval** — semantic search alone misses keyword-specific questions (e.g., "What does Table 3 show?"); keyword search alone misses paraphrased queries.
3. **One-shot failure** — if the first attempt fails, the user sees a bad answer with no recovery.

### 1.2 The thesis
A pipeline that:
- *Plans* before retrieving (knows what kind of answer is needed),
- *Hybridizes* retrieval (lexical + semantic),
- *Grades* its own output sentence-by-sentence,
- *Retries* with diagnosis when confidence is low,

…produces answers that are demonstrably more grounded than a single-pass RAG. This is the project's hypothesis and what every component exists to serve.

### 1.3 What "agentic" means here (defend this carefully)
**Agentic = the system makes decisions based on its own intermediate state**. Specifically:
- It **decides** the retrieval strategy (single-pass vs multi-hop) from the query plan's `complexity` field.
- It **decides** whether to retry from the confidence score.
- It **decides** *why* to retry from the diagnosis (`retrieval` vs `generation`).
- It **decides** to remove sentences from its own output via the evidence grader.

It is *not* full ReAct-style tool use — it is a directed control flow with LLM-driven decisions at four key branch points. Be honest about this.

---

## 2. The Full Tech Stack — every library, why it was chosen

### 2.1 Backend (Python)
| Library | Version | Used for | Why this and not alternatives |
|---|---|---|---|
| **pdfplumber** | ≥0.11 | PDF parsing — character-level extraction with x/y coordinates, font size, upright flag | PyPDF2/PyMuPDF give you the text but not the per-character metadata needed to detect two-column layouts and filter rotated watermarks. pdfplumber exposes `page.chars` (dict per character) which is essential for the custom column detector. |
| **tiktoken** | ≥0.12 | Tokenization — `cl100k_base` encoding | Token counts must match the embedding model's tokenization. Counting words is wrong — "Uncharacteristically" is 1 word but 4 tokens. `cl100k_base` is GPT-4/Ada's encoding and matches all modern OpenAI-compatible models. |
| **chromadb** | ≥1.5 | Vector store (PersistentClient, on-disk) | Lightweight, embedded, file-based — no Docker, no separate server. Good enough for single-host. Alternatives: FAISS (no metadata filtering), Qdrant (heavier), Pinecone (cloud-only, costs money). |
| **sentence-transformers** | ≥5.0 | Embedding (`all-MiniLM-L6-v2`) + reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | MiniLM-L6 is 22M params, 384-dim, runs on CPU at ~1000 sentences/sec — small enough for a laptop, accurate enough for academic retrieval. CrossEncoder is the standard reranker for MS MARCO benchmarks. |
| **rank-bm25** | ≥0.2 | BM25 lexical retrieval (`BM25Okapi`) | Pure-Python, no native deps, drop-in implementation of the Okapi BM25 ranking function. |
| **fastapi** | ≥0.130 | HTTP API framework | Async-native, Pydantic-validated request bodies, OpenAPI docs auto-generated, modern. Flask would have required `flask-pydantic` and a sync→async shim. |
| **uvicorn[standard]** | ≥0.46 | ASGI server | Bundled standard extras give httptools + uvloop for performance. |
| **python-multipart** | ≥0.0.20 | Required by FastAPI for multipart/form-data (file upload) | Without it, `UploadFile` raises `RuntimeError`. |
| **openai** | ≥2.0 | LLM client — used as a *protocol*, not a vendor | The OpenAI SDK accepts a `base_url` parameter, so we point it at Gemini, Cerebras, Mistral, or Groq endpoints. Single SDK, single auth flow, four providers. |
| **python-dotenv** | ≥1.0 | Load `.env` into `os.environ` | Standard. |
| **numpy** | ≥2.0 | Cosine similarity math in the evaluator | Used for `dot()` and `linalg.norm()`. Not used elsewhere — the evaluator is the only consumer. |

### 2.2 Frontend
| Library | Version | Used for |
|---|---|---|
| **React** | 19.2 | UI framework |
| **Vite** | 8 | Dev server + build tool. `vite.config.js` proxies `/api` → `http://localhost:8000` so the React app and FastAPI run on different ports without CORS issues |
| **Tailwind CSS** | 4 (via `@tailwindcss/vite`) | Utility-first styling. Custom theme in `index.css` defines fonts (Inter, Space Grotesk, JetBrains Mono) and component classes (`.pm-card`, `.cosmic-orb-cyan`, etc.) |
| **react-markdown** | 10 | Renders the LLM's markdown answer (bold, lists, code) safely |

### 2.3 Why no LangChain / LlamaIndex
**Direct answer if asked:** "I wanted to understand each component, not chain pre-built abstractions. Every decision in this pipeline — chunk size, RRF k-value, confidence weighting, retry strategy — required me to look at the math and the failure modes. LangChain hides those decisions behind defaults; here every parameter is in source code I wrote."

This is also defensively honest — you avoided a 400-MB dependency tree.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React + Vite)                 │
│   UploadPage  ─────────────────────►  ChatPage                   │
│   (drag-drop)                          (Q&A, compare mode)       │
└────────────────────────────┬─────────────────────────────────────┘
                             │ /api proxy (vite.config.js)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI (api/main.py)                      │
│   POST /upload   ─►  BackgroundTasks ─►  ingest_document()       │
│   POST /query    ─►  asyncio.wait_for ─► answer_query()          │
│   GET  /papers   ─►  storage.list_papers()                       │
│   DELETE /papers/{id}                                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                ┌────────────┴───────────────────┐
                ▼                                ▼
   ┌──────────────────────┐         ┌────────────────────────────┐
   │   INGESTION PATH     │         │       QUERY PATH           │
   │ ingest_document.py   │         │    pipeline.py             │
   │   1 pdf_parser       │         │  1 query_planner           │
   │   2 section_detector │         │  2 query_router            │
   │   3 chunker          │         │  3 hybrid_retriever        │
   │   4 table_extractor  │         │     (bm25 + vector + RRF)  │
   │   5 embedder_worker  │         │  4 reranker (CrossEncoder) │
   │     └─ subprocess ─► │         │  5 generator (CoT)         │
   │     ChromaDB         │         │  6 evidence_grader         │
   └──────────────────────┘         │  7 evaluator               │
                                    │  8 retry_engine            │
                                    │  9 (compare_retriever)     │
                                    └────────────────────────────┘
                                                 │
                                                 ▼
                                    ChromaDB (data/chroma_db)
                                    Registry (data/papers.json)
```

### 3.1 Two entry points, one shared store
- **Ingestion** is one-shot per paper: PDF → chunks → vectors → disk.
- **Query** is repeated per question: query → retrieval → answer → evaluate → maybe retry.

Both share `data/chroma_db` (vector store) and `data/papers.json` (registry of uploaded papers).

### 3.2 Data flow at runtime
1. User drops a PDF → `POST /upload` → registry row created with `status="processing"` → response returned immediately → ingestion runs in a FastAPI `BackgroundTasks` → status flips to `ready` when done.
2. User asks a question → `POST /query` → `answer_query()` runs the 9-stage pipeline → returns answer + sources + confidence + evidence grades + reasoning chain → frontend renders ESSENCE/DETAIL/metrics/source chips.

---

## 4. Ingestion Pipeline — File by File

### 4.1 `ingestion/pdf_parser.py` — robust text extraction

**What it does:** Convert a PDF to a list of `{page_num, text, chars, page_width}` dicts.

**The non-obvious problem:** PDFs don't store space characters. They store character positions. "Hello" and "H e l l o" look identical to a naive reader. Also, research papers use **two-column** layouts, so reading top-to-bottom gives garbled output (left column line 1, right column line 1 merged together).

**How it solves them:**

1. **Filter rotated characters** — `[c for c in page.chars if c.get("upright")]` drops watermarks (arXiv stamps rotate them 90°).

2. **Reconstruct spaces** — `chars_to_text()` sorts characters by `x0`, then inserts `" "` whenever the gap between `chars[i].x0 - chars[i-1].x1 > 2` points. The 2-point threshold was chosen by inspection — fonts typically have <1pt inter-character spacing within a word.

3. **Detect column boundary per page** — `_detect_column_boundary()`:
   - Excludes the top 30% of the page (titles/authors are full-width and would obscure the gutter).
   - Builds a 1-D coverage histogram across page width — count of characters covering each x-coordinate.
   - Looks for a *valley* (bin with coverage <15% of median) in the middle 30% of the page.
   - Requires the valley to be ≥5pt wide.
   - Returns the midpoint, or `None` if single-column.

4. **Process two-column pages in reading order** — `_process_two_column()`:
   - Partitions chars into left and right by their center x.
   - Groups each side into lines independently (avoids "Le ftrightLeft right" merges).
   - Detects full-width lines (titles, footnotes) by checking if left text *flows* into the gutter (gap < 8pt) — those don't have the typical column gap and span the page.
   - Reading order: header full-width → left column top-to-bottom → right column top-to-bottom → footer full-width.

5. **`remove_credits_block`** — find `\nAbstract\n`, drop everything before. Authors/affiliations are noise.

6. **`remove_references_section`** — find `\nReferences\b`, drop everything after. Bibliography is noise for downstream retrieval.

**Why per-page column detection (and not once for the whole PDF):** Papers can be single-column on page 1 (title page) and two-column from page 2 onwards (body). Detecting per-page is the correct invariant.

**Known limitations (own these):**
- LaTeX-typeset tables (like in Attention Is All You Need) often render as positioned text, not as a `<table>` object — pdfplumber's `extract_tables()` misses them. Table extraction works for grid-bordered tables.
- Math equations are not parsed structurally. They appear as flat text with weird spacing.
- Multi-column layouts with more than 2 columns are not detected — the gutter search only looks for one valley.

---

### 4.2 `ingestion/section_detector.py` — find headings, build sections

**What it does:** Identify which lines in the parsed text are section headings (Introduction, Methodology, Results, 3.1 Encoder, etc.) and group all the body text under them.

**Why this matters:** Chunks must respect section boundaries. A chunk that's half "Introduction" and half "Methodology" gets retrieved for either topic but answers neither. Sections are also the natural unit for citations like `[Section: Encoder, Page: 4]`.

**Two-stage detection:**

**Stage 1 — Multi-signal scoring (`score_candidate`).**  
For every line, compute a score from six signals. A line scoring ≥ 8 becomes a *candidate*:

| Signal | Weight | Why |
|---|---|---|
| Avg font size > body+1.5pt | +3 | Headings are visually larger |
| Word count 1–8 | +2 | "Introduction" not "In this paper we discuss several…" |
| Doesn't end with `. , ; : ! ?` | +2 | Headings don't have terminal punctuation |
| Matches `^\d+(\.\d+)*\s+` | +2 | "1.", "3.1", "2.3.4" prefixes |
| `istitle()` or `isupper()` | +1 | Headings often title-cased |
| Vertical gap to previous line > 10pt | +2 | Visual spacing above a heading |

**Why multi-signal scoring instead of a single rule?** No paper formats things identically — some use bold not larger font, some don't number sections. Any one rule has false negatives. Scoring across six lets you tolerate any one missing.

**Why threshold 8 and not 5?** The original was 5. We tightened to 8 after empirically inspecting `out.txt` — at 5 we got noise (figure captions, table rows). At 8 we get real headings reliably, and the LLM confirmation layer (stage 2) cleans up the rest.

**Stage 2 — LLM confirmation (`confirm_headings_with_llm`).**  
Take all candidates, send to LLM in one batch call with the candidate line and 3 lines of context, get back `SECTION | SUBSECTION | NONE` per ID.

The prompt is structured for parse-ability: `ID | VERDICT` per line. We strip-and-split on `|`. Anything not matching the three valid labels is dropped — the LLM occasionally hallucinates labels and we just ignore those rather than crashing.

**Why batch?** One LLM call vs N. Saves time, saves quota, lets the LLM see candidates relative to each other (table-of-contents-style "Introduction" vs body-text "introduction" disambiguates more easily with neighbors visible).

**Stage 3 — Assembly (`assemble_sections`).**  
Walk every line on every page in reading order. When the line matches a confirmed heading, close the previous section bucket and open a new one. Pour subsequent lines into the active bucket. Stop entirely at "References".

Output: list of `{heading, type, page_num, body}` dicts.

**Defense for "Why do you need build_page_lines exposed from pdf_parser?":** The section detector and the section assembler must produce the **same** line list and the same line indexing, otherwise the (page_num, line_index) lookup from stage 1 won't match stage 3. Centralizing the ordering logic in `build_page_lines()` enforces that invariant.

---

### 4.3 `ingestion/chunker.py` — token-aware chunking

**What it does:** Split each section's body into overlapping ~512-token chunks.

**Parameters: 512 tokens, 100 token overlap.** Decided empirically — tested 256, 512, 1024 sizes × 50, 100 overlap on "Attention Is All You Need":
- **256** over-fragments — short sections (Conclusion) get split into 2-3 pieces, losing local coherence.
- **1024** under-splits — long sections (Methodology) become one giant chunk, hurting retrieval precision (you retrieve everything or nothing).
- **512** produces 27 chunks for that paper — short sections fit cleanly, long sections get 2-3 chunks, all the right tradeoffs.
- **100 overlap** vs 50: academic sentences average 20-30 tokens. 100 captures 3-4 complete sentences of shared context at boundaries; 50 risks cutting in mid-sentence.

**Token-level sliding window, not word-level.** Uses `tiktoken.get_encoding("cl100k_base")`:
```python
tokens = tokenizer.encode(text)        # list of int token IDs
while start < len(tokens):
    chunk = tokenizer.decode(tokens[start:start+chunk_size])
    start += (chunk_size - overlap)
```

**Why cl100k_base specifically:** It's the encoding GPT-4 and modern OpenAI-compatible models use. Our token counts match what the LLMs see, so we never accidentally exceed context windows.

**Why chunk per section, not on full text:** If you slide across the entire concatenated document, chunks at section boundaries contain content from two unrelated sections. Bad. Per-section chunking guarantees every chunk belongs to one topic.

Output per chunk: `{chunk_id, section, section_type, page_num, chunk_index, total_chunks_in_section, text, token_count}`. The metadata is essential — it's what becomes the `[Section: X, Page: N]` citation later.

---

### 4.4 `ingestion/table_extractor.py` — tables as queryable chunks

**What it does:** Use `pdfplumber.page.extract_tables()` to detect bordered tables and convert them to markdown text chunks.

**Output schema mirrors regular chunks** but with `section_type: "table"` and `section: "Table N (page P)"`. Same downstream behavior — they get embedded and stored alongside text chunks.

**Filters out noise:** Skip tables with `<2 rows` or `<2 columns` (those are usually layout artifacts, not real tables).

**Markdown format chosen:** `| col | col |` with `| --- | --- |` separator row. LLMs handle markdown tables well; row order and column alignment are preserved.

**Honest limitation:** LaTeX-positioned tables (like in Attention Is All You Need) often render as text and don't trigger `extract_tables()`. Works on grid-bordered tables in 10-Ks, ML papers with proper table formatting.

---

### 4.5 `ingestion/embedder.py` + `embedder_worker.py` — vectorize and store

**What it does:** Convert each chunk's text to a 384-dim vector via `all-MiniLM-L6-v2`, store in ChromaDB with metadata.

**Why MiniLM-L6-v2 specifically:**
- 22M params, ~80MB on disk — small enough to bundle and run on CPU.
- 384-dim output — small vectors mean fast cosine search.
- Trained on 1B+ sentence pairs — strong semantic representation for general English including academic text.
- The standard cheap baseline; nothing fancier is needed for paper-scale (~100 chunks per paper).

**Why a subprocess (`embedder_worker.py`) instead of in-process?**  
There's a **silent native Windows DLL collision** between PyTorch (which sentence-transformers loads) and pdfplumber/Groq libs. They both depend on certain Microsoft Visual C++ runtimes that mismatch. In-process embedding caused random segfaults on Windows. The fix: serialize chunks to a temp JSON file, spawn a Python subprocess that imports sentence-transformers fresh, embeds, writes to ChromaDB, exits. Subprocess isolation guarantees a clean DLL space.

**How storage works:**
- ChromaDB collection name = sanitized paper_id (alphanumeric + hyphen only, lowercased).
- IDs = `md5(text)` — content-hashed so re-ingesting doesn't duplicate.
- Metadata is stringified (ChromaDB requires str/int/float — no lists/dicts).
- `collection.upsert()` — idempotent, safe to re-run.
- After upsert, **invalidate the BM25 cache** so the BM25 retriever rebuilds with the new chunks next query.

---

### 4.6 `ingestion/ingest_document.py` — the orchestrator

Composes the previous five steps in order:
1. `extract_text_from_pdf(pdf_path)`
2. Clean credits + references
3. `build_candidates(pages)` → `confirm_headings_with_llm(candidates)` → `assemble_sections(...)`
4. `chunk_sections(sections, 512, 100)`
5. `extract_tables_from_pdf(pdf_path)` — append to chunks with sequential IDs
6. Write chunks to a temp JSON, spawn `embedder_worker.py` as subprocess.

Returns `{success, paper_name, total_pages, num_chunks, error}`.

**Failure mode:** if the subprocess returns non-zero, `subprocess.CalledProcessError` is caught, stdout/stderr printed, RuntimeError raised. Caller (in `api/main.py`) catches it and flips the paper's status to `failed`.

---

## 5. Query Pipeline — the 9 stages, deeply

### 5.1 `ingestion/query_planner.py` — Stage 1: planning

**Big idea:** Before retrieving, classify the query so downstream components know what kind of answer to produce.

**Single LLM call** producing JSON:

```json
{
  "answer_type":      "<factual | causal_explanation | mechanism | comparison | critique | summarization | analysis | hypothetical>",
  "key_concepts":     ["term1", "term2"],
  "sub_questions":    ["...", "..."],
  "answer_structure": ["step 1", "step 2", "step 3"],
  "complexity":       "<simple | multi_hop>"
}
```

**Each field's role downstream:**
- `answer_type` → drives `ANSWER_TYPE_CONFIG` in the router (controls `retrieval_k`/`llm_k`) AND drives the tone instruction in the generator.
- `key_concepts` → boost terms appended to the BM25 query (helps lexical retrieval catch the right jargon).
- `sub_questions` → if `complexity == multi_hop`, run retrieval for each.
- `answer_structure` → injected into the generator's prompt as a numbered checklist the model writes against (silently).
- `complexity` → branches the retrieval strategy.

**Fallback plan:** if JSON parse fails, return a minimal safe plan (`answer_type: factual`, `complexity: simple`, `sub_questions: [query]`). The pipeline never hard-crashes at planning.

**Validator (`_validate_plan`)** mutates the dict in place to fill missing keys — defense against partial LLM output.

**Why a single LLM call for planning, not separate intent + decomposition calls?**  
Before Session 1 we had `intent_detector.py` + `multi_hop.decompose_query()` as separate calls. That's two round-trips, two prompt+output overheads, two cache misses. Merging them into one structured plan: faster, cheaper, and lets the LLM reason about intent and decomposition jointly ("this is a comparison so the sub-questions should target each side"). Backwards compat: `multi_hop_retrieve()` still accepts `sub_questions=None` and falls back to its own decomposition if a caller doesn't supply a plan.

---

### 5.2 `ingestion/query_router.py` — Stage 2: routing

**Resolves retrieval config from `answer_type`:**

```python
ANSWER_TYPE_CONFIG = {
    "factual":            {"retrieval_k": 10, "llm_k": 3},
    "summarization":      {"retrieval_k": 15, "llm_k": 7},
    "critique":           {"retrieval_k": 10, "llm_k": 5},
    "comparison":         {"retrieval_k": 12, "llm_k": 6},
    "mechanism":          {"retrieval_k": 10, "llm_k": 5},
    "causal_explanation": {"retrieval_k": 10, "llm_k": 5},
    "hypothetical":       {"retrieval_k": 12, "llm_k": 6},
    "analysis":           {"retrieval_k": 12, "llm_k": 7},
}
```

`retrieval_k` = how many chunks BM25+vector retrieves (cast a wide net).  
`llm_k` = how many chunks survive reranking and go to the generator (focus the LLM).

**Why this differs per type:**
- Factual: small `llm_k=3` because the answer should be one fact from one chunk.
- Summarization: `llm_k=7` because you need broad coverage.
- Critique: `llm_k=5` because you need contrasting passages.

**Branches on `complexity`:**
- `simple` → `hybrid_retrieve()` once with `key_concepts` as boost terms.
- `multi_hop` → `multi_hop_retrieve()` runs hybrid retrieval for the original query AND each sub-question, dedupes by chunk ID.

**Always reranks** with CrossEncoder after retrieval, keeping top `llm_k`.

Returns `{query, plan, config, chunks}` to the pipeline.

---

### 5.3 `ingestion/hybrid_retriever.py` — Stage 3: retrieval

**Hybrid = BM25 (lexical) + dense vectors (semantic), fused with Reciprocal Rank Fusion.**

**Why both?**
- **BM25** excels at exact-keyword queries: "What is the dropout rate?" → finds "dropout = 0.1" because "dropout" appears verbatim.
- **Dense vectors** excel at paraphrased queries: "How does the model avoid overfitting?" → finds the dropout chunk even though the query word "overfitting" doesn't appear.
- Neither alone covers both cases. Hybrid does.

**Reciprocal Rank Fusion** — the merge function:
```
score(d) = Σ over each retriever:  1 / (k + rank_of_d_in_that_retriever)
```
With `k=60` (Cormack et al., 2009; standard value).

**Why RRF and not weighted score sums?** BM25 scores are unbounded (10–50 typical); cosine similarity is bounded [-1, 1]. Mixing raw scores requires per-corpus normalization that's brittle. RRF only uses *ranks* — robust to scale differences, no tuning, well-cited.

**Boost terms** — if a `key_concepts` list is passed, the query string is concatenated with them before being sent to BM25 (vector search uses the original query — boosting via BM25 is sufficient and avoids embedding drift).

**Code path:**
```
bm25_retrieve(boosted_query, top_k=10) ─┐
                                        ├─► RRF fusion ─► top-K
vector_retrieve(query, top_k=10) ───────┘
```

---

### 5.4 `ingestion/bm25_retriever.py` — BM25 backend

**Builds the BM25 index lazily and caches it per paper.** Cache key = `paper_name`. Invalidated on re-ingestion (called from `embedder.py` after upsert).

**Tokenization:** `doc.lower().split()` — whitespace split, lowercased. Naive but works because:
- BM25 doesn't need stemming or stop-word removal — its IDF term down-weights common words automatically.
- Academic papers are mostly ASCII; punctuation-heavy text isn't the failure mode.

**Returns** `[{text, metadata, bm25_score}]` for the top-k chunks. Metadata flows from ChromaDB (the same one written at ingestion time).

---

### 5.5 `ingestion/retriever.py` — dense vector backend

Loads the shared embedding model (via `models.py` — see 5.10), encodes the query, runs `collection.query()` against the paper's ChromaDB collection. Returns `[{text, metadata, distance}]`.

ChromaDB's default similarity metric is L2 (euclidean) on the embedding space, not cosine. For normalized embeddings these rank-identically — both are monotonic with each other when vectors are unit-norm. sentence-transformers outputs are normalized by default, so this is fine.

---

### 5.6 `ingestion/reranker.py` — Stage 4: cross-encoder reranking

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`. **About 22M params, runs on CPU in <1s for 10 pairs.**

**What it does that bi-encoders can't:** A bi-encoder (the embedding model from §5.5) encodes query and document *separately* and compares vectors. A cross-encoder encodes them **together** — the transformer attends across both, producing a single relevance score with awareness of all term interactions.

**Trade-off:** Cross-encoders are accurate but slow — O(N) inference vs O(N) embedding lookups. You don't want to use them for the initial retrieval (would need to score every chunk in the paper). You use them after retrieval narrows the candidate set to ~10 chunks.

**Pipeline:** RRF top-15 → CrossEncoder rescore → top-`llm_k` (3–7 depending on `answer_type`).

This is the "rerank stage" in the textbook RAG playbook (Pinecone's "two-stage retrieval"). It's what consistently boosts retrieval quality on benchmarks.

---

### 5.7 `ingestion/generator.py` — Stage 5: CoT answer generation

**The crown jewel of the pipeline.** Two things make it work: **chain-of-thought scratchpad** and **ESSENCE + DETAIL format**.

**The 6-step CoT scratchpad** (the model writes this *before* the final answer):

| Step | What the model does | Why |
|---|---|---|
| `[INVENTORY]` | Lists what each chunk explicitly states | Forces it to ground in the chunks, not its parametric memory |
| `[GAPS]` | Lists what the question asks but no chunk addresses | Forces it to admit gaps before "filling them with imagination" |
| `[INFERENCE]` | Lists what can be reasonably inferred | Distinguishes "stated" from "inferred" — used later by the grader |
| `[UNCERTAINTY]` | Flags anything that should be labeled as inferred | Last chance to hedge before writing |
| `[STRUCTURE]` | Maps `answer_structure` steps to evidence | Plans the answer's flow against the plan |
| `[WRITE]` | Writes the final ESSENCE + DETAIL answer | Reasoning is done; now produce output |

**Why CoT helps:** A model that's been forced to enumerate facts and gaps before generating an answer hallucinates less, because the hallucination would visibly contradict its own inventory. This is the standard "reasoning before answering" technique, applied with structured labels so we can split it back out programmatically.

**ESSENCE + DETAIL format:**
- **ESSENCE:** 2-3 sentences capturing the single most important insight. Sharp, standalone — the reader should grasp the core answer from this alone.
- **DETAIL:** Expansion using ONLY context chunks, following `answer_structure` invisibly as a writing guide. Maximum 2 paragraphs.

The frontend renders ESSENCE prominently and DETAIL as a collapsible accordion (see §7.4).

**Citation rules:**
- Max 3 citations per answer.
- Format `[Section: <section>, Page: <num>]`.
- Never reference chunks by number — there's a regex post-processor (`_strip_chunk_refs`) that scrubs `Chunk N` references the model writes despite instructions. This is needed because LLMs *always* slip in "Chunk 3" no matter how loudly you tell them not to.

**Tone instruction per `answer_type`** — `ANSWER_TYPE_INSTRUCTIONS` dict. E.g. for `factual`: "Answer precisely and concisely. State the exact fact, number, or definition asked for." For `critique`: "Identify limitations… distinguish between what the paper claims and what it empirically demonstrates."

**Split logic** — `_extract_reasoning_and_answer()` searches for `[WRITE]` or `STEP N — WRITE` (handles em/en/hyphen dash variants — models choose any). Everything before is the reasoning chain (returned as `reasoning_chain` for debugging); everything after is the final answer.

---

### 5.8 `ingestion/evidence_grader.py` — Stage 6: per-sentence grading

**The single most impactful module in the pipeline.**

**What it does:** After the answer is written, take each sentence in the answer + the source chunks, send to an LLM as a fact-checker, get back a grade per sentence:
- `DIRECT` — explicitly stated in a chunk (near-verbatim support exists)
- `INFERRED` — logically follows from chunk content, but not directly stated
- `UNSUPPORTED` — no chunk supports this; model used outside knowledge or hallucinated

**UNSUPPORTED sentences are physically removed** from the answer before it reaches the user or the evaluator.

**This is what actually fixes faithfulness.** Not better metrics — better content. The metrics are descriptive; the grader is prescriptive. If the model writes a hallucinated sentence and the grader removes it, the answer the user sees is *strictly more faithful* than the answer the LLM produced.

**Sentence splitting (`_split_sentences`):**
- Treats `**ESSENCE:**` / `**DETAIL:**` headers as their own units (preserved with `chunk_ref: "header"`).
- Splits on `. ` then re-glues the period.
- Drops lines shorter than 15 chars (avoids partial fragments).

**LLM prompt** sends numbered sentences + numbered chunks, asks for a JSON array of `{sentence, grade, chunk_ref}`.

**Reconstruction (`_reconstruct_answer`):**
- Builds a `grade_map` keyed on the first 60 chars of each sentence (robust to whitespace drift).
- For each sentence in order, decides keep/remove.
- Does string `replace` on the original text — preserves whitespace, headers, formatting.
- Cleans up: collapse 3+ newlines to 2, collapse double spaces, drop orphaned ≤2-word lines.

**Failure mode:** if the grader's JSON parse fails, `grades = None` → returns the **original answer unchanged** with `grading_failed: True`. The pipeline never breaks because grading failed.

**What the pipeline does with the result** (in `pipeline.py`):
```python
_pre_eval      = evaluate_answer(query, raw_answer, chunks)
grading_result = grade_answer(raw_answer, chunks)
answer         = grading_result["cleaned_answer"]

if _pre_eval["faithfulness"] > 0.75 or grading_result["removed_count"] == 0:
    eval_scores = _pre_eval               # reuse — answer didn't change meaningfully
else:
    eval_scores = evaluate_answer(query, answer, chunks)   # re-score the cleaned answer
```
This is the Session 3 optimization — saves one full `evaluate_answer()` call when grading didn't change anything (~50% reduction in evaluation overhead).

---

### 5.9 `ingestion/evaluator.py` — Stage 7: self-evaluation

**Two metrics, both computed locally without API calls.**

**Faithfulness** — "Is every sentence in the answer supported by the chunks?"
1. Split answer into sentences (>20 chars only).
2. Embed each sentence + each chunk via `all-MiniLM-L6-v2`.
3. For each sentence, compute its max cosine similarity to any chunk.
4. Count sentences where that max ≥ `_SUPPORT_THRESHOLD = 0.55`.
5. `faithfulness = supported_count / total_sentences`.

**Why 0.55?** Empirically tuned on test queries. Below ~0.45 you get false positives (the sentence is loosely related to a chunk on the same topic, not actually supported). Above ~0.65 you get false negatives (a clearly-supported sentence with paraphrased wording fails the cutoff).

**Why local embedding similarity instead of an LLM judge or RAGAS?**
- An LLM judge requires an API call per sentence — expensive and slow.
- RAGAS is good but originally needed OpenAI for its built-in metrics; we wanted no required API for evaluation.
- Local embedding similarity is fast (~50ms per answer), free, deterministic — and good enough for the use case.

**Answer relevancy** — "Is the answer about the query?"
- `cosine(embed(query), embed(answer))`.
- Bounded [-1, 1]. In practice 0.4–0.9.

**Confidence** (used as the retry gate):
```
confidence = round((faithfulness * 0.7 + answer_relevancy * 0.3) * 100, 2)
```

**Why 70/30 and not 50/50?**  
Faithfulness is the dominant safety property — an answer can be perfectly *on-topic* but completely *made up*. We weight it higher to penalize fluent fabrication.

---

### 5.10 `ingestion/models.py` — shared singleton

**One sentence:** Returns the same `SentenceTransformer("all-MiniLM-L6-v2")` instance to anyone who calls `get_embedding_model()`.

**Why this exists:** Before Session 3, `retriever.py` and `evaluator.py` each instantiated their own copy → 2 × ~80MB in memory + 2 × 6-second model loads at startup. Now: one instance, lazy-loaded on first call.

**Lazy init pattern:**
```python
_embedding_model: SentenceTransformer | None = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model
```

The reranker (`reranker.py`) does the same pattern for the CrossEncoder.

---

### 5.11 `ingestion/retry_engine.py` — Stage 8: diagnose + retry

**The "agentic" part of the system.** If `confidence < 50`, the pipeline doesn't just return a bad answer — it tries to fix it.

**`diagnose_failure()`** — classify the failure as either `retrieval` or `generation`:

```
retrieval_ok = top_rerank_score >= 0.0    (or, fallback, faithfulness >= 0.05)

if not retrieval_ok:                          → "retrieval"
elif faithfulness < 0.80:                     → "generation"
elif answer_relevancy < 0.40:                 → "generation"
else:                                         → "retrieval"
```

**The logic:**
- If the reranker couldn't find any chunk worth keeping (its top score is bad), the chunks themselves are wrong → **retrieval failure**.
- If retrieval was fine (good chunks) but the answer doesn't faithfully reflect them, the LLM hallucinated → **generation failure**.
- If retrieval and faithfulness are both ok but relevancy is bad, the LLM answered a different question than was asked → **generation failure**.

**`retry_query()`** — execute different strategies per failure type:

| failure_type | attempt 2 | attempt 3 |
|---|---|---|
| retrieval | LLM expands query → new retrieval run | Different expansion + only top 3 chunks |
| generation | Same query, slice to top 4 chunks (force focus) | Expand + slice to top 3 |

**`_expand_query()`** — a small LLM call that rewrites the user's question using technical vocabulary that would appear in the paper. E.g., "How is relevance computed between tokens?" → "How is the scaled dot-product attention score computed between query and key vectors?" The expansion helps BM25 find passages that use the paper's jargon rather than the user's plain language.

**MAX_ATTEMPTS = 3.** After three, return whatever has the best confidence so far.

**Out-of-domain shortcut** (in `pipeline.py`):
```python
if attempt == 1 and eval_scores["answer_relevancy"] < 0.05:
    out_of_domain = True
    # skip retries — they only make things worse
```

If the relevancy is *catastrophically* low on attempt 1, retrying with query expansion won't help (you can't expand a query that's about something entirely outside the paper). We short-circuit and surface a "this question doesn't appear related to the paper" warning.

---

### 5.12 `ingestion/multi_hop.py` — multi-hop retrieval for complex questions

**When triggered:** `plan["complexity"] == "multi_hop"`.

**What it does:** Run hybrid retrieval for the **original query** AND for **each sub-question** from the plan, then deduplicate by chunk ID. Returns a wide pool of chunks before reranking trims it.

**Example:** Query "Why can the Transformer train faster than RNNs?" decomposes into:
1. How do RNNs process sequences sequentially and what limits their parallelization?
2. How does self-attention connect positions with constant operations?
3. What training times and hardware did the Transformer use?

Each sub-question retrieves from a different part of the paper. Merging gives the LLM context from RNN limitations + self-attention parallelization + concrete training numbers — content that single-pass retrieval would never have aggregated.

**Optimization (Session 1):** if `sub_questions` is provided by the router (from the plan), skip the decomposition LLM call entirely. Falls back to its own `decompose_query()` only for direct callers/tests that don't have a plan.

---

### 5.13 `ingestion/compare_retriever.py` + multi-paper comparison

**Used when `POST /query` receives `paper_ids: [id1, id2]`.**

**`compare_retrieve()`:** Run `hybrid_retrieve()` against each paper's collection separately, tag chunks with `paper_label: "A"` or `"B"` and `paper_id: <uuid>`, **interleave** them (A1, B1, A2, B2, …) so both papers always appear in the LLM's context window.

**`compare_papers()` in pipeline.py:** Forces `plan["answer_type"] = "comparison"` and overrides `answer_structure`:
```python
["State how Paper A addresses the topic with supporting evidence",
 "State how Paper B addresses the topic with supporting evidence",
 "Highlight the key differences between the two papers",
 "Note any agreements or common ground"]
```

Returns the same dict shape as `answer_query` plus `is_comparison: True` and `paper_ids: [a, b]`. The frontend renders the "A vs B" badge and source chips with A/B labels off these fields.

---

### 5.14 `ingestion/llm_client.py` — unified provider rotation

**One function, `chat_completion(messages, max_tokens, temperature)`. Every LLM call in the entire codebase goes through it.**

**Provider list (priority order):**
1. **Gemini** (`gemini-2.0-flash`) — 1M tokens/day free tier. Endpoint: `https://generativelanguage.googleapis.com/v1beta/openai/`. Yes, Gemini has an OpenAI-compatible endpoint — that's the trick that lets us use one SDK.
2. **Cerebras** (`llama3.1-8b`) — ~1M tokens/day. Endpoint: `https://api.cerebras.ai/v1`.
3. **Mistral** (`mistral-small-latest`) — 1B tokens/month. Endpoint: `https://api.mistral.ai/v1`.
4. **Groq-1** (`llama-3.3-70b-versatile`) — 100k tokens/day.
5. **Groq-2** — second key, same model.

**How rotation works:**
```python
for provider in _PROVIDERS:
    try:
        response = provider["client"].chat.completions.create(...)
        return response.choices[0].message.content.strip()
    except Exception as e:
        if any(kw in str(e).lower() for kw in _SKIP_KEYWORDS):
            continue                # rate-limited / quota — try next
        raise                       # unexpected error — surface it
```

**`_SKIP_KEYWORDS`**: `rate`, `quota`, `429`, `limit`, `exhausted`, `invalid_argument`, `api_key_invalid`, `invalid api key`, `resource_exhausted`, `too many requests`, `not_found`, `404`, `does not exist`.

**Why this list:** Each provider returns different error messages for "stop bothering me." Catching any of these substrings handles all of them.

**What's caught vs raised:**
- Rate / quota / "not found" / 404 → fall through to the next provider.
- Network error, auth error, schema error → raise immediately (the user has something genuinely misconfigured; silent retry would mask it).

**The Session 1 bug fix:** the original code had `continue` inside `except` *without* the keyword filter, which silently swallowed every error type. After the fix, only rate-limit-shaped errors trigger fallback; everything else surfaces.

**Why provider rotation matters:** Each free tier exhausts. Without rotation, the pipeline dies the moment Gemini 429s. With rotation, the user gets answers as long as *any* provider has capacity. No single outage crashes the pipeline.

**Why OpenAI as the protocol:** Every modern provider has implemented OpenAI-compatible endpoints because the OpenAI SDK is the de-facto standard. Using one client means switching models is a one-line config change per provider entry — not a code refactor.

---

## 6. The Pipeline — full sequence summary

When a query comes in via `POST /query`:

```
answer_query(query, paper_name, request_id):

  for attempt in 1..MAX_ATTEMPTS:
      if attempt == 1:
          plan        = plan_query(query)                         # LLM
          chunks_raw  = if plan.complexity == "multi_hop":
                            multi_hop_retrieve(query, sub_questions, key_concepts)
                        else:
                            hybrid_retrieve(query, boost_terms=key_concepts)
          chunks      = rerank(query, chunks_raw, top_k=llm_k)    # CrossEncoder
          generated   = generate_answer(query, chunks, plan)      # LLM (CoT)
          raw_answer  = generated["answer"]
      else:
          retry_result = retry_query(query, paper_name, failure_type, attempt)
          raw_answer   = retry_result["answer"]
          chunks       = retry_result["chunks"]

      pre_eval     = evaluate_answer(query, raw_answer, chunks)   # local embeddings
      grading      = grade_answer(raw_answer, chunks)             # LLM (fact-check)
      answer       = grading["cleaned_answer"]                    # UNSUPPORTED removed

      if pre_eval.faithfulness > 0.75 OR grading.removed_count == 0:
          eval_scores = pre_eval                                  # reuse
      else:
          eval_scores = evaluate_answer(query, answer, chunks)    # re-score

      confidence = (eval_scores.faithfulness * 0.7 +
                    eval_scores.answer_relevancy * 0.3) * 100

      if attempt == 1 AND eval_scores.answer_relevancy < 0.05:
          out_of_domain = True
          break                                                   # skip retries

      track best result

      if confidence >= 50 AND not out_of_domain:
          return best result

      failure_type = diagnose_failure(...)                        # retrieval vs generation

  return best result with warning
```

**Total LLM calls per typical query:**
- Attempt 1: 1 planning + 1 generation + 1 grading = 3
- Multi-hop: same as above; sub-questions come from the plan, no extra decomposition call.
- Retry attempts (2 and 3): 1 query expansion + 1 routing (re-plans inside `route_query`) + 1 generation + 1 grading = ~4 per retry.

**Typical end-to-end latency:**
- Simple query, no retry: ~7-8 seconds.
- Multi-hop query: ~12-15 seconds.
- Query that retries once: ~20-25 seconds.

---

## 7. Frontend — file by file

### 7.1 `frontend/vite.config.js` — dev proxy

```js
server.proxy['/api'] = {
  target: 'http://localhost:8000',
  changeOrigin: true,
  rewrite: (path) => path.replace(/^\/api/, ''),
}
```

**Why:** Vite serves the React app on `:5173`, FastAPI on `:8000`. Without a proxy, `fetch('/api/upload')` would 404. The proxy rewrites `/api/upload` → `http://localhost:8000/upload`. **Same-origin to the browser**, no CORS issues during development.

In production you'd serve the built `dist/` from FastAPI directly or front everything with nginx — same path.

### 7.2 `frontend/src/api.js` — HTTP client

Five functions:
- `uploadPaper(file)` — POST `/api/upload` with multipart FormData.
- `getPaperStatus(paperId)` — GET `/api/status/{id}`, polled by the upload page every 2s.
- `queryPaper(paperId, question)` — POST `/api/query` with `{paper_id, question}`.
- `comparePapers(a, b, question)` — POST `/api/query` with `{paper_ids: [a, b], question}`.
- `listPapers()` / `deletePaper(id)` — workspace switcher.

Plain `fetch()`, no Axios. All return parsed JSON.

### 7.3 `frontend/src/App.jsx` — top-level routing

Tiny state machine: if `currentPaper` is set, show `<ChatPage>`; otherwise show `<UploadPage>`. The "go back" button from chat sets `currentPaper = null`.

No router library — there are only two pages and they're cleanly conditional.

### 7.4 `frontend/src/pages/UploadPage.jsx`

**Drag-and-drop PDF upload with polling.**

State machine: `idle → uploading → processing → (ready→redirect | error)`.

Flow:
1. User drops or selects a `.pdf`.
2. `setPhase('uploading')` → call `uploadPaper(file)`.
3. Response contains `paper_id`. Switch phase to `processing`.
4. `setInterval` every 2s — poll `/api/status/{paper_id}`.
5. When status flips to `ready`, clear the interval, call `onPaperReady(paper)` → navigates to chat.
6. If status `failed`, show error.

Why polling and not WebSockets/SSE? Simpler, ingestion is short enough (30-60s) that 2s polling is fine.

UI: cosmic-orb gradient background, animated triple-ring spinner, stage indicator (`Parse → Chunk → Embed → Index` — purely cosmetic; the backend doesn't expose intermediate stages, but visual feedback eases the wait).

### 7.5 `frontend/src/pages/ChatPage.jsx`

The main interface. **904 lines** — UI-heavy. Key components:

- **`CosmicOrbs`** — decorative background blobs.
- **`MetricRing`** — animated SVG ring showing confidence/faithfulness/relevancy. `stroke-dashoffset` interpolation gives the fill animation.
- **`SparkLine`** — decorative procedural sine wave seeded by confidence (yes, it's pseudo-data — not a real time series).
- **`SourceChip`** — renders one source as a pill. Two variants:
  - text chunks: cyan, shows section + page.
  - table chunks (`section_type === "table"`): violet pill labeled "TABLE".
  - In compare mode: extra A/B badge.
- **`EvidenceGrading`** — shows the evidence quality summary:
  - Chip counts: DIRECT (cyan), INFERRED (amber), REMOVED (red).
  - Click "Breakdown" → expands per-sentence list with colored dots; REMOVED sentences shown with strikethrough.
  - Renders nothing if `grading_failed` or no grades exist.
- **`TelemetryPanel`** — fake-but-plausible per-stage timing bars (Query Planning, Retrieval, Reranking, Generation, Evidence Grading, Evaluation). The `request_id` shown is the real one from the backend; the per-stage timings are derived deterministically from confidence (not real measurements).
- **`Message`** — renders one assistant turn. Parses the answer into `essence` and `detail` using regex (`/\n\s*\*{0,2}DETAIL:?\*{0,2}\s*/i`), shows essence in the open and detail behind a "Full Breakdown" accordion.

**ChatPage state:**
- `paper` — currently active paper.
- `allMessages[paper_id]` — message history *per paper*, so switching papers doesn't lose history.
- `compareMode`, `comparePaper2` — toggles the compare flow.
- `papers`, `showSwitcher` — workspace selector.

**Send flow:**
```js
const result = (compareMode && comparePaper2)
  ? await comparePapers(paperId, comparePaper2.paper_id, question)
  : await queryPaper(paperId, question)
```

**Delete flow:** confirm prompt → `deletePaper(id)` → remove from `papers` list. If the deleted paper was the comparison target, clear `comparePaper2`.

### 7.6 `frontend/src/index.css` — Tailwind theme + custom components

Defines fonts, the `.pm-card` glassmorphism component, cosmic orb keyframe animations, evidence chip variants (cyan/amber/red), accordion CSS-grid trick (`grid-template-rows: 0fr → 1fr`) for smooth expand/collapse, markdown content overrides.

**The accordion trick:** A pure-CSS way to animate from collapsed to fully-expanded *without* knowing the content height in advance. CSS grid interpolates `0fr → 1fr` smoothly, and the inner element has `overflow: hidden`. No JS height measurement needed.

---

## 8. API Layer — `api/` deep dive

### 8.1 `api/main.py` — FastAPI server

**Routes:**

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | Liveness probe — `{status: "ok"}` |
| POST | `/upload` | Accept PDF, create registry record, write PDF to `data/papers/{id}.pdf`, schedule `run_ingestion` as a `BackgroundTasks`, return `{paper_id, filename, status: "processing"}` immediately |
| GET | `/status/{paper_id}` | Return the paper's registry row (status, errors, timestamps) — polled by frontend |
| GET | `/papers` | List all papers, newest first |
| DELETE | `/papers/{paper_id}` | Remove registry row, delete PDF file, drop ChromaDB collection |
| POST | `/query` | Run pipeline — single paper OR comparison (if `paper_ids` has 2 entries) |

**Why `BackgroundTasks` for ingestion?**  
Ingestion takes 30-60s (parsing, LLM section detection, embedding). If we ran it synchronously, the upload request would block for that long. Background tasks let the HTTP response return immediately (with `paper_id` and `status: processing`), and the frontend polls `/status/{id}` to know when it's done.

**Why `asyncio.wait_for(..., timeout=60.0)` for `/query`?**  
The pipeline is mostly CPU-bound (embedding) + LLM API blocking. Run it in a thread pool via `loop.run_in_executor(None, answer_query, ...)`. The `asyncio.wait_for` wraps it with a hard timeout — if a request hangs past 60s (an LLM provider that's *slow* rather than rate-limited), return HTTP 504 instead of holding the connection forever. Comparison gets 120s because two retrievals + a longer answer take longer.

**Why `paper_id` and `paper_ids` are both optional in the request model?**  
Backwards compatibility. Single-paper queries use `paper_id`. Comparison uses `paper_ids: [a, b]`. We accept either.

**Validation order on `/query`:**
1. If `paper_ids` has exactly 2, verify both exist and are `ready` → run `compare_papers`.
2. Else, use `paper_id` (or `paper_ids[0]` if `paper_id` is empty) → verify it's `ready` → run `answer_query`.
3. 404 if not found, 400 if not ready.

**`request_id`** is generated at the top of every `/query` and threaded through:
- Logged at the end (with duration, confidence, attempts, pass/fail).
- Injected into the response JSON so the frontend's Trace panel shows the real request ID.

### 8.2 `api/storage.py` — paper registry

**Data store:** `data/papers.json` — a flat JSON dict keyed by `paper_id`.

```json
{
  "uuid-1": {
    "paper_id": "uuid-1",
    "filename": "Attention.pdf",
    "status": "ready",       // processing | ready | failed
    "uploaded_at": "...",
    "completed_at": "...",
    "error": null
  }
}
```

**Critical detail: every read-modify-write is wrapped in `threading.Lock()`.** Without it, two concurrent uploads would race:
- Both `_load_registry()` → both see N-1 records.
- Each adds their own → both write back N records.
- One of them wins, the other is lost.

**Session 2 verification:** 5 concurrent uploads → 5 unique records, 0 errors. Without the lock, you'd reliably lose 1-2.

**`uuid.uuid4()` for paper_id** — globally unique, no coordination needed across uploads.

### 8.3 `api/logger.py` — structured logging

```
[abc12345] PASS paper=2d4db181 query="How does attention work?" duration=7420ms confidence=68.5 attempts=1
```

- `req_id` — 8-char hex (`uuid.uuid4().hex[:8]`). 16M possible values — collision-free per session.
- Query truncated to 80 chars.
- Status = PASS/FAIL (above/below 50 confidence).
- Duration in ms, confidence with 1 decimal, attempts as int.

This is grep-friendly, easy to pipe into log aggregation, and matches one line per request.

---

## 9. Storage Layout

```
data/
├── papers.json           # registry — single source of truth for paper state
├── papers/
│   └── {paper_id}.pdf    # raw uploaded files
└── chroma_db/            # vector store (binary, persistent)
    └── (ChromaDB internals — one collection per paper_id)
```

- **`papers.json` and `chroma_db` must stay in sync.** Delete operations remove from both. If they diverge (e.g., a partial ingestion crashes), the `delete_paper` endpoint tolerates a missing ChromaDB collection (try/except, pass).
- **PDFs are kept after ingestion** — useful for re-processing if ingestion logic changes; the user can re-trigger ingestion without re-uploading.

---

## 10. Concept & Theory Reference (interview ammo)

### 10.1 RAG (Retrieval-Augmented Generation)
**Definition:** A pattern where, instead of asking an LLM to recall facts from its weights, you retrieve relevant text from an external source and put it in the prompt. The LLM grounds its answer in that retrieved context.

**Why it matters:** LLMs hallucinate, and updating their weights is expensive. RAG lets you keep the LLM static and update knowledge by updating the retrieval index.

**Limits:** RAG is only as good as its retrieval. If the right chunk isn't retrieved, the LLM either says "I don't know" or hallucinates.

### 10.2 Embeddings
A learned mapping from text → a dense vector in ℝᵈ (here d=384) such that semantically similar texts have small angular distance. The standard tool for "find me text like this."

`all-MiniLM-L6-v2` was fine-tuned on contrastive sentence pairs (similar pairs pulled together, dissimilar pushed apart) — so cosine similarity in its output space correlates with semantic similarity.

### 10.3 Cosine similarity
`cos(a, b) = (a · b) / (||a|| ||b||)`. Range [-1, 1]. For unit-normalized vectors, it's equivalent to ranked-by-dot-product. Used in `evaluator.py` for both faithfulness and answer relevancy.

### 10.4 BM25
Probabilistic lexical retrieval. Term frequency × inverse document frequency, modulated by:
- Saturation: doubling the term count doesn't double the score.
- Length normalization: long documents don't unfairly accumulate score.

Parameters `k1` (default 1.5) and `b` (default 0.75) — we use library defaults; they're well-validated for general English.

### 10.5 Reciprocal Rank Fusion (RRF)
`score(doc) = Σ_retrievers 1 / (k + rank_in_that_retriever)`. With k=60.

**Why it works:** Top-ranked documents in either retriever contribute ~1/60 to the total. A doc that ranks 1st in both retrievers scores ~2/61. A doc that ranks 1st in one and 50th in the other scores ~1/61 + 1/110 ≈ 0.025. So unanimity beats split confidence.

**Why not weighted score sums?** Scores from BM25 and cosine are on different scales. RRF ignores scale, uses only rank — robust by construction.

### 10.6 Cross-encoder vs bi-encoder
- **Bi-encoder:** encodes query and document separately → two vectors → cosine. Fast (precompute doc vectors). What MiniLM-L6 is in retrieval mode.
- **Cross-encoder:** concatenates query and doc, encodes them together → single relevance score. More accurate (attention attends across both) but O(N) inference — must score every query/doc pair fresh.

We use bi-encoder for retrieval (fast on whole corpus), cross-encoder for reranking (accurate on top-10).

### 10.7 Chain-of-Thought (CoT)
The technique of having the LLM produce intermediate reasoning *before* the final answer. Wei et al., 2022. Improves accuracy on reasoning tasks because the model can use its own intermediate tokens as scratch space.

Our 6-step scratchpad ([INVENTORY], [GAPS], …) is structured CoT — labeled steps so we can parse them programmatically and so the model has a fixed schema to follow.

### 10.8 Chunking strategies (defend the 512/100 choice)
- **Fixed-size token sliding window** (what we use): predictable, simple, well-tested.
- **Sentence-level chunking:** semantic boundaries, but chunks vary wildly in length, hurting retrieval consistency.
- **Recursive split (LangChain default):** splits on `\n\n` first, then `\n`, then `. ` — handles narrative text well but can break sections badly.
- **Semantic chunking:** uses embeddings to find break points where topics shift. More compute, marginal improvement.

We pick fixed-size + section-boundaried because academic papers already have natural sections, and 512/100 was empirically validated.

### 10.9 Reranking — why not just use cross-encoder for retrieval?
You'd have to cross-encode the query with every chunk in the corpus. For 100 chunks per paper this is feasible; for 10,000 it's prohibitive. The two-stage approach (cheap retrieval → expensive rerank on top-N) is the standard scalable design.

### 10.10 The hallucination problem (and how the grader addresses it)
**Hallucination = the model generates plausible-sounding content that isn't supported by the context.** Causes:
- The model fills gaps with parametric knowledge instead of saying "I don't know."
- The model interpolates between two retrieved facts in a way that produces a false third "fact."

**Mitigations layered in this project:**
1. CoT scratchpad with explicit [GAPS] and [UNCERTAINTY] steps — forces gap-acknowledgment before writing.
2. Strong system prompt rules ("Use ONLY the provided context").
3. Per-sentence evidence grading — physically removes hallucinated sentences after generation.
4. Evaluation — even after grading, if faithfulness is still low, retry.

The grader is the most aggressive of these because it can't be ignored by the LLM — it runs *outside* the LLM that wrote the answer.

### 10.11 Provider rotation as a reliability pattern
Inspired by load balancer health-check fallbacks. Each provider is a backend; the "health check" is the catch-block keyword match; the policy is priority-with-fallback (not round-robin, because the free-tier tokens are uneven). This trades latency-on-failure (~200ms per skipped provider) for high availability.

### 10.12 Out-of-domain detection
A query whose answer relevancy is < 0.05 means the model's answer is essentially perpendicular in embedding space to the query. That happens when the paper has *nothing* related to ask, so the model produced something tangential. Retrying with query expansion can't fix this — it only makes the expanded query equally unrelated. Better to short-circuit and tell the user.

---

## 11. Sessions Done (history)

### Session 1 — Unified LLM Layer (critical fix)
- Fixed `chat_completion`: keyword-filtered exception handling so only rate-limit-shaped errors fall through.
- Added Gemini Flash as provider #1.
- Migrated `section_detector.py` and `multi_hop.py` off direct Groq client → `chat_completion`.
- Deleted dead `intent_detector.py` (replaced by `query_planner.py`).

### Session 2 — Dependencies + Storage (critical fix)
- Rewrote `requirements.txt` with the 12 actual direct deps.
- Added `threading.Lock()` to `storage.py` — fixed concurrent-upload corruption.
- Set chunk params to `512/100` (was `400/50`).

### Session 3 — Pipeline Efficiency
- Created `models.py` singleton — one embedding model instance, not two.
- Fixed double `evaluate_answer()` call in `pipeline.py` — reuse pre-eval when grading didn't change the answer (50% reduction in evaluation overhead).
- Unicode fix: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` so emoji/special-character LLM output doesn't crash on Windows cp1252.

### Session 4 — Evidence Grading UI
- Added `EvidenceGrading` component in `ChatPage.jsx` — cyan/amber/red chips with expandable breakdown.

### Session 5 — Table Extraction
- New `table_extractor.py` — pdfplumber `extract_tables()` → markdown chunks tagged `section_type="table"`.
- Plumbed `section_type` through `generator.py` sources → frontend violet table chips.

### Session 6 — Structured Logging + Request Tracing
- `api/logger.py` — `generate_request_id()` + `log_query()` structured line.
- `/query` is async with 60s timeout (504 on hang); `request_id` returned in response.

### Session 7 — Multi-paper Comparison
- `compare_retriever.py` — interleave chunks from two papers.
- `compare_papers()` in pipeline forces a comparison-shaped plan and answer_structure.
- `POST /query` accepts `paper_ids: [a, b]` (120s timeout).
- Frontend Compare Mode with A/B badges.

---

## 12. Anticipated Interview Questions — and direct answers

**Q: "Walk me through what happens when a user uploads a PDF."**  
A: Frontend `UploadPage` POSTs the file to `/api/upload`. FastAPI creates a registry row with `status=processing`, writes the PDF to disk under `data/papers/{id}.pdf`, schedules `run_ingestion` as a BackgroundTask, returns `paper_id` immediately. Ingestion runs in the background: pdfplumber parses with column detection, section detector finds candidate headings via 6-signal scoring + LLM batch confirmation, chunker produces 512-token chunks with 100-token overlap, table extractor adds markdown table chunks, embedder subprocess (DLL-isolated on Windows) calls `all-MiniLM-L6-v2` to vectorize all chunks and `upsert`s them into the paper's ChromaDB collection. When it finishes, registry status flips to `ready`. The frontend was polling `/api/status/{id}` every 2 seconds — the next poll sees `ready` and navigates to the chat page.

**Q: "How do you handle a query?"**  
A: `POST /query` runs `answer_query`. Stage 1: `plan_query` produces a JSON plan (answer_type, sub_questions, complexity, etc.) via one LLM call. Stage 2: router resolves retrieval config from answer_type, chooses hybrid_retrieve or multi_hop based on complexity. Stage 3: hybrid retrieval — BM25 + dense vector search, fused with RRF. Stage 4: CrossEncoder rerank, keep top `llm_k`. Stage 5: generator runs a 6-step CoT scratchpad ([INVENTORY], [GAPS], [INFERENCE], [UNCERTAINTY], [STRUCTURE], [WRITE]) then writes an ESSENCE + DETAIL answer. Stage 6: evidence grader classifies each sentence DIRECT/INFERRED/UNSUPPORTED and removes UNSUPPORTED. Stage 7: evaluator scores faithfulness + answer_relevancy locally via cosine similarity. Stage 8: if confidence < 50, diagnose failure (retrieval vs generation) and retry with different strategy. Returns the cleaned answer, confidence, per-sentence grades, sources, plan, and reasoning chain.

**Q: "Why hybrid retrieval?"**  
A: Lexical and semantic retrieval fail in opposite cases. BM25 misses paraphrased queries; dense vectors miss keyword-specific queries. Combining them via RRF catches both. RRF is robust to score-scale differences because it only uses ranks.

**Q: "Why not just use a bigger embedding model?"**  
A: MiniLM-L6 (22M params) is the bottom of the size/quality Pareto frontier — small enough to run on CPU at <100ms per query, accurate enough for academic English. Going to a 7B-param embedding model would 5-10x the latency and require a GPU for not-much-better retrieval. For our use case (~100 chunks per paper, single-host) it's the right trade.

**Q: "Why a subprocess for embedding?"**  
A: PyTorch (which sentence-transformers loads) and pdfplumber/Groq share Visual C++ runtime dependencies on Windows that conflict at the DLL level. In-process embedding caused intermittent segfaults. Isolating embedding to a fresh subprocess gives it a clean DLL space. The cost is one process spawn per ingestion — negligible vs the 30-60s of total ingestion time.

**Q: "Walk me through a query that fails on attempt 1."**  
A: The pipeline computes confidence. If it's below 50, `diagnose_failure` looks at the top reranker score and the faithfulness/relevancy split: low rerank score = retrieval problem (didn't find the right chunks); high rerank, low faithfulness = generation problem (the LLM hallucinated despite good chunks). Retry strategy: for retrieval failure, `_expand_query` rewrites the question with technical vocabulary that's more likely to match the paper's text; for generation failure, slice to fewer top chunks to force focus. Maximum 3 attempts. Best confidence across attempts wins.

**Q: "Why an evidence grader instead of a better generation prompt?"**  
A: Prompts can be ignored. The model can be told "Use only the context" and still hallucinate — instruction following is statistical, not guaranteed. The grader runs *outside* the LLM that wrote the answer and can physically remove sentences the LLM produced. Defense in depth: prompt as best-effort, grader as enforcement.

**Q: "What's the confidence threshold and why 50?"**  
A: 50 corresponds to faithfulness ≈ 0.5 and relevancy ≈ 0.5 (with 70/30 weighting). Empirically, below that level, the answer is more likely to be wrong than right. The threshold was calibrated against a labeled set of 20 questions (the Phase 3 step-2 calibration) — at 50, the precision/recall trade-off was best.

**Q: "Why do you weight faithfulness 0.7 and relevancy 0.3?"**  
A: Faithfulness is the safety property — an off-topic answer is annoying, but a fluent fabrication is dangerous. Weighting faithfulness higher means the system flags hallucinated-but-on-topic answers as low confidence.

**Q: "What's an out-of-domain query and how do you handle it?"**  
A: A query whose answer relevancy is below 0.05 — the model's answer is essentially orthogonal in embedding space to the query. That means the paper has nothing relevant to the question. Retrying with query expansion doesn't help (you can't expand a query into a domain that doesn't exist). We short-circuit, return the best answer we have, and attach a warning telling the user the question isn't related to the paper.

**Q: "Walk through your LLM failover."**  
A: `llm_client.py` has a `_PROVIDERS` list in priority order: Gemini, Cerebras, Mistral, Groq-1, Groq-2. Every LLM call iterates. If the call raises an exception whose stringified message contains "rate", "quota", "429", "limit", "exhausted", "not_found", or similar — fall through to the next provider. If it raises anything else (network error, auth error) — re-raise immediately so the user gets a real error instead of a silent retry that masks misconfiguration. If all providers are exhausted, raise `RuntimeError`.

**Q: "Why is Gemini the OpenAI client's base_url?"**  
A: Gemini exposes an OpenAI-compatible endpoint at `generativelanguage.googleapis.com/v1beta/openai/`. The OpenAI SDK accepts `base_url` — so we use one SDK with five different URLs. The cost of switching providers is one config-line edit.

**Q: "How do you prevent race conditions on `data/papers.json`?"**  
A: All read-modify-write operations are wrapped in a `threading.Lock()` declared at module level. Five concurrent uploads tested → five unique records, no corruption. Without the lock, concurrent loaders would each see an older state and overwrite each other.

**Q: "Why JSON file instead of SQLite?"**  
A: The registry is small (a few dozen rows max), schema is flat, and JSON is human-readable for debugging. SQLite would be overkill — at single-user scale a JSON file with a Python lock is sufficient. If this scaled to thousands of papers or multiple users I'd move to SQLite or Postgres.

**Q: "How do you handle very large PDFs?"**  
A: The pipeline scales linearly in chunks. A 50-page paper produces ~150 chunks at 512 tokens each → ~75k tokens of total content, ingested in 30-60s on a laptop CPU. There's no architectural limit; the practical limit is ChromaDB collection size, which is well beyond paper-scale.

**Q: "How do you make the retry deterministic?"**  
A: Each retry must *change something* from the previous attempt — that's the contract. Same query + same retrieval = same answer. So:
- Retrieval retries change the *query* (via LLM-driven expansion).
- Generation retries change the *chunks* (slice to top-K to force focus).

Deterministic-on-purpose-change.

**Q: "What's the biggest weakness?"**  
A: The eval set is small. I built the infrastructure (evidence grading, retry, hybrid retrieval) but haven't run it against a 200+ Q&A benchmark with ground-truth strings to compute exact-match / F1. The PLAN.md "future work" section calls this out — the next high-value step is integrating with FinanceBench (SEC 10-Ks with 150 public questions) or building a domain-specific eval set, then comparing PaperMind's confidence + correctness against vanilla RAG.

**Q: "What would you change with more time?"**  
A: (1) End-to-end eval harness with ground-truth string matching. (2) Citation verification — check that every kept sentence has a [Section: ...] tag, not just that unsupported ones are removed. (3) Streaming answer generation — the answer takes 5-8 seconds; streaming would make it feel instant. (4) Caching — same query against the same paper should hit a cache instead of re-running the LLM. (5) Move embedding to a real worker queue (Celery / RQ) for scalability.

**Q: "Why didn't you use LangChain?"**  
A: I wanted to know what every parameter does. LangChain's defaults work, but I couldn't have told you why chunk_size=1000 or why their retriever uses k=4 — those are just defaults someone picked. Writing the pieces myself meant every choice (512/100 chunks, RRF k=60, 0.7/0.3 confidence weighting) was something I actually measured or reasoned about. Plus I avoided a 400MB dependency tree for a project that fits in 2000 lines.

**Q: "What does the frontend tell the user that the backend can't?"**  
A: Three things:
1. **Per-sentence confidence visibility** — cyan/amber/red dots show which sentences were directly supported, inferred, or removed. The backend can return the grades; the UI makes them readable.
2. **ESSENCE/DETAIL separation** — collapsible accordion means the user gets the answer in one breath if they want, and the full breakdown if they need it.
3. **Source provenance** — every source chip links section + page; for tables it's labeled "TABLE"; for comparison it's labeled A or B. The user can verify any claim themselves.

**Q: "What's the most clever thing in the codebase?"**  
A: Two-tie:
1. The CoT scratchpad with `[WRITE]` as a delimiter — the model is free to write reasoning in any format, but the structured label lets us split reasoning from answer with one regex.
2. Hybrid two-column PDF reading — partitioning chars by column *first*, then grouping each into lines independently, then re-merging full-width lines that flow through the gutter. It's the only way to read research papers correctly and most parsers get it wrong.

**Q: "What's the single most important fix you made?"**  
A: The exception-handler bug in `llm_client.py`. The original had `continue` inside `except` without filtering on the error type, so it silently swallowed every error and tried the next provider — including auth errors, network errors, programmer errors. After the fix, only rate-limit-shaped errors trigger fallback; everything else surfaces to the user immediately so they can see what's actually wrong.

---

## 13. Failure Modes (and how the system degrades)

| Failure | What happens |
|---|---|
| All LLM providers rate-limited | `chat_completion` raises `RuntimeError`. The pipeline catches it per attempt, prints error, continues to next attempt. After 3 attempts, returns the graceful-degradation answer "Unable to answer this question from the provided paper." |
| Query plan JSON parse fails | `_fallback_plan` returns a minimal safe plan (factual, simple, sub_questions=[query]) and the pipeline continues. |
| Retrieval returns zero chunks | The pipeline's exception handler in the per-attempt try/except logs the error and falls through to retry. After 3 attempts, graceful-degradation answer. |
| Evidence grader's JSON parse fails | `grade_answer` returns `grading_failed: True` with the original answer unchanged — the answer is shown to the user without the cleaned-up sentences, but the pipeline doesn't crash. |
| Embedding subprocess crashes | `ingest_document` catches `CalledProcessError`, prints stdout/stderr, raises RuntimeError → status flipped to "failed". User sees error in upload UI. |
| LLM produces non-JSON when JSON was expected | Three places handle this: query_planner (`_fallback_plan`), multi_hop decompose (returns `[query]`), evidence_grader (returns original). Same pattern in each: try/except + safe default. |
| Concurrent uploads | Threading lock around the registry prevents corruption. Five tested → five unique records. |
| Pipeline timeout (>60s single, >120s compare) | `asyncio.wait_for` raises `TimeoutError` → HTTP 504. User sees "Query timed out". |
| Unicode in LLM output crashes Windows print | `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` at module load — no more `UnicodeEncodeError` on cp1252. |
| Deleted paper still referenced | `delete_paper` removes registry row, file, AND ChromaDB collection. If the collection was already gone (partial ingestion), the try/except passes silently. |

---

## 14. Files Inventory (every file, one line each)

### Backend
- `api/main.py` — FastAPI app, routes (/upload, /query, /papers, /status, /health), background ingestion, request_id tracing
- `api/storage.py` — papers.json registry with `threading.Lock()`, CRUD operations
- `api/logger.py` — `generate_request_id`, structured `log_query` one-liner
- `ingestion/pdf_parser.py` — pdfplumber-based PDF extraction with per-page two-column detection
- `ingestion/section_detector.py` — 6-signal scoring + LLM batch confirmation for section headings; assembles sections
- `ingestion/chunker.py` — tiktoken cl100k_base sliding window, 512 tokens, 100 overlap, per-section
- `ingestion/table_extractor.py` — pdfplumber `extract_tables` → markdown-formatted chunks
- `ingestion/embedder.py` — sentence-transformers + ChromaDB upsert + BM25 cache invalidation
- `ingestion/embedder_worker.py` — subprocess entry point for DLL-isolated embedding on Windows
- `ingestion/ingest_document.py` — orchestrates the 5-step ingestion
- `ingestion/models.py` — shared `get_embedding_model()` singleton
- `ingestion/retriever.py` — dense vector retrieval against ChromaDB
- `ingestion/bm25_retriever.py` — BM25Okapi with per-paper cache, invalidation hook
- `ingestion/hybrid_retriever.py` — runs BM25 + vector in parallel, RRF fusion, boost-term support
- `ingestion/reranker.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking
- `ingestion/query_planner.py` — single LLM call → JSON plan (answer_type, sub_questions, etc.)
- `ingestion/query_router.py` — resolves retrieval config from answer_type, branches on complexity, calls retriever + reranker
- `ingestion/multi_hop.py` — runs hybrid retrieval per sub-question, dedupes
- `ingestion/generator.py` — CoT 6-step scratchpad + ESSENCE/DETAIL prompt, answer_structure injection, chunk-ref scrubbing
- `ingestion/evidence_grader.py` — per-sentence LLM grader, removes UNSUPPORTED, returns enriched grades
- `ingestion/evaluator.py` — local embedding-similarity faithfulness + relevancy + confidence calc
- `ingestion/retry_engine.py` — `diagnose_failure` + `retry_query` strategies, query expansion
- `ingestion/compare_retriever.py` — parallel retrieval from two papers with A/B labels, interleaved
- `ingestion/pipeline.py` — `answer_query` (full pipeline with retry) + `compare_papers` (forced comparison plan)
- `ingestion/llm_client.py` — unified `chat_completion` with provider rotation, keyword-filtered fallback

### Frontend
- `frontend/index.html` — entry HTML, font preloads (Inter, Space Grotesk, JetBrains Mono)
- `frontend/vite.config.js` — Vite + React + Tailwind plugin, `/api` proxy to FastAPI
- `frontend/src/main.jsx` — React root render
- `frontend/src/App.jsx` — top-level state machine: UploadPage vs ChatPage
- `frontend/src/api.js` — fetch wrappers for /upload, /query, /papers, /status, /compare, /delete
- `frontend/src/pages/UploadPage.jsx` — drag-and-drop upload + 2s status polling
- `frontend/src/pages/ChatPage.jsx` — chat interface, MetricRing, EvidenceGrading, TelemetryPanel, SourceChip, workspace switcher, compare mode
- `frontend/src/index.css` — Tailwind + custom .pm-card / .cosmic-orb-* / .evidence-chip components

### Config + meta
- `requirements.txt` — 12 direct deps
- `.env` — API keys for Gemini, Cerebras, Mistral, Groq (gitignored)
- `data/papers.json` — registry
- `data/chroma_db/` — vector store
- `data/papers/` — uploaded PDFs
- `PLAN.md` — session-by-session implementation log
- `decisions.md` — architectural decisions log
- `progress.txt` — running development journal (also gitignored)

### Tests (kept around for reference; not part of the production path)
- `test_*.py` (parser, intent, evaluator, retry, hybrid, generator, pipeline, etc.) — single-purpose verification scripts written during development

---

## 15. The 60-Second Defense (memorize)

If someone gives you 60 seconds: **"PaperMind is a research-paper Q&A system that fixes the three failures of naive RAG. First, it plans the query — one LLM call produces a structured plan with answer type, key concepts, sub-questions, and answer structure. Second, it retrieves hybridly — BM25 for keywords plus dense vector search for semantics, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder. Third — and this is the part that actually fixes hallucination — it grades the generated answer sentence-by-sentence against the retrieved chunks and physically removes any sentence that's not supported. Then it evaluates itself with local embedding-similarity scoring, and if confidence is below 50 it diagnoses whether the failure was retrieval or generation and retries with a different strategy. Up to three attempts. The whole pipeline rotates across five LLM providers — Gemini, Cerebras, Mistral, two Groq keys — through one OpenAI-compatible client, so no provider outage crashes it. Stack: FastAPI, ChromaDB, sentence-transformers MiniLM-L6, rank-bm25, pdfplumber for column-aware extraction, React 19 with Tailwind for the UI."**

You now own this project. Defend it.

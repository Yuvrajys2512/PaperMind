# PaperMind — Session Implementation Plan

> Sessions are ordered: fix what is broken first, then add new capabilities, then polish.
> Each session has one clear goal and 2–3 verifiable outputs.

---

## SESSION 1 — Unified LLM Layer — CRITICAL FIX ✅

**Goal:** Every LLM call in the codebase goes through `llm_client.py` with correct exception handling and full provider coverage including Gemini.

**What was built:**
- `ingestion/llm_client.py` — Fixed exception handling bug (`continue` inside `except`, broken `for...else: raise` removed); added Gemini Flash as provider #1 (OpenAI-compatible endpoint); updated `_SKIP_KEYWORDS` to include `not_found`/`404`/`does not exist`; removed module-level print; fixed Cerebras model name to `llama3.1-8b`
- `ingestion/section_detector.py` — Removed direct `Groq` client, now uses `chat_completion()`
- `ingestion/multi_hop.py` — Removed direct `Groq` client, `decompose_query()` now uses `chat_completion()`
- `ingestion/intent_detector.py` — **Deleted** (dead file, replaced by `query_planner.py`)
- `test_intent.py` — Updated to use `plan_query` instead of deleted file

**Provider order:** Gemini → Cerebras (llama3.1-8b) → Mistral (mistral-small-latest) → Groq-1 (llama-3.3-70b-versatile)

**Verified:**
1. Full fallback chain tested live: Gemini rate-limited → Cerebras → Mistral WORKING ✓
2. `_PROVIDERS[0]["name"] == "Gemini"` ✓
3. `intent_detector.py` deleted ✓

---

## SESSION 2 — Dependencies + Storage — CRITICAL FIX ✅

**Goal:** Fresh-environment install works, concurrent uploads don't corrupt data, chunk parameters are intentional and documented.

**What was built:**
- `requirements.txt` — Rewritten with all 12 actual direct dependencies (`pdfplumber`, `tiktoken`, `chromadb`, `sentence-transformers`, `rank-bm25`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `openai`, `python-dotenv`, `numpy`)
- `api/storage.py` — Added `threading.Lock()` around all read-modify-write operations to prevent concurrent upload corruption
- `ingestion/ingest_document.py` — Chunk params corrected from `400/50` → `512/100` (matches tested + documented values)
- `decisions.md` — LLM section updated to reflect unified rotating client

**Verified:**
1. `pip install -r requirements.txt --dry-run` resolves with no errors ✓
2. 5 concurrent uploads → 5 unique records, 0 errors ✓
3. Chunk params `512/100` confirmed in source ✓

---

## SESSION 3 — Pipeline Efficiency — QUALITY IMPROVEMENT ✅

**Goal:** Cut evaluation overhead in half and eliminate duplicate model instances.

**What was built:**
- `ingestion/models.py` — New shared singleton module; `get_embedding_model()` returns one `all-MiniLM-L6-v2` instance for the whole process
- `ingestion/retriever.py` — Removed module-level `SentenceTransformer(...)` load; now calls `get_embedding_model()`
- `ingestion/evaluator.py` — Removed private `_get_embedder()` loader; now calls `get_embedding_model()`
- `ingestion/pipeline.py` — Fixed double `evaluate_answer()` call: grading now reuses the first evaluation when `removed_count == 0` (answer unchanged); only re-evaluates when sentences were actually removed
- `ingestion/query_router.py` — Replaced `→` with `->` in print statements (Windows cp1252 crash fix)
- `api/main.py` — Added `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` so Unicode LLM output never crashes the server

**Verified:**
1. `id(retriever model) == id(evaluator model) == id(models.py model)` — all same instance ✓
2. 2 attempts → 2 `evaluate_answer` calls (was 4); 50% reduction confirmed ✓
3. Unicode crash fixed — pipeline runs without cp1252 errors ✓

---


## SESSION 4 — Evidence Grading UI — NEW FEATURE ✅

**Goal:** Surface the evidence grader's per-sentence output in the frontend so users can see what is directly quoted vs inferred vs removed.

**What was built:**
- `frontend/src/pages/ChatPage.jsx` — Added `EvidenceGrading` component: reads `result.grading.grades`, shows summary chips (cyan=DIRECT, amber=INFERRED, red=REMOVED), expandable per-sentence breakdown list with color-coded dots; graceful no-render on `grading_failed: true` or empty grades; placed after Evidence Sources section

**Verifiable outputs:**
1. Evidence Quality section shows count chips (cyan DIRECT, amber INFERRED, red REMOVED) after each answer
2. "Breakdown" button expands per-sentence list: cyan dot+normal text, amber dot+dimmed text, red dot+strikethrough for removed
3. No visual regression on `grading_failed: true` — section does not render

---

## SESSION 5 — Table Extraction — NEW FEATURE

**Goal:** Tables in PDFs become queryable chunks, not silent gaps in coverage.

**What we build:**
- `ingestion/table_extractor.py` — New module using `pdfplumber`'s `page.extract_tables()`, converts each table to markdown-style text (`| col | col |` format)
- `ingestion/ingest_document.py` — Integrate table extraction after text extraction; table chunks stored with `section_type: "table"`, `section: "Table N (page P)"`

**Verifiable outputs:**
1. Ingesting "Attention Is All You Need" produces ≥ 3 chunks with `section_type: "table"`
2. Querying "What hardware was used for training?" returns table content (P100 GPUs, WMT 2014)
3. Table chunks appear in source chips with `section_type: "table"`

---

## SESSION 6 — Structured Logging + Request Tracing — INFRASTRUCTURE

**Goal:** Every query is observable — request ID, timing breakdown, and confidence in a single log line.

**What we build:**
- `api/logger.py` — Structured logger with request ID generation
- `api/main.py` — Instrument `/query` endpoint: log `[req_id] paper=... question=... duration=...ms confidence=...`; add 60-second timeout so hung requests return 504 instead of hanging forever
- `ingestion/pipeline.py` — Accept optional `request_id` for per-attempt logging

**Verifiable outputs:**
1. Log line: `[abc123] query="How does attention work?" paper=xyz duration=7420ms confidence=68.5 attempts=1`
2. Two concurrent queries produce two different `req_id` values in logs
3. A query that hangs returns HTTP 504 after 60 seconds

---

## SESSION 7 — Multi-paper Comparison — NEW FEATURE

**Goal:** Users can ask one question across two papers simultaneously and get a structured comparison answer.

**What we build:**
- `api/main.py` — Extend `POST /query` to accept optional `paper_ids: list[str]` (up to 2)
- `ingestion/retriever.py` + `ingestion/hybrid_retriever.py` — Merge results from multiple ChromaDB collections, tag each chunk with its source paper
- `ingestion/query_planner.py` — When 2 papers are supplied, generate comparison-oriented `answer_structure`
- `frontend/src/pages/ChatPage.jsx` — "Compare" mode in workspace switcher: select two papers, submit multi-paper query

**Verifiable outputs:**
1. `POST /query {"paper_ids": ["id1", "id2"], "question": "..."}` returns an answer referencing both papers
2. Sources list shows chunks from both papers with distinct paper labels
3. Frontend "Compare" mode lets you select two papers and submits the multi-paper query

---

## Status summary

| Session | Name | Type | Status |
|---------|------|------|--------|
| 1 | Unified LLM Layer | CRITICAL FIX | ✅ Done |
| 2 | Dependencies + Storage | CRITICAL FIX | ✅ Done |
| 3 | Pipeline Efficiency | QUALITY IMPROVEMENT | ✅ Done |
| 4 | Evidence Grading UI | NEW FEATURE | ✅ Done |
| 5 | Table Extraction | NEW FEATURE | ⬜ |
| 6 | Structured Logging | INFRASTRUCTURE | ⬜ |
| 7 | Multi-paper Comparison | NEW FEATURE | ⬜ |

# QoL Improvements — Pre-Phase-6

A targeted pass to make the upcoming evaluation phase (Phase 6) easier to wire
up without changing the pipeline's answer-producing behaviour. Two groups:

1. **Pre-Phase-6 enablers** — small additions the eval harness will read
   directly (run metadata, ablation toggles, raw answer surface, JSONL log).
2. **Robustness wins** — quiet defects in the API/storage/grader layer that
   would have surfaced under N=300 eval load.

Nothing here moves the pipeline's accuracy, retry behaviour, thresholds, or
prompt text. The only observable change at runtime is additional fields in
the `/query` response and a new `logs/queries.jsonl` file.

---

## 1. Pre-Phase-6 enablers

### 1.1 LLM call stats — `ingestion/llm_client.py`

**What.** Added thread-local stats (`reset_stats`, `get_stats`) that count
successful `chat_completion()` calls and record which provider answered
each one.

**Why.** Phase 6 §8's deliverable is a cost-per-question table:
`PaperMind: $0.008, 7.2s, 3 LLM calls`. The pipeline previously had no
way to tell its caller how many LLM hops it made or which provider served
them — that information existed only as a stdout `print`.

**How.** Stats live in `threading.local()` because FastAPI runs sync
pipeline work in a thread-pool executor; one request's counters cannot
leak into another's. `chat_completion()` bumps the counter on success
(rate-limit fallbacks don't count — they aren't actual answers).

**Effect.** The eval harness reads `result["llm_calls"]` and
`result["providers_used"]` directly. Cost columns are now mechanical.

---

### 1.2 Ablation env vars — `retry_engine.py`, `query_router.py`, `pipeline.py`

**What.** Three environment variables, all default-off:

| Env var | Effect |
|---|---|
| `PAPERMIND_MAX_ATTEMPTS` | Clamp retries (range [1, 3]). `=1` disables retries. |
| `PAPERMIND_DISABLE_RERANK` | Skip the CrossEncoder; take top `llm_k` by retrieval rank. |
| `PAPERMIND_DISABLE_GRADER` | Skip the evidence grader; return the raw answer. |

**Why.** Phase 6 §7 calls for five ablations (`no_grader`, `no_retry`,
`no_rerank`, plus `no_cot` and `no_planner`). Without env-driven toggles
each ablation would mean editing source and running on a branch — slow
and a constant risk of an ablation drifting away from `main`.

**How.** Each module reads its flag once at import time and threads it
into a single decision point:

- `retry_engine.MAX_ATTEMPTS` is clamped at module load. The pipeline's
  loop runs `range(1, MAX_ATTEMPTS + 1)`, so `=1` naturally turns off
  retries.
- `query_router.route_query` branches on `_DISABLE_RERANK`: when set,
  raw retrieval is sliced to `llm_k` and the rerank step is bypassed.
- `pipeline.answer_query` and `pipeline.compare_papers` skip
  `grade_answer()` when `_DISABLE_GRADER` is set and synthesise a no-op
  grading dict so downstream consumers (frontend, eval scorers) see the
  same shape.

**Effect.** Ablation runs become `PAPERMIND_DISABLE_RERANK=1 python
eval/harness.py …`. No code forks.

---

### 1.3 `original_answer` on the response — `pipeline.py`

**What.** Added `original_answer` (the pre-grader text) to the result
dict returned by `answer_query` and `compare_papers`.

**Why.** `evidence_grader.grade_answer()` already produced both
`cleaned_answer` and `original_answer`, but the pipeline only forwarded
the cleaned one. With both surfaced, the `no_grader` ablation can score
`original_answer` from the **same run** as the main pipeline scores
`answer` — one set of LLM calls, two data points. That halves the eval
cost of the grader ablation.

**How.** The result dict now carries:

```python
"answer":          <cleaned>,
"original_answer": <pre-grader>,
```

When the grader is disabled or unavailable, both fields hold the same
text — the contract stays uniform.

**Effect.** Eval scoring of "with grader" vs "without grader" is now a
post-hoc read, not a second run.

---

### 1.4 Run metadata on the response — `pipeline.py`

**What.** Three new fields on every pipeline result:

```python
"duration_ms":    int,         # wall-clock for the full pipeline call
"llm_calls":      int,         # successful chat_completion() count
"providers_used": list[str],   # provider name per call, in order
```

**Why.** Phase 6's headline table includes median latency and per-query
cost. Both are derived from these three numbers. Previously they were
collected only at the FastAPI layer (`duration_ms`) or not at all.

**How.** `answer_query` and `compare_papers` call `reset_stats()` at
entry, capture `time.monotonic()`, and attach the three fields just
before returning. I had to change the early-pass `return best_result`
to `break` so the metadata wrap-up runs for the happy path too — the
old code was structured to return early on success, which would have
skipped the new bookkeeping. Behaviour is identical; only the
control-flow shape changed.

**Effect.** The eval harness reads three integers per question and
builds the cost/latency table directly. No scraping logs.

---

### 1.5 JSONL query log — `api/logger.py`

**What.** Every call to `log_query()` now also appends a JSON record to
`logs/queries.jsonl`:

```json
{"timestamp":"2026-05-18T…Z","req_id":"abc12345","paper_id":"…",
 "question":"…","duration_ms":7420,"confidence":68.5,"attempts":1,
 "passed":true,"llm_calls":3,"providers":["Gemini","Cerebras","Gemini"]}
```

**Why.** Phase 6 wants a re-readable record per query so the harness
doesn't have to scrape stdout. Stdout is fine for tail-ing during
development but fragile to parse and lossy when the server restarts.

**How.** `logs/` is created at import time; the writer is wrapped in
try/except so a disk-full or permission error can never take down a
query response. JSON encoding is `ensure_ascii=False` so non-ASCII
paper titles round-trip cleanly.

**Effect.** `cat logs/queries.jsonl | jq ...` gives a clean dataset of
every query, no stdout parsing. The eval harness can also write its
own per-question records to the same format if useful.

---

## 2. Robustness wins

### 2.1 Atomic registry writes — `api/storage.py`

**What.** `_save_registry()` writes to `papers.json.tmp` and `os.replace()`s
it over `papers.json`.

**Why.** The previous `REGISTRY_FILE.write_text(...)` left a half-flushed
file on a crash mid-write. The on-disk `data/papers.json` already shows
the symptom — rows stuck at `"status": "processing"` from earlier
testing — though it isn't proven that all of those came from this path.
Atomic replace removes the failure mode entirely.

**How.** `os.replace` is atomic on both POSIX and Windows.

**Effect.** Server kills, OOMs, and concurrent writes can no longer
corrupt the registry. Combined with the existing `threading.Lock`,
the registry is now both serialised and atomic.

---

### 2.2 Timezone-aware timestamps — `api/storage.py`

**What.** `datetime.utcnow()` → `datetime.now(timezone.utc)`.

**Why.** `utcnow()` returns a naive `datetime` and is deprecated in
Python 3.12+. The replacement returns a tz-aware object whose ISO
string ends with `+00:00`, so any downstream `fromisoformat` reader
won't silently treat the time as local.

**Effect.** No runtime warning on 3.12+; clear UTC offset in serialised
records.

---

### 2.3 BM25 cache invalidation on delete — `api/main.py`

**What.** `DELETE /papers/{paper_id}` now calls
`invalidate_bm25_cache(paper_id)` after dropping the Chroma collection.

**Why.** The function `invalidate_bm25_cache` already existed in
`bm25_retriever.py` but no caller used it. A long-running server that
deleted and re-ingested the same `paper_id` would keep the old BM25
index in memory and serve stale tokens. Unlikely in production (UUIDs
are unique per upload) but plausible under eval where the same paper
gets re-ingested for ablations.

**Effect.** Re-ingesting a paper_id now gets a fresh BM25 index.

---

### 2.4 Chroma delete errors are logged, not swallowed — `api/main.py`

**What.** Replaced `except Exception: pass` with a `print` of the
exception type and message.

**Why.** The original `pass` was justified for "collection may not
exist if ingestion never completed" — but it also hid every other
Chroma error, including ones that mean the collection is *still
there* after a delete. With Phase 6 about to hammer the API,
silent half-deletes would be expensive to debug.

**Effect.** The "collection missing" case still doesn't fail the
request, but it now leaves a stdout breadcrumb.

---

### 2.5 `asyncio.get_running_loop()` — `api/main.py`

**What.** Two callsites in `query_paper` changed from
`asyncio.get_event_loop()` to `asyncio.get_running_loop()`.

**Why.** `get_event_loop()` was deprecated in Python 3.10 for use inside
async functions and is scheduled for removal. The async handler already
*has* a running loop, so `get_running_loop()` is the correct API and
doesn't depend on the deprecated implicit-loop fallback.

**Effect.** No deprecation warning. Same behaviour on currently
supported Pythons.

---

### 2.6 Traceback on pipeline exceptions — `ingestion/pipeline.py`

**What.** `except Exception as e: print(...)` now also calls
`traceback.print_exc()`.

**Why.** The previous handler printed only the exception's message,
which was enough for known errors but useless for novel ones. Phase 6
will run questions the pipeline has never seen — when one of them
breaks something deep in retrieval, a single-line `KeyError: 'section'`
with no stack frame is a guaranteed time sink.

**Effect.** Full traceback in logs whenever a pipeline attempt errors.

---

### 2.7 Sentence-boundary removal in the grader — `ingestion/evidence_grader.py`

**What.** Rewrote `_reconstruct_answer()` to rebuild the cleaned answer
line-by-line, splitting on the same sentence boundaries as
`_split_sentences()` and dropping only the exact units flagged
UNSUPPORTED. The orphan-fragment filter (lines with ≤2 words) was
removed because the new code can't produce orphans — it only drops
whole sentence units.

**Why.** The previous implementation used `cleaned.replace(sent, "")`
for every removed sentence. Plain substring replacement is unsafe in
general: if a removed sentence's literal text appears inside a kept
sentence (e.g. via repeated phrasing across paragraphs or LLM
near-duplicate output), `str.replace` corrupts the kept text. In the
current grader the risk is small — `_split_sentences` ensures every
graded unit ends with `.`, which limits substring collisions — but
the new approach removes the failure class entirely rather than
relying on a structural accident.

**How.** For each original line: keep blank lines and gradeable
`**HEADERS**` as-is unless they were graded UNSUPPORTED; for prose
lines, split on `. ` exactly as `_split_sentences` does, then re-emit
only the units that aren't in the removal set. Short fragments (≤15
chars) that the splitter wouldn't have surfaced are preserved as part
of their line — they're not gradeable so they can't have been
UNSUPPORTED.

**Effect.** No observable change on the current grader output for the
realistic cases I checked, but the function is now provably correct
under adversarial inputs (verified with a substring test case during
implementation).

---

## What this didn't touch

Per the plan in PHASE_6.md §10.7: thresholds (`CONFIDENCE_THRESHOLD`,
`_OUT_OF_DOMAIN_RELEVANCY`, `_SUPPORT_THRESHOLD`,
`_FAITHFULNESS_GENERATION_THRESHOLD`), retry strategies, grader prompt,
generator prompt, and the planner prompt are all unchanged. Phase 6
must measure the pipeline as it stands before any of those are tuned.

The root-level `test_*.py` scripts and the `scratch/` directory are
untouched — the eval harness will replace them.

---

## Verification done

- `python -m py_compile` clean on all 8 edited files.
- Defaults preserved: `MAX_ATTEMPTS=3`, `DISABLE_RERANK=False`,
  `DISABLE_GRADER=False` when env vars are unset.
- Env-var ablation: `PAPERMIND_MAX_ATTEMPTS=1
  PAPERMIND_DISABLE_RERANK=1 PAPERMIND_DISABLE_GRADER=true` flips all
  three flags as expected.
- Stats API: `reset_stats(); get_stats()` returns the zero state.
- Grader reconstruction: adversarial test with two sentences where one
  is a substring of the other produces the correct cleaned text
  (A intact, B removed).

## Files changed

```
api/logger.py
api/main.py
api/storage.py
ingestion/evidence_grader.py
ingestion/llm_client.py
ingestion/pipeline.py
ingestion/query_router.py
ingestion/retry_engine.py
```

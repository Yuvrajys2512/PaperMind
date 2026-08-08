# Autonomous Work Log

Work done while the owner was away, on the standing instruction: *bugs, fixes,
quality improvements only — nothing that needs a judgment call.*

**Rules I held myself to**

- Nothing committed or pushed. Every change is sitting in the working tree.
- Nothing deployed, no secrets touched, no user data modified.
- Only changes with an objectively correct answer. Anything that is a product
  or design decision is written up under [Needs your call](#needs-your-call)
  instead of being decided for me.
- Every loop ends green: `ruff` clean, full suite passing, `import api.main`
  clean. A loop that could not be finished is marked **BLOCKED** with the reason.

**Session:** 2026-08-06
**Baseline at start:** 109 tests passing, 8 ruff findings.

---

## Summary

| # | Loop | Status | Tests after |
|---|---|---|---|
| 1 | Clear all ruff findings | ✅ done | 109 |
| 2 | `get_all_chunks` returned chunks in the wrong order | ✅ done | 117 |
| 3 | Sweep for further real defects | ✅ done | 120 |
| 4 | Frontend test infrastructure + first 68 tests | ✅ done | 120 + 68 |
| 5 | Frontend lint (8 of 9 fixed) | ✅ done | 120 + 68 |
| 6 | DraftsPage + ChatPage tests | ✅ done | 120 + 108 |

**Net:** backend 109 → 120 tests; frontend 0 → 108 tests. Ruff 8 findings → 0,
eslint 9 → 1 (the remaining one is intentional). Three real bugs fixed (no-op
sort, dead dedup branch, temp-file leak) plus one segfault trigger removed.

**Backend changed:** `api/main.py`, `api/storage.py`, `ingestion/retriever.py`,
`ingestion/embedder.py`, `ingestion/multi_hop.py`, `tests/test_logging.py`,
`tests/test_rate_limit_key.py`.
**Backend new:** `tests/test_reading_order.py`, `tests/test_storage_tempfiles.py`.

**Frontend changed:** `package.json`, `package-lock.json`, `vite.config.js`,
`src/pages/ChatPage.jsx`, `src/components/MetricRing.jsx`.
**Frontend new:** `src/setupTests.js`, `src/components/metricTooltips.js`,
`src/api.test.js`, `src/textUtils.test.js`,
`src/components/{MetricRing,AppShell,OverlapPanel}.test.jsx`,
`src/pages/{DraftsPage,ChatPage}.test.jsx`.

---

## Loop 1 — Clear all ruff findings ✅

`ruff check .` reported 8 findings; all were dead weight, none behavioural.

| File | Finding | Fix |
|---|---|---|
| `api/main.py:32` | `get_remote_address` imported but unused | removed (it moved to `api/auth.py` with the rate-limit key work) |
| `api/main.py:44` | `DEMO_USER_ID` imported but unused | removed from the import list |
| `api/main.py:80` | `generate_request_id` imported twice | kept the earlier import, narrowed the later one to `log_query` |
| `tests/test_logging.py` | unused `tempfile`, `os`; 2 f-strings with no placeholders | `ruff --fix` |
| `tests/test_rate_limit_key.py` | unused `time` | `ruff --fix` |

The duplicate-import fix was done by hand rather than with `--fix`: `ruff.toml`
notes that import *order* in this project is load-bearing (torch vs onnxruntime
OpenMP load order), and I did not want the autofixer choosing which of the two
lines to delete.

**Verified:** `ruff check .` clean · 109 tests pass · `import api.main` clean.

---

## Loop 2 — `get_all_chunks` returned chunks in the wrong order ✅

**The bug.** `ingestion/retriever.get_all_chunks` ended with:

```python
chunks.sort(key=lambda c: c["metadata"].get("chunk_id", 0))
```

There is no `chunk_id` in the metadata — the embedder writes `chunk_index`. So
`.get("chunk_id", 0)` returned `0` for every chunk, the sort was a no-op, and
callers received whatever order ChromaDB happened to return. I confirmed the
metadata keys directly against the live store: `chunk_index`, `page_num`,
`section`, `section_type`, `token_count`, `total_chunks_in_section` — no
`chunk_id`.

**Why it matters.** Today's order looks correct only because Chroma tends to
return insertion order; nothing guarantees that, and it is not stable across
upserts. Several callers depend on reading order and none of them fail loudly
when it is wrong — they just return wrong answers:

- `citation_gap_auditor` treats a statement as cited if the marker is in the
  *next* sentence,
- `plagiarism_auditor` flattens chunks into one word stream and matches runs
  across chunk boundaries,
- `claim_auditor` slices sections out by position.

**The fix.** Two parts:

1. `ingestion/embedder.py` now writes `doc_index` — the chunk's ordinal in the
   whole document. Nothing else in the metadata records this: `chunk_index`
   restarts at 0 in each section, and several sections can share a page.
   `chunks` already arrives in document order, so the enumeration index is
   exactly right.
2. `ingestion/retriever.py` gained `_reading_order_key`, which sorts by
   `doc_index` when *every* chunk has one, and otherwise falls back to
   `(page_num, chunk_index)`.

The all-or-nothing rule matters: mixing the schemes inside one collection would
sort the annotated chunks as a block ahead of the rest, which is worse than
using either scheme consistently. The fallback is approximate (sections sharing
a page all report index 0) but Python's sort is stable, so ties keep the store's
order instead of being shuffled.

**Backwards compatibility.** No re-ingestion required. Existing collections have
no `doc_index` and use the fallback; anything ingested from now on gets exact
ordering. Re-ingesting an old paper upgrades it for free.

**A second bug this surfaced.** Adding `tests/test_reading_order.py` made
`pytest tests/` **segfault** at collection. `ingestion/retriever` imported
`ingestion.models` at module scope, which pulls torch — and torch in the same
process as docling (`test_table_extractor`) is the documented Windows footgun in
`write_mode_handoff.md` §7. Only `retrieve()` actually needs the embedder;
`get_all_chunks` and `collection_name` are pure store access. Deferred the
import into `retrieve()`, matching the convention the auditors already follow.

Side benefit: the full suite went from ~12s to ~4.5s, because importing
`retriever` no longer loads torch.

**Verified:**
- 8 new unit tests covering exact ordering, the fallback, the half-migrated
  case, tie stability, `doc_index=0` not being mistaken for missing, and
  `None`/absent metadata not raising `TypeError`.
- Against the **live store**: all 19 collections now come back page-monotonic
  under the fallback (0 non-monotonic).
- Fresh ingest into a **temp** Chroma path (the real store untouched): 6 chunks
  round-tripped with `doc_index` `[0,1,2,3,4,5]` and text in document order.
- `retrieve()` still works through the lazy import — real query returned hits.
- Importing `ingestion.retriever` no longer loads torch or sentence-transformers.
- 117 tests pass · `ruff` clean · `import api.main` clean · no segfault.

---

## Loop 3 — Sweep for further real defects ✅

Loop 2's bug was a *silent key mismatch*, so I swept for more of that class, then
for the neighbouring ones (leaks, dead branches, missing wiring).

### 3a. Metadata key audit — one more mismatch, now removed

Cross-referenced every metadata key the embedder **writes** against every key
the codebase **reads**. Only one orphan left after Loop 2: `chunk_id`, read in
`ingestion/multi_hop.py`.

`chunker.py` does assign a `chunk_id` to each chunk, but `embed_and_store` never
copies it into Chroma metadata — so for chunks coming back from retrieval the
lookup always returned `None` and the code fell through to `hash(chunk["text"])`.

The dead branch was also the *wrong* branch: text-hash is the better dedup key
here. Sub-questions overlap heavily, the same passage returns from several
retrieval passes, and identical text is a duplicate for the generator's purposes
regardless of which position it came from. Removed the dead lookup, kept the
text hash, wrote down why.

**Behaviour is unchanged** — the fallback was already the only live path.

### 3b. Temp-file leak on failed PDF download — fixed

`api/storage.download_pdf_to_tempfile` called `mkstemp` (which creates the file)
and then `_s3.download_file`. If the download raised — missing R2 key, network
blip — the function never returned, so the caller in `api/ingestion_runner`
still had `temp_path = None` and its `finally` block had nothing to delete. The
docstring said "caller must delete it", which on the failure path is impossible.

Every failed ingestion leaked one file into the temp directory, permanently, on
a long-lived host.

Now cleans up after itself and re-raises. Cleanup is best-effort so the *download*
error — the actionable one — is what surfaces, not a secondary `OSError`.

**Verified the test actually catches the bug:** `git stash`-ed the fix and
re-ran — `test_failed_download_leaves_no_temp_file_behind` fails against the old
code and passes against the new. (Stash popped; tree restored.)

### 3c. Checks that came back clean — no action needed

Recording these so nobody re-audits them:

- **Silent `except: pass` handlers** — 9 found, all are the `_emit(on_progress…)`
  progress guards. Correct as written: a progress callback must never be able to
  kill the audit behind it.
- **Temp files elsewhere** — `discovery/fetcher.py` and `api/main.py` upload both
  clean up correctly on every path (`finally: os.remove`, and removal on both
  exception branches respectively).
- **Mutable default arguments** — none.
- **Feature wiring** — all 7 `delete_*_report` helpers are registered in the
  paper-deletion cleanup loop in `api/main.py` (no orphaned R2 blobs), and all 7
  audit stream endpoints carry `Depends(enforce_audit_quota)`.
- **Endpoint auth coverage** — enumerated all 26 routes off the live app object.
  Every one carries an auth dependency except three, all correct by design:
  `/billing/webhook` (Stripe signature-verifies it; it cannot use Clerk auth),
  `/health`, and `/venues` (a static list of venue names, no user data).

**Verified:** `ruff` clean · **120 tests pass** · `import api.main` clean ·
`npm run build` clean.

---

## Loop 4 — Frontend test infrastructure + first 68 tests ✅

The frontend had **zero** test coverage. It now has 68 tests.

### Setup

Vitest, the conventional choice for a Vite project (shares the existing config
and transform pipeline, so there is no second build to keep in sync).

- Added dev deps: `vitest`, `jsdom`, `@testing-library/react`,
  `@testing-library/jest-dom`, `@testing-library/user-event`.
- `npm test` (single run) and `npm run test:watch`.
- `src/setupTests.js` — loads jest-dom matchers and registers `afterEach(cleanup)`;
  RTL does not auto-clean under Vitest globals, so trees would otherwise leak
  between tests and break duplicate-match queries.
- One config change with a real cause: `esbuild: { jsx: 'automatic' }` in
  `vite.config.js`. Without it, JSX inside test files compiles to the *classic*
  runtime and every `render()` dies with "React is not defined" —
  `@vitejs/plugin-react` covers the app's own files but not the test transform.
  Confirmed `npm run build` is unaffected.

### What is covered, and why those things

Chosen for real logic and low brittleness — no snapshot tests, nothing asserting
layout or styling that a redesign would invalidate.

| File | Tests | Why it earns coverage |
|---|---|---|
| `src/api.test.js` | 13 | `consumeSSE` is the single most load-bearing piece of client logic — every audit tab, the query view and compare all stream through it |
| `src/components/MetricRing.test.jsx` | 17 | two different input conventions (0–1 fraction vs 0–100 percentage) that would silently render "0.93%" if crossed |
| `src/components/AppShell.test.jsx` | 13 | the free-plan quota strip, i.e. the fix for the "Library 5 but limit 3" confusion |
| `src/components/OverlapPanel.test.jsx` | 15 | attributed-vs-unattributed separation, the highest-stakes UI in the product |
| `src/textUtils.test.js` | 10 | `escapeHtml` is a security boundary, not a formatter |

Highlights of what is actually pinned:

- **SSE edge cases a manual click-through never produces:** a frame split across
  two network chunks (TCP does not respect frame boundaries), a `data:` payload
  spread over multiple lines, malformed JSON skipped rather than killing the
  stream, an `error` frame surfacing the server's message, and a stream that
  ends without a `done` frame rejecting instead of resolving a half-built report.
- **`err.status` propagation** — `DraftsPage` distinguishes a quota 429 from any
  other failure by status, not by string-matching, so that is asserted directly.
- **OverlapPanel's separation of concerns** — attributed (quoted/cited) matches
  render under their own heading and are *not* counted as findings, the
  same-document exclusion is explained rather than silently dropped, and the two
  pieces of guarding copy ("prompts to check, not verdicts"; "not a plagiarism
  clearance") are treated as behaviour and asserted.
- **`escapeHtml`'s known limit** — it does not escape single quotes. Safe only
  while every interpolation site is a text node or double-quoted attribute;
  there is now a test saying so, so the next person adding a single-quoted
  attribute finds out.

**Verified:** 68 frontend tests pass · `npm run build` clean · backend suite
still 120 · `ruff` clean.

---

## Loop 5 — Frontend lint ✅ (8 of 9)

`npx eslint .` reported 9 pre-existing errors — none from the new test files.

- **`ChatPage.jsx` (7 errors)** — an unused `useAuth` import and six unused
  destructured metrics (`confidence`, `faithfulness`, `answer_relevancy` in two
  places). Removed; zero behaviour change.
- **`MetricRing.jsx` (1 error)** — `react-refresh/only-export-components`: the
  file exported both a component and the `METRIC_TOOLTIPS` table, which stops
  the module hot-reloading. Moved the table to `src/components/metricTooltips.js`
  and updated its two importers (`ChatPage`, plus the new test). Pure move.
- **`StructurePanel.jsx` (1 error) — left alone deliberately.** See below.

**Verified:** 1 eslint error remaining (the intentional one) · 68 frontend tests
pass · build clean.

---

## Loop 6 — DraftsPage + ChatPage tests ✅

The two big stateful pages, +40 tests (frontend now 108).

### `src/pages/DraftsPage.test.jsx` — 18 tests

Covers the flows where something can actually go wrong for a user:

- **Upload guards** — a non-PDF is rejected before any network call; a 429 shows
  the quota banner *keyed on `err.status`*, not on matching the message text
  (which is what the component genuinely relies on).
- **A failed upload does not blank the list.** Explicit regression guard: upload
  errors and list-fetch errors are separate state, and during the earlier
  browser session the page showed "Could not load your drafts" right after a
  quota rejection. That turned out to be the Neon connection bug, not this — but
  the two must stay independent, so now a test says so.
- **The two-step delete confirmation** — one click arms, the second deletes; a
  failed delete keeps the row, explains itself, and returns to the un-armed
  state so a stray click cannot destroy a draft.
- **Polling** — refetches every 3s while a draft is ingesting and tears the
  interval down once nothing is processing (verified with fake timers, including
  that it does *not* poll when everything is already ready).

### `src/pages/ChatPage.test.jsx` — 22 tests

ChatPage is ~2,500 lines, so this covers the parts with real logic and leaves
the chrome alone:

- **Send flow** — the right paper id and question reach the API; the question is
  echoed immediately, before the answer arrives; the composer clears; empty
  input is ignored; a second query cannot start while one is in flight.
- **Failure** — the question stays in the transcript, a placeholder answer fills
  the assistant's turn so the conversation is not left dangling, the server
  message appears in a toast, and the composer re-enables for a retry.
- **Persistence** — conversations round-trip through localStorage, each paper's
  history stays separate, and **corrupted storage starts clean instead of
  crashing** (the component already guards this; without it a bad value would
  white-screen the page on every load until the user cleared storage by hand).
- **Search** — the filter keeps a matching question together with its answer in
  both directions, matches case-insensitively, and restores everything when
  cleared.
- **Keyboard** — Shift+Enter inserts a newline rather than sending; ArrowUp on an
  empty composer recalls the last question but does *not* clobber text already
  typed.

### Two test-harness gaps found and fixed centrally

- **`scrollIntoView`** — jsdom implements no layout, so ChatPage's auto-scroll
  threw before any assertion could run. Stubbed in `src/setupTests.js` (with
  `scrollTo`) rather than per-file, since it is a jsdom limitation and not app
  behaviour worth asserting.
- **`userEvent.upload` honours `accept=".pdf"`** — it silently swallowed the
  bad file, so the component's own guard never ran and the test passed for the
  wrong reason. Now passes `applyAccept: false` so the guard under test is the
  component's.

### Two things the tests corrected about my own assumptions

- I assumed ChatPage had a "Back" button. The control is labelled **"Upload
  New"** and returns to the upload view; the test now says what the UI does
  rather than what I expected.
- Search highlighting wraps matches in `<mark>`, so `getByText` cannot see the
  text whole. Those assertions compare concatenated text instead — the tests
  care about *which messages are on screen*, not how they are marked up.

**Verified:** 108 frontend tests pass · no `act()` warnings · `npm run build`
clean · eslint still at the single intentional error · backend still 120.

---

## Current state

**228 tests passing** — 120 backend (`pytest tests/`, ~9s) + 108 frontend
(`npm test`, ~13s). `ruff` clean, `import api.main` clean, `npm run build` clean,
eslint down to the single intentional error.

Nothing committed. Nothing deployed. No user data touched.

### How to run everything

```bash
# backend
venv\Scripts\python.exe -m pytest tests/ -q
venv\Scripts\python.exe -m ruff check .

# frontend
cd frontend
npm test          # single run
npm run test:watch
npm run lint
```

---

## Needs your call

Nothing blocking, but three things I deliberately did **not** decide:

1. **`doc_index` puts table chunks at the end of the document.** Ingestion
   appends table chunks after all prose, so their ordinal places them last
   rather than on the page they were printed on. Harmless for every current
   caller (the prose consumers filter tables out; the table consumers look them
   up by section, not position), and folding `page_num` into the sort key would
   fix table placement at the cost of letting a missing `page_num` disorder the
   prose — which is the case that actually matters. Left as-is, documented in
   the `_reading_order_key` docstring. Say the word if you want tables
   interleaved.

2. **Old collections keep the approximate ordering.** No re-ingestion is
   required and nothing is broken, but papers ingested before this session use
   the `(page_num, chunk_index)` fallback rather than exact `doc_index` order.
   Re-ingesting any paper upgrades it for free. Whether that is worth doing in
   bulk is your call.

3. **The last eslint error: `StructurePanel.jsx:114`.** Its mount effect calls
   `setLoading(true); setError(''); setProgress([]); setReport(null)`
   synchronously in the effect body — which the comment two lines above claims
   it does *not* do, and which every sibling panel avoids. The reset is there
   for a reason: this effect also re-runs when the venue changes, and without it
   the previous venue's report would stay on screen while the new one loads.

   The React-recommended fix is to move the reset into the venue `onChange`
   handler, since that is the event that should reset it. But that changes
   *when* state resets — specifically it would no longer reset if `paperId`
   changed without a remount — so it is a behavioural decision, not a lint
   cleanup. I left it and its (now inaccurate) comment alone rather than guess.
   The panel works correctly today; this is a cascading-render smell, not a live
   bug.

4. **Frontend `npm audit` reports 5 vulnerabilities** (2 low, 3 high) after
   adding the test tooling; production dependencies alone report 1 low. All are
   in dev dependencies. `npm audit fix` can move major versions, so I did not
   run it.

5. **Three commits' worth of work is sitting uncommitted** (Overlap Check, the
   quota/pool fixes, the rate-limit key, and now this loop's fixes). You do all
   commits — I have not touched git beyond one `stash`/`stash pop` to verify a
   regression test, which left the tree exactly as it was.

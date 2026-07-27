# PaperMind — "Write Mode" Handoff

**Last updated:** 2026-07-25
**Status:** Write Mode is **feature-complete** — all 6 audit tabs built, browser-verified, working. The queued feature menu is exhausted. The next move is a decision, not a build (see §6).

---

## 0. Prompt for the next Claude session

> Copy everything in the block below into a fresh session. It's written to be self-sufficient — the rest of this file is the detail it refers to.

```
I'm working on PaperMind at C:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind.

Read write_mode_handoff.md in the repo root first — it's the full state of the
"Write Mode" track and it's current as of 2026-07-25.

Short version: Write Mode (upload your own unpublished draft -> pre-submission
audits) is FEATURE-COMPLETE. All 6 tabs are built and browser-verified:
Weakness Review, Claim Audit, Novelty Scan, Venue Fit, Numbers Check, and
Citation Gaps. Tabs 1-5 are committed (through 14b53cf). The 6th (Citation
Gaps) is code-complete, browser-verified, and UNCOMMITTED in my working tree.

Before doing anything, two things I need from you:

1. There is a real open decision documented in section 6 of that file: my own
   stated priority was "deploy first, then research, don't interleave", but six
   feature builds have happened instead of finishing the deploy. Surface that
   decision to me and let me choose the track. Do NOT silently pick one.

2. Tell me exactly what's sitting uncommitted and what you'd suggest the commit
   split should be. I do all git commits myself — never run git commit, and
   never add Co-Authored-By or any AI attribution to a commit message.

Repo conventions that will bite you if you don't know them (all detailed in
sections 3 and 7 of the handoff):
- Windows/import-order footgun: import torch/auditors BEFORE docling or
  api/ingestion_runner, or the process segfaults (exit 139) at import time.
  api/main.py is only safe because of its import ordering. After touching
  imports there, verify with: venv\Scripts\python.exe -c "import api.main"
- No react-router. Page nav is hand-rolled useState('page') in App.jsx.
- No migration framework. Schema changes are idempotent ALTER TABLE ... IF NOT
  EXISTS inside each module's _ensure_schema(), run at import time.
- Run the backend WITHOUT --reload (known stale-process bug on this machine —
  two processes end up bound to :8000). Kill and restart manually.
- Chrome automation: screenshot resolution != real viewport (~1.96x off), so
  pixel-coordinate clicks land wrong. Use read_page(filter: interactive) to get
  element refs and click by ref.

Don't touch the deploy/hardening track (final_day.md, product/DEPLOYMENT.md,
product/LAUNCH_CHECKLIST.md) unless I pick that track.
```

---

## 1. Current state at a glance

| Tab | Engine | External dep | Committed | Browser-verified |
|---|---|---|---|---|
| 1. Weakness Review | `ingestion/reviewer_auditor.py` | — | ✅ | ✅ 2026-07-23 |
| 2. Claim Audit | `ingestion/claim_auditor.py` | — | ✅ | ✅ (incidental) |
| 3. Novelty Scan | `ingestion/novelty_scout.py` | Semantic Scholar | ✅ | ✅ 2026-07-25 |
| 4. Venue Fit | `ingestion/structure_auditor.py` | — | ✅ | ✅ 2026-07-23 |
| 5. Numbers Check | `ingestion/numbers_auditor.py` | — | ✅ | ✅ 2026-07-23 |
| 6. Citation Gaps | `ingestion/citation_gap_auditor.py` | Semantic Scholar | ❌ **uncommitted** | ✅ 2026-07-25 |

**Commits already on `main`:** `2dfd9e4` (My Drafts + tabs 1–2), `bbafa63` (tabs 3–5 + landing/home redesign), `14b53cf` (numbers_auditor false-positive fix).

**Uncommitted working tree right now:**

```
?? ingestion/citation_gap_auditor.py            <- new, feature 6
?? frontend/src/components/CitationGapPanel.jsx <- new, feature 6
 M api/main.py                                  <- feature 6 endpoint + imports
 M api/storage.py                               <- feature 6 R2 helpers
 M frontend/src/api.js                          <- citationGapStream()
 M frontend/src/components/MetricRing.jsx       <- rose accent
 M frontend/src/pages/DraftReviewPage.jsx       <- 6th tab
 M write_mode_handoff.md                        <- this file
 M ingestion/embedder_worker.py                 <- NOT MINE, see below
```

`ingestion/embedder_worker.py` is a **stray whitespace-only change** (2 blank lines added) that predates this work. It's unrelated to Write Mode — discard it or commit it separately, don't bundle it.

**Suggested commit split:** everything except `embedder_worker.py` is one coherent "Citation Gap Check" commit.

---

## 2. Why this exists (product decisions — don't re-litigate without asking)

The owner's college mentor pushed for PaperMind to grow beyond "Q&A over already-published papers" into helping people *while they're writing*. Agreed direction: a separate **write mode** where a researcher uploads their own unpublished draft and gets pre-submission audits.

Decisions locked in before any code was written:

1. **Build order** — reuse-based audits first, external-search features after. (Followed; all done now.)
2. **UX shape** — a new top-level section ("My Drafts"), not a toggle bolted onto the existing paper-chat view.
3. **Quota** — drafts share the existing `max_papers` / `max_audits_per_month` pools. Deliberate "ship fastest" call; splitting later is cheap because the `papers.paper_type` column already exists. Every audit tab records usage as `kind="audit"` against one shared pool.
4. **Every audit is a thinking prompt, not a verdict.** No tab ever asserts the author is wrong — it surfaces evidence and lets them judge. This is why every engine is aggressively conservative (§3).

Original plan doc: `C:\Users\Yuvraj Srivastava\.claude\plans\proud-riding-wave.md`.

---

## 3. Architecture — the 4-layer clone pattern

**This is the single most useful thing in this file.** Every one of the six tabs is the same four layers. A 7th would be too.

### Layer 1 — `ingestion/<name>_auditor.py`

One public entry point: `def audit_x(paper_id: str, on_progress=None) -> dict`.

- **Never raises.** Any hard failure returns an `_empty_report(paper_id, failed=True, reason=...)` dict so the endpoint and frontend always get one consistent shape.
- Pure-sync (no async), so it drops into `run_in_executor` with no event-loop-in-thread hazard. This is why `novelty_scout` and `citation_gap_auditor` each carry a **synchronous** Semantic Scholar client rather than reusing the async `discovery/sources/semantic_scholar.py`.
- Progress via `_emit(on_progress, stage, message)` — the stage keys must match the frontend panel's stage checklist.
- Reads only from the per-paper Chroma collection via `get_all_chunks(paper_id)` / `hybrid_retrieve(...)`. **No Postgres, no venue, no publication-status coupling.** This is what made write mode cheap to build — keep it that way.

### Layer 2 — `api/storage.py`

Three R2 helpers per feature, cloned verbatim:

```python
def upload_x_report(paper_id, report): ...   # put_object, JSON blob
def get_x_report(paper_id) -> dict | None:   # NoSuchKey / any error -> None (cache miss)
def delete_x_report(paper_id): ...
```

Single key per feature: `{paper_id}.<name>.json`. **The delete MUST be added to the cleanup loop in `api/main.py`** (search for `delete_novelty_report`) or you leak orphaned R2 objects on paper deletion.

### Layer 3 — `api/main.py` SSE endpoint

`POST /papers/{paper_id}/<name>/stream?force=`. Clone the nearest sibling:
- external-API-backed, no venue param → clone **novelty scan**
- purely local → clone **claim audit**
- venue-parameterized → clone **structure check** (its cache is served only when `cached["venue"] == requested`)

Non-negotiable pieces:
- `user_id: str = Depends(enforce_audit_quota)` and `record_usage(kind="audit", ...)` — shared pool, never a new quota bucket.
- Cache-first serve when `force=False` and a cached report exists (no LLM, no usage recorded).
- `with paper_locked(paper_id):` + `reset_stats()` / `get_stats()` **inside the executor function** — those stats are thread-local, so they must run in the same thread as the LLM work or the usage numbers belong to the wrong request.
- `asyncio.wait_for(..., timeout=180.0)` for external-API paths.
- Only cache + record usage when the report didn't hard-fail.

### Layer 4 — Frontend

- `frontend/src/components/<Name>Panel.jsx` — clone the closest existing panel. Each has a distinct accent (see §4). Initial load is an inline `useEffect` with a `cancelled` flag (so no `setState` fires synchronously in the effect body); the re-run button calls a separate `useCallback`.
- `frontend/src/components/MetricRing.jsx` — add the accent case if you use a new colour (currently cyan / violet / amber / emerald / rose).
- `frontend/src/api.js` — `xStream(paperId, onEvent, force = false)`, same `consumeSSE` pattern.
- `frontend/src/pages/DraftReviewPage.jsx` — register the tab in the header row, the empty-state row, and the render switch (three places).

### The house conservatism rule

**A false accusation is the only real failure mode** — it damages trust in PaperMind, not in the paper. So every engine applies *deterministic downgrades after the LLM answers*, always one-way toward "no flag":

| Engine | Downgrade |
|---|---|
| `claim_auditor` | flag without a resolvable cited chunk → `GROUNDED` |
| `numbers_auditor` | `MISMATCH` without both a cited chunk and a `found_value` → `NOT_FOUND`; numerically-equal claimed/found → `MATCH` |
| `structure_auditor` | `MISSING` when a matching section exists → `THIN` |
| `citation_gap_auditor` | marker in the *next* sentence → OK; first-person contribution → OK; flag with no reason → OK; unknown verdict → OK |

If you add a 7th tab, it needs its own downgrade path. Don't ship one without it.

---

## 4. The six tabs — reference

### Tab 1 — Weakness Review (`reviewer_auditor.py`, accent violet `#a78bfa`)
Grades the draft against methodological norms a reviewer would check (baselines, ablations, error bars, N, threats to validity, related work). Coverage-first: section map + per-component `hybrid_retrieve`.

### Tab 2 — Claim Audit (`claim_auditor.py`, accent cyan `#00f5ff`)
Extracts falsifiable claims from abstract/intro/conclusion, retrieves the draft's *own* evidence (biased toward results + linearized table cells), and grades each: `GROUNDED` / `OVERCLAIM` / `SCOPE_MISMATCH` / `UNVERIFIABLE`. Table linearization (`"BERT — SQuAD F1: 0.79"`) is what lets it judge effect size rather than topical overlap.

### Tab 3 — Novelty Scan (`novelty_scout.py`, accent fuchsia `#f472b6`)
The only tab that reasons *across* the literature. Pipeline: pull abstract/intro → 1 LLM call distils 2–3 search queries → sync S2 search per query → merge/dedup → batched LLM rates each neighbour `HIGH`/`MEDIUM`/`LOW` closeness + overlap + differentiator → 1 LLM synthesis for a positioning read.

Deliberately **no local SPECTER2/torch** — avoids the docling/torch import-order segfault and keeps cold-starts cheap on free hosts. S2 relevance + LLM re-rank is v1; a SPECTER2 embedding re-rank is a clean phase-2 add.

Never declares work novel or not novel — it only shows the closest prior work and lets the author judge.

### Tab 4 — Venue Fit (`structure_auditor.py`, accent amber `#fbbf24`)
Presence/completeness check, not a quality judgement: does the draft *contain* the components a venue expects? Component registry (`related_work`, `method`, `experiments`, `limitations`, `ethics`, `broader_impacts`, `reproducibility`, `conclusion`) composed per venue (`generic`/`neurips`/`icml`/`iclr`/`acl`/`emnlp`/`cvpr`) with a `required` flag.

Verdicts `PRESENT`/`THIN`/`MISSING`; **severity is deterministic** from `(required, verdict)` — a required component that's MISSING drives the headline `required_missing` count. The venue lives *inside* the single cached blob, and the cache is served only on a venue match (keeps deletion to one key). `GET /venues` feeds the selector.

### Tab 5 — Numbers Check (`numbers_auditor.py`, accent emerald `#34d399`)
Reconciles *exact figures*: the abstract says "92.4 F1 on SQuAD" — does a results table show 92.4? Catches stale numbers, transcription slips (28.4 → 82.4), abstract↔results drift. Verdicts `MATCH`/`MISMATCH`/`NOT_FOUND`.

`consistency_score = matched / (matched + mismatched)` — `NOT_FOUND` is excluded from the denominator because retrieval may simply have missed it, which isn't the paper's fault.

**Bug fixed 2026-07-23 (`14b53cf`):** the model flagged `MISMATCH` on numerically-identical values (claim "3.5" vs found "3.5 days") by second-guessing dataset attribution — contradicting its own prompt's "different dataset ≠ mismatch" rule. `_numeric_value()` now parses both sides (handles `K`/`M`/`B`, `%`, commas) and downgrades to `MATCH` within 0.5% relative tolerance.

### Tab 6 — Citation Gaps (`citation_gap_auditor.py`, accent rose `#fb7185`) — NEW, uncommitted

Mirror image of the Claim Audit. Claim Audit asks "does the draft's own evidence back its own claims?"; this asks the **outward** question: where does the draft assert something about the world — prior work, an external statistic, an attributed method, a comparison — with nothing pointing at a source? Those are the sentences a reviewer marks "[citation needed]".

**Pipeline** (stages: `reading` → `scanning` → `classifying` → `searching`):
1. Gather prose body chunks.
2. **Deterministic regex pass** drops every sentence that already carries a citation marker (`[12]`, `[3, 4]`, `[5-8]`, `(Smith et al., 2020)`, `[Devlin, 2019]`, `Vaswani et al. (2017)`). This is the primary precision guard — cheap, high-recall on markers, and it means the LLM only ever sees genuinely uncited sentences.
3. Batched LLM calls classify the survivors as `NEEDS_CITATION` (kinds: `prior_work` / `external_stat` / `attributed_method` / `comparison`) or `OK`, and emit search keywords for the flagged ones.
4. Bounded S2 lookups for the strongest gaps, then **one batched LLM pass must return `SUPPORTS`** before a suggested reference is shown at all.

**Three design calls worth knowing:**

- **Did NOT reuse `claim_auditor.extract_claims`.** It extracts claims about the paper's *own* contribution — the exact opposite scope. Filtering its output would have started from the wrong set. Only the pipeline *shape* was cloned.
- **Section filtering is a deny-list**, not an allow-list. An allow-list of expected section names (`introduction`, `related work`, `method`, …) was tried first and silently scanned **2 of 26 chunks** on a real paper, because real headings are numbered and idiosyncratic ("3 Model Architecture", "4 Why Self-Attention"). The deny-list (`reference`/`bibliograph`/`acknowledg`/`appendix`/`abstract` + tables) took coverage from 20 sentences to 145. The abstract stays excluded on purpose — venues discourage citations there, so scanning it manufactures false positives.
- **Self-match guard.** On the very first live run the tool suggested the author cite **their own paper** — S2 returned the draft itself. Now any candidate whose abstract overlaps the draft's content-word fingerprint ≥ 0.65 is dropped. Verified against a live S2 response: drops "Attention is All you Need", keeps topically-adjacent papers like "Self-Attention with Relative Position Representations".

**Budget:** 4 S2 searches + 1 verification LLM call per run (novelty scan does 3 searches + 2 rating calls — same ballpark). `MAX_CANDIDATES` is round-robined across sections so the cap isn't spent entirely in the introduction.

**Coverage metric** (`coverage_score`, the "sourced" ring): `cited / (cited + gaps)` — of the statements that appear to need a source, how many already carry a citation. Sentences the model cleared as common knowledge are excluded from both sides, since they never needed one.

**Verification done:**
- `import api.main` boots clean, route registered (torch/docling order OK); `npm run build` clean.
- Offline unit tests on the deterministic layer: marker detection (6 cited / 3 uncited forms), abbreviation-safe sentence split, all four downgrade guards, self-match guard (drops the draft, keeps unrelated, no-ops on thin text). All pass.
- **Browser click-through** on the "Attention is all you need" draft: streams all four stages → 145 sentences scanned, 32 cited, 32 checked → 1 gap. That gap is a correct catch — the uncited *"outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU"* in §6.1. Evidence button opens the PDF at p.8 with the sentence highlighted. Cached re-serve renders with the "cached · re-run to refresh" badge. Console clean.

**Caveat on output quality:** the only draft in My Drafts is a published, heavily-cited paper — a good *negative* control (97% sourced, one legitimate flag), but not a real test of a sparsely-cited draft. **Worth running against an actual early-stage draft before fully trusting the flag rate.**

---

## 5. Known-good local dev runbook

```powershell
# Backend — NO --reload (see gotcha in §7)
cd "C:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind"
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend; npm run dev        # :5173

# Import sanity after touching api/main.py imports
.\venv\Scripts\python.exe -c "import api.main"
```

**Audit quota during testing.** The dev user's audit quota (10/month, shared by all six tabs) will exhaust fast when testing. Every limit in `api/usage.py`'s `TIER_LIMITS` reads from env, so raise it for a local session without touching the DB or code:

```powershell
$env:PAPERMIND_FREE_MAX_AUDITS_PER_MONTH = "500"
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

A plain restart drops back to the real limit of 10. **The backend was last left running with this override.**

---

## 6. What's left — and the one open decision

### The decision (owner's call, do not resolve silently)

The project's stated priority was **"deploy first, then research — don't interleave."** Six write-mode features have shipped instead of finishing the deploy. That memory predates the write-mode ask, so it's genuinely ambiguous whether write mode counted as "research" or as a product feature that jumps the queue. This has been carried unresolved across several sessions. **Surface it and let the owner choose.**

The three candidate tracks:

1. **Deploy** — `final_day.md` (backend → HF Spaces, frontend → Vercel). Everything §1–§5 of the launch checklist is code-complete; only manual deploy/ops steps remain.
2. **QASPER eval / paper** — the research track, currently blocked on throughput.
3. **More write mode** — the queued menu is empty; anything here is net-new scope.

### Concrete leftovers regardless of track

- **Commit the Citation Gap work** (§1 has the file list and suggested split). Owner does all commits.
- **`SEMANTIC_SCHOLAR_API_KEY` must be added to HF Space production secrets at deploy time.** It's in local `.env` and working, but two tabs (Novelty Scan, Citation Gaps) degrade to an empty state without it in prod. Manual, owner's-end, at deploy time — per `project_launch_checklist_deployment` env tables.
- **Test Citation Gaps against a genuinely sparsely-cited draft** (see caveat in §4).

### Deferred / phase-2 ideas (not commitments)

- SPECTER2 embedding re-rank for Novelty Scan (would reintroduce torch into that import path — see §7).
- Splitting drafts into their own quota pool (schema groundwork already exists via `paper_type`).

---

## 7. Repo conventions and gotchas

**These are the landmines. Read before editing.**

- **⚠️ torch/docling import order (Windows).** Import `torch` and the auditors **before** `docling` or `api/ingestion_runner`, or the process segfaults (exit 139) at import time. `api/main.py` is only safe today because of its careful import ordering. After touching imports there, always verify with `venv\Scripts\python.exe -c "import api.main"`. This is why `reviewer_auditor` and `structure_auditor` do lazy retriever imports, and why Novelty Scan avoids local embedding models.

- **⚠️ `uvicorn --reload` is broken on this machine.** After a file edit the reloader can leave *two* processes bound to :8000 (one stale, one fresh), so requests hit either one and you get inconsistent results. Run without `--reload` and restart manually. Don't trust a 200 alone — confirm exactly one LISTENING PID on :8000.

- **⚠️ Chrome automation coordinates.** Screenshots come back at a different resolution than the real viewport (e.g. 1568x745 screenshot vs 3072x1459 actual, ~1.96x). Pixel-coordinate clicks computed from a screenshot land in the wrong place. Use `read_page(filter: interactive)` to get element refs and click **by ref**.

- **No react-router.** All page navigation is hand-rolled `useState('page')` in `App.jsx`. Follow that pattern.

- **No migration framework.** Schema changes are idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` inside each module's `_ensure_schema()`, run at import time. Don't introduce Alembic.

- **Audit engines take only a `paper_id`** and read a per-paper Chroma collection. No Postgres/venue/publication-status coupling. This is what made write mode cheap — preserve it.

- **`.env` has spaces around `=`** (`KEY = value`). python-dotenv parses this fine; don't "fix" it.

- **Git:** the owner does all commits and pushes. Never run `git commit`. Never add `Co-Authored-By` or any AI attribution to a commit message.

- **Deploy track is separate.** `final_day.md`, `product/DEPLOYMENT.md`, `product/LAUNCH_CHECKLIST.md` — don't merge those concerns into this document or into write-mode commits.

---

## 8. File map

```
ingestion/
  reviewer_auditor.py       tab 1
  claim_auditor.py          tab 2
  novelty_scout.py          tab 3   (sync S2 client lives here)
  structure_auditor.py      tab 4   (+ list_venues())
  numbers_auditor.py        tab 5   (+ _numeric_value() guard)
  citation_gap_auditor.py   tab 6   NEW, uncommitted
api/
  main.py                   6 SSE endpoints + GET /venues + delete-cleanup loop
  storage.py                paper_type column + 6 x (upload/get/delete)_*_report
  usage.py                  TIER_LIMITS (all env-overridable), enforce_audit_quota
frontend/src/
  api.js                    6 *Stream() fns + getVenues()
  pages/DraftsPage.jsx      "My Drafts" list + upload
  pages/DraftReviewPage.jsx 6-tab switcher
  components/
    ReviewPanel.jsx  AuditPanel.jsx  NoveltyPanel.jsx
    StructurePanel.jsx  NumbersPanel.jsx  CitationGapPanel.jsx   <- new
    MetricRing.jsx          accents: cyan/violet/amber/emerald/rose
    PDFPreviewPanel.jsx     shared evidence viewer (page + highlight)
```

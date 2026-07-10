# PaperMind — "Write Mode" Handoff

**For:** whoever (human or Claude) picks this up next.
**State as of this handoff:** Feature 1 below is code-complete and locally verified (backend import + frontend build), but **not yet click-tested in a browser, not committed, not deployed.**

---

## 1. Why this exists

The user's college mentor pushed for PaperMind to grow beyond "Q&A over already-published papers" into also helping people while they're *writing* a paper. Agreed direction: a separate **write mode** for researchers, starting with letting them upload their own unpublished draft and get a pre-submission review.

Two product decisions were made with the user before any code was written — don't re-litigate these without asking:
1. **Build order**: reuse-based pre-submission audits first, *before* novelty-search or citation-gap-check (see §3 for why, and what's queued after).
2. **UX shape**: a new top-level section ("My Drafts" / "write mode"), not a toggle bolted onto the existing paper-chat view.

Full plan (already executed) is at `C:\Users\Yuvraj Srivastava\.claude\plans\proud-riding-wave.md` if you want the original reasoning/verification detail.

---

## 2. What's done — Feature 1: pre-submission audits in "My Drafts"

Repurposed the two audit engines PaperMind already had (`ingestion/reviewer_auditor.py`, `ingestion/claim_auditor.py`) to run on a user's own unpublished draft instead of a published paper. Both were confirmed (by reading the code) to depend on nothing but a `paper_id` — no "published" flag anywhere — so **zero changes were needed to the audit logic itself.** This was a product/UX build, not new AI work.

### Backend
- `api/storage.py`: added `papers.paper_type` column (`'paper'` default, `'draft'` for drafts), idempotent migration — verified running clean against the live dev Neon DB. `create_paper_record()` and `list_papers()` got an optional `paper_type` param, backward-compatible with every existing caller.
- `api/main.py`: `POST /upload` accepts an optional `paper_type` form field (default `'paper'`); `GET /papers` accepts an optional `?paper_type=` query param (default `'paper'`, so existing behavior is unchanged for every current caller).
- `/papers/{id}/audit/stream` and `/papers/{id}/review/stream` — **untouched**, confirmed to work on any `paper_id` regardless of type.
- Quota: drafts currently share the existing `max_papers`/`max_audits_per_month` pools (zero new quota logic). This was a deliberate "ship fastest" call — splitting into separate pools later is cheap (the schema groundwork is already there) if free users start feeling squeezed by mixing drafts and reference papers in one cap.

### Frontend
- Extracted `AuditPanel`, `ReviewPanel`, `MetricRing`, `PDFPreviewPanel`, `escapeHtml` out of the 3300-line `ChatPage.jsx` into standalone files (`frontend/src/components/*.jsx`, `frontend/src/textUtils.js`) so they can be reused outside chat. This was a pure mechanical extraction — `ChatPage.jsx`'s existing Audit/Review behavior should be byte-identical, not a redesign.
- New pages: `frontend/src/pages/DraftsPage.jsx` (upload + list your drafts, modeled on `LibraryPage.jsx`) and `frontend/src/pages/DraftReviewPage.jsx` (tab switcher between Weakness Review / Claim Audit, hosting the extracted panels as primary content instead of overlay drawers).
- Wired into `App.jsx` (`page === 'drafts' | 'draftReview'`, same hand-rolled page-state pattern as every other page — there's no react-router in this app) and a new "Drafts" nav button in `UploadPage.jsx`.

### Verified so far
- `python -c "import api.storage"` — schema migration ran clean against the real dev DB.
- `python -c "import api.main"` — app boots, `/upload` and `/papers` routes present.
- `npm run build` in `frontend/` — passes clean, no errors.

### NOT yet done — pick up here
1. **Manual click-through** (nobody has opened this in a browser yet):
   - Run backend + `cd frontend && npm run dev`.
   - Upload → Drafts nav → drag a PDF in → watch `processing → ready` → click Review → confirm both tabs (Weakness Review, Claim Audit) stream and render → click "view evidence" on a flagged item → confirm the PDF preview opens with highlight.
   - Regression-check: open an **existing normal paper** in Chat, run Audit and Review from there too — this is the actual risk of the extraction refactor, confirm nothing broke (metric ring tooltips, PDF preview from a normal chat source-chip, Markdown/PDF export).
   - Confirm a draft does **not** leak into Library or Discover.
   - Delete a draft, confirm full cleanup (Postgres row, R2 PDF, Chroma collection, cached reports) via the existing unmodified delete route.
2. **Nothing is committed.** Current uncommitted files (new + modified) as of this handoff:
   ```
   M  api/main.py
   M  api/storage.py
   M  frontend/src/App.jsx
   M  frontend/src/api.js
   M  frontend/src/pages/ChatPage.jsx
   M  frontend/src/pages/UploadPage.jsx
   ?? frontend/src/components/  (MetricRing.jsx, PDFPreviewPanel.jsx, AuditPanel.jsx, ReviewPanel.jsx)
   ?? frontend/src/pages/DraftsPage.jsx
   ?? frontend/src/pages/DraftReviewPage.jsx
   ?? frontend/src/textUtils.js
   ```
   **The user commits everything themselves — do not run `git commit` unless explicitly asked.** Also note `eval/analyze_grader.py`, `eval/run_eval.py`, `ingestion/evaluator.py`, `ingestion/llm_client.py`, `ingestion/pipeline.py`, `research/to_do.md` show as modified too — those are pre-existing, unrelated in-flight QASPER-eval work from before this session started. Don't touch or bundle them into a drafts-feature commit; they belong in their own commit(s) on the eval track.
3. **Not deployed.** This is local-only code. Deployment is a separate, already-documented track — see `final_day.md` at repo root (backend → HF Spaces, frontend → Vercel). **Open question for the user, not yours to decide silently:** the project's own priority order (per memory) was "deploy first, then research — don't interleave," and this write-mode work happened instead of finishing deploy. Surface that explicitly next session rather than assuming which one continues.

---

## 3. What's queued next

Ranked by leverage vs. effort, as discussed with the user:

1. **Novelty / related-work search** — ✅ **code-complete 2026-07-10** (see §5). Searches *across* the literature via Semantic Scholar for the prior work closest to a draft.
2. **Citation gap check** — flag claims in the draft with no nearby citation. Smaller, but most useful paired with #1 (needs a corpus to cross-reference against). *Its useful form needs Semantic Scholar too — so it's a poor "while waiting for the key" pick; the S2 client + LLM rating scaffolding from #1 is reusable when it's built.*
3. **Venue-fit / structure check** — ✅ **code-complete 2026-07-10** (see §6).
4. **Abstract/intro/results consistency check** — ✅ **shipped as the "Numbers Check" (reshaped)**, code-complete 2026-07-10 (see §7). Reshaped away from the Claim-Audit overlap into a sharp numbers-only reconciliation (abstract figures vs results tables).

Only #2 (citation-gap) remains, and it's key-blocked like novelty — treat as a starting menu, not a committed roadmap.

---

## 5. Feature #1 — Novelty / related-work scan (code-complete 2026-07-10)

A third tab ("Novelty Scan") in the draft reviewer. Unlike the two audits (which reason over the draft's *own* Chroma collection), this searches *across the literature* via Semantic Scholar to surface the prior work closest to the draft, rates each neighbour for closeness, and synthesizes a positioning read. This was the one genuinely-new piece (a cross-literature external integration), but it reused far more than expected — the app already had a Semantic Scholar client (`discovery/sources/semantic_scholar.py`) and the whole audit/SSE/cache/quota rig.

### Backend
- **`ingestion/novelty_scout.py`** (new) — `find_related_work(paper_id, on_progress) -> dict`, same never-raises contract as `audit_paper`/`review_paper`. Pipeline: pull abstract/intro from Chroma → 1 LLM call distils 2–3 search queries → **synchronous** S2 search (a sync sibling of the async discovery client, so it drops into `run_in_executor` with no event-loop-in-thread hazard) → merge/dedup candidates → batched LLM call rates each for closeness (HIGH/MEDIUM/LOW) + overlap + differentiator → 1 LLM synthesis for an overall positioning read. Deliberately **no local SPECTER2/torch** (avoids the docling/torch import-order segfault footgun + heavy cold-starts); S2 relevance + LLM re-rank is v1, SPECTER2 embedding re-rank is a clean phase-2 add.
- **`api/storage.py`** — `_novelty_key` + `upload/get/delete_novelty_report` (clone of the audit-report R2 helpers); `delete_novelty_report` wired into the paper-delete cleanup loop.
- **`api/main.py`** — `POST /papers/{id}/novelty/stream`, a near-exact clone of `audit_paper_stream` (cache-serve → `run_in_executor` → SSE), guarded by `enforce_audit_quota`, usage recorded as `kind="audit"` (shares the audit pool, consistent with the v1 quota decision).

### Frontend
- **`frontend/src/components/NoveltyPanel.jsx`** (new) — clone of `AuditPanel`'s SSE/staged-progress shell; renders the related-work list (closeness badge, out-links to real papers, overlap/"your angle" lines) + a synthesis card with positioning bullets. Accent fuchsia `#f472b6` (distinct from review-violet / audit-cyan).
- **`frontend/src/api.js`** — `noveltyScanStream(paperId, onEvent, force)`.
- **`frontend/src/pages/DraftReviewPage.jsx`** — third "Novelty Scan" tab.

### Verified
- `import ingestion.novelty_scout` clean; `import api.main` boots with the route wired (via `venv/`).
- `npm run build` passes clean.
- Live S2 request path exercised: request is well-formed, **429 (rate-limit) handling works** (retry-once → graceful empty state, no crash).

### NOT done / operational flags — pick up here
1. **⚠️ Needs a `SEMANTIC_SCHOLAR_API_KEY` to be reliable (manual, user's end).** The code already reads it (`api/storage`-style env pattern; falls back to unauthenticated). Right now no key is set, and S2's *unauthenticated* public pool 429s constantly (~1 req/sec shared globally) — so without a key the scan will usually show the "no related papers / rate-limited" empty state. Free key: https://www.semanticscholar.org/product/api → add `SEMANTIC_SCHOLAR_API_KEY=...` to `.env`. (Same key also makes the existing Discover feature reliable.)
2. **Parse path not yet exercised against a live 200** (only the 429 branch, because the pool was throttling during the build). The parse block mirrors the proven `discovery/sources/semantic_scholar.py`, so it's trusted by parity — but confirm on first real run with a key.
3. **No browser click-through yet** — same as Feature 1. Run backend + `npm run dev`, open a draft → Novelty Scan tab, confirm it streams stages and renders the list.
4. **Not committed.** New: `ingestion/novelty_scout.py`, `frontend/src/components/NoveltyPanel.jsx`. Modified: `api/main.py`, `api/storage.py`, `frontend/src/api.js`, `frontend/src/pages/DraftReviewPage.jsx`. (User commits everything themselves.)

---

## 4. Quick orientation for whoever picks this up

- Full original plan + verification detail: `C:\Users\Yuvraj Srivastava\.claude\plans\proud-riding-wave.md`.
- This app has **no react-router** — all page navigation is hand-rolled `useState('page')` in `App.jsx`. Follow that pattern for anything new.
- Both audit engines (`ingestion/claim_auditor.py`, `ingestion/reviewer_auditor.py`) take only a `paper_id` and read from a per-paper Chroma collection — no Postgres/venue/publication-status coupling. Keep it that way; it's what made this feature cheap.
- No migration framework exists — schema changes are idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` statements inside each module's `_ensure_schema()`, run at import time. Follow that pattern, don't introduce Alembic or similar.
- Deploy checklist (separate, unrelated track) lives in `final_day.md` — don't merge these two documents' concerns.

---

## 6. Feature #3 — Venue-fit / structure check (code-complete 2026-07-10)

A fourth tab ("Venue Fit") in the draft reviewer. Sibling of the reviewer audit, but a *presence/completeness* check rather than a quality judgement: does the draft CONTAIN the structural components a target venue expects (Limitations — mandatory at ACL; Broader Impacts — NeurIPS; Reproducibility — ICLR; Ethics; etc.)? **No external API — fully testable in a browser today** (this was the deliberate reason it was built before citation-gap, which needs the S2 key).

### Backend
- **`ingestion/structure_auditor.py`** (new) — `check_structure(paper_id, venue="generic", on_progress) -> dict` + `list_venues()`. Same coverage-first design as `reviewer_auditor` (section map + per-component `hybrid_retrieve` + deterministic **MISSING→THIN guard** when a matching section exists). Rubric = a component registry (`related_work`, `method`, `experiments`, `limitations`, `ethics`, `broader_impacts`, `reproducibility`, `conclusion`) composed per venue (`generic`/`neurips`/`icml`/`iclr`/`acl`/`emnlp`/`cvpr`) with a `required` flag. **Severity is deterministic** from `(required, verdict)` — a *required* component that's MISSING is high-severity (the headline `required_missing` count). Lazy retriever/torch imports mirror `reviewer_auditor` (import-order footgun).
- **`api/storage.py`** — `upload/get/delete_structure_report`, single key `{paper_id}.structure.json` with the venue stored *inside* the blob; delete wired into paper cleanup.
- **`api/main.py`** — `GET /venues` (static selector list) + `POST /papers/{id}/structure/stream?venue=&force=`, cloned from the audit stream. **Venue-aware cache:** a cached report is served only when `cached["venue"] == requested venue`, else recompute (keeps deletion to one key). `enforce_audit_quota`, `kind="audit"`.

### Frontend
- **`frontend/src/components/StructurePanel.jsx`** (new) — clone of the audit shell + a **venue `<select>`** (changing it re-runs the check; venue is in the effect deps). Renders the component checklist (PRESENT/THIN/MISSING with a `required` tag, a per-item `fix` line, evidence via `PDFPreviewPanel`), a completeness ring, and a red **required-missing banner**. Accent amber `#fbbf24`.
- **`frontend/src/components/MetricRing.jsx`** — added an `amber` accent case (shared component; cyan/violet were the only options).
- **`frontend/src/api.js`** — `structureCheckStream(paperId, venue, onEvent, force)` + `getVenues()`.
- **`frontend/src/pages/DraftReviewPage.jsx`** — fourth "Venue Fit" tab (the empty-state row now `flex-wrap`s for four tabs).

### Verified
- `import ingestion.structure_auditor` + `import api.main` boot clean (venues load, ACL correctly marks Limitations `required`).
- Deterministic layer unit-checked offline: MISSING→THIN guard, `(required,verdict)`→severity, evidence-ref resolution, fix-cleared-for-PRESENT all pass.
- `npm run build` clean.

### NOT done — pick up here
1. **Browser click-through** — open a draft → Venue Fit tab, switch venues (Generic → ACL → NeurIPS) and confirm each re-runs and the required-missing banner/severity reflect the venue. Fully unblocked (no key needed).
2. **Not committed.** New: `ingestion/structure_auditor.py`, `frontend/src/components/StructurePanel.jsx`. Modified: `api/main.py`, `api/storage.py`, `frontend/src/api.js`, `frontend/src/components/MetricRing.jsx`, `frontend/src/pages/DraftReviewPage.jsx`.

---

## 7. Feature #4 — Numbers Check (code-complete 2026-07-10)

A fifth tab ("Numbers Check"). Cousin of the Claim Audit, but reshaped to avoid overlap: where Claim Audit judges *qualitative* claims ("significantly outperforms"), this reconciles *exact figures* — the abstract says "92.4 F1 on SQuAD", does a results table actually show 92.4? Catches stale numbers, copy-paste/transcription slips, and abstract↔results drift. **No external API — fully browser-testable now.**

### Backend
- **`ingestion/numbers_auditor.py`** (new) — `audit_numbers(paper_id, on_progress) -> dict`, modelled directly on `claim_auditor` (extract from abstract/intro/conclusion → retrieve results/table evidence biased with `_EVIDENCE_BOOST` → batched verdicts). Verdicts **MATCH / MISMATCH / NOT_FOUND**. Inherits claim_auditor's conservative stance: a **MISMATCH must cite both an evidence chunk AND the conflicting `found_value`**, else it's downgraded to NOT_FOUND (never accuse without showing the two numbers). `consistency_score = matched / (matched + mismatched)` — NOT_FOUND excluded from the denominator (retrieval may have missed it). Table linearization is what makes the abstract-number → table-cell comparison work.
- **`api/storage.py`** — `upload/get/delete_numbers_report` (key `{paper_id}.numbers.json`); delete wired into paper cleanup.
- **`api/main.py`** — `POST /papers/{id}/numbers/stream`, clone of the audit stream, `enforce_audit_quota`, `kind="audit"`.

### Frontend
- **`frontend/src/components/NumbersPanel.jsx`** (new) — clone of the audit shell; mismatch cards show the discrepancy inline (`abstract: 92.4 → results: 91.2`), a red mismatch banner, NOT_FOUND surfaced as a manual-check nudge, MATCH behind a toggle, evidence via `PDFPreviewPanel`. Accent emerald `#34d399`.
- **`frontend/src/components/MetricRing.jsx`** — added an `emerald` accent case.
- **`frontend/src/api.js`** — `numbersCheckStream(paperId, onEvent, force)`.
- **`frontend/src/pages/DraftReviewPage.jsx`** — fifth "Numbers Check" tab.

### Verified
- `import ingestion.numbers_auditor` + `import api.main` boot clean.
- Downgrade guard unit-checked offline: MISMATCH without a cited chunk → NOT_FOUND; MISMATCH with cite+value → stays, keeps `found_value`; MISMATCH with cite but no value → NOT_FOUND. All pass.
- `npm run build` clean.

### NOT done — pick up here
1. **Browser click-through** — open a draft → Numbers Check tab, confirm it streams + renders; ideally test on a draft with a *known* abstract/results number mismatch to see a real MISMATCH card. Fully unblocked.
2. **Not committed.** New: `ingestion/numbers_auditor.py`, `frontend/src/components/NumbersPanel.jsx`. Modified: `api/main.py`, `api/storage.py`, `frontend/src/api.js`, `frontend/src/components/MetricRing.jsx`, `frontend/src/pages/DraftReviewPage.jsx`.

---

## Write-mode reviewer — current tab inventory (as of 2026-07-10)

Five tabs in `DraftReviewPage`, all code-complete, **none yet browser-tested**:
1. **Weakness Review** (`reviewer_auditor`) — methodological bar vs venue norms.
2. **Claim Audit** (`claim_auditor`) — qualitative claims vs own evidence.
3. **Novelty Scan** (`novelty_scout`) — closest prior work via Semantic Scholar. *Needs `SEMANTIC_SCHOLAR_API_KEY` for reliability (§5).* 
4. **Venue Fit** (`structure_auditor`) — expected-section presence per venue (§6). No external dep.
5. **Numbers Check** (`numbers_auditor`) — abstract figures vs results tables (§7). No external dep.

Remaining queued: **#2 citation-gap** (key-blocked, like novelty). The whole track still sits ahead of the deploy work in `final_day.md`.


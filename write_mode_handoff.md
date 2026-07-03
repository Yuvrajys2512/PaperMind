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

## 3. What's queued next (not started)

Ranked by leverage vs. effort, as discussed with the user:

1. **Novelty / related-work search** — embed the draft's abstract, search Semantic Scholar (free Graph API, has SPECTER2 embeddings built for exactly this) or arXiv/OpenAlex for close prior work, surface nearest neighbors. This is genuinely new work: a new external API integration, not a repurposing of existing code, because it searches *across* the literature rather than reasoning over one uploaded PDF like everything else in this app does.
2. **Citation gap check** — flag claims in the draft with no nearby citation. Smaller, but most useful paired with #1 (needs a corpus to cross-reference against).
3. **Venue-fit / structure check** — flags missing sections a target venue expects (limitations, reproducibility, ethics statement). Cheap: mostly reuses the `reviewer_auditor.py` scaffolding/pattern.
4. **Abstract/intro/results consistency check** — does the abstract's claims match what the results section shows. A variant of `claim_auditor.py`'s approach.

None of these have been scoped into a plan yet — treat this list as a starting menu for the next planning conversation, not a committed roadmap.

---

## 4. Quick orientation for whoever picks this up

- Full original plan + verification detail: `C:\Users\Yuvraj Srivastava\.claude\plans\proud-riding-wave.md`.
- This app has **no react-router** — all page navigation is hand-rolled `useState('page')` in `App.jsx`. Follow that pattern for anything new.
- Both audit engines (`ingestion/claim_auditor.py`, `ingestion/reviewer_auditor.py`) take only a `paper_id` and read from a per-paper Chroma collection — no Postgres/venue/publication-status coupling. Keep it that way; it's what made this feature cheap.
- No migration framework exists — schema changes are idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` statements inside each module's `_ensure_schema()`, run at import time. Follow that pattern, don't introduce Alembic or similar.
- Deploy checklist (separate, unrelated track) lives in `final_day.md` — don't merge these two documents' concerns.

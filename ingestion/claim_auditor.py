"""
ingestion/claim_auditor.py

Claim → Evidence grounding + overclaim detection.

This is the evidence_grader pointed at a *verification* task instead of a Q&A
task. Where grade_answer() classifies the sentences of a *generated answer*
against retrieved chunks, this module classifies the paper's *own headline
claims* (from the abstract / introduction / conclusion) against the paper's
*own evidence* (results prose + tables).

The three failure modes we surface:
  OVERCLAIM       — evidence exists but is weaker than the claim implies
                    (e.g. "significantly outperforms" over a 0.3% gap with no
                    error bars / significance test).
  SCOPE_MISMATCH  — the claim's scope exceeds what was actually tested
                    (e.g. "works on long documents" but only 512-token inputs).
  UNVERIFIABLE    — no in-paper evidence supports the claim at all.
Everything else is GROUNDED.

Why table retrieval matters: table_extractor linearizes every cell into
header-anchored text ("BERT — SQuAD F1: 0.79"), so a numeric claim retrieves
the actual cell — that's what lets the auditor judge effect size rather than
just topical overlap.

Design stance: false accusations are the only real failure mode (they damage
trust in *us*, not the paper). So the verdict prompt is deliberately
conservative — a flag requires the model to cite a specific evidence chunk,
and when the chunk reference is missing for a flag, we downgrade it. The user
always sees the cited evidence and is the final judge.

Public API
----------
audit_paper(paper_id, on_progress=None) -> dict
"""

from __future__ import annotations

import time

from ingestion.retriever import get_all_chunks
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# ── Tunables ───────────────────────────────────────────────────────────────
MAX_CLAIMS        = 12   # cap extracted claims so cost/latency stay bounded
EVIDENCE_PER_CLAIM = 6   # chunks retrieved per claim
VERDICT_BATCH      = 4   # claims graded per verdict LLM call (cost compromise)
_CLAIM_SECTION_CHARS = 6000  # cap on the claim-extraction context

# Section-name fragments we treat as claim-bearing. Section detection is
# imperfect, so we match case-insensitive substrings rather than exact names.
_CLAIM_SECTIONS = ("abstract", "introduction", "conclusion")

_VALID_VERDICTS = {"GROUNDED", "OVERCLAIM", "SCOPE_MISMATCH", "UNVERIFIABLE"}
_FLAG_VERDICTS  = {"OVERCLAIM", "SCOPE_MISMATCH"}  # require a cited evidence chunk


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — Claim extraction
# ───────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM_PROMPT = """You extract the falsifiable claims a research paper makes about itself.

You are given the Abstract, Introduction, and/or Conclusion of a paper. Pull out
the specific, checkable claims the authors make about THEIR OWN contribution —
the kind a skeptical reviewer would want to verify against the results.

INCLUDE claims of these types:
  performance  — "achieves X", "outperforms Y", "improves by Z%", "state-of-the-art"
  scope        — "works on long documents", "generalizes to unseen domains", "scales to N"
  capability   — "can do X", "enables Y", "is robust to Z"
  comparison   — "faster/cheaper/better than baseline B"

EXCLUDE: background, motivation, related-work statements, definitions, and
descriptions of what the paper *does* structurally ("we present", "in Section 3").
Only claims that could be checked against experimental results.

For each claim return:
  claim          — the claim in one sentence, faithful to the paper's wording
  type           — performance | scope | capability | comparison
  source_section — Abstract | Introduction | Conclusion
  hedge          — the strength word/phrase if any ("significantly", "substantially",
                   "consistently", "state-of-the-art"), else ""

Return ONLY a JSON array, at most __MAX_CLAIMS__ items, strongest claims first. No prose, no markdown.

[
  {"claim": "...", "type": "performance", "source_section": "Abstract", "hedge": "significantly"}
]
""".replace("__MAX_CLAIMS__", str(MAX_CLAIMS))


def _gather_claim_sections(chunks: list) -> str:
    """Join the abstract/intro/conclusion bodies for claim extraction.

    Falls back to the earliest chunks (by reading order) when section detection
    didn't label any claim-bearing section — the opening of a paper is almost
    always abstract+intro, so this keeps extraction working on messy PDFs.
    """
    picked = [
        c for c in chunks
        if any(s in str(c["metadata"].get("section", "")).lower() for s in _CLAIM_SECTIONS)
        and c["metadata"].get("section_type") != "table"
    ]
    if not picked:
        picked = [c for c in chunks if c["metadata"].get("section_type") != "table"][:6]

    out, total = [], 0
    for c in picked:
        text = c["text"].strip()
        section = c["metadata"].get("section", "?")
        block = f"[{section}]\n{text}"
        if total + len(block) > _CLAIM_SECTION_CHARS:
            break
        out.append(block)
        total += len(block)
    return "\n\n---\n\n".join(out)


def extract_claims(chunks: list) -> list[dict]:
    """Extract checkable claims from a paper's opening/closing sections."""
    context = _gather_claim_sections(chunks)
    if not context.strip():
        return []

    raw = chat_completion(
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Paper sections:\n\n{context}\n\nExtract the claims as JSON."},
        ],
        max_tokens=1400,
        temperature=0.1,
    )
    try:
        claims = parse_llm_json(raw, expect="array")
    except Exception as exc:
        print(f"[claim_auditor] Claim extraction parse failed: {exc}")
        return []

    if not isinstance(claims, list):
        return []

    cleaned = []
    for c in claims:
        if isinstance(c, dict) and isinstance(c.get("claim"), str) and c["claim"].strip():
            cleaned.append({
                "claim":          c["claim"].strip(),
                "type":           c.get("type", "performance"),
                "source_section": c.get("source_section", "?"),
                "hedge":          c.get("hedge", "") or "",
            })
    return cleaned[:MAX_CLAIMS]


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — Evidence retrieval
# ───────────────────────────────────────────────────────────────────────────

# Appended to the BM25 side of retrieval so result-bearing chunks (and the
# linearized table cells) surface over generic prose.
_EVIDENCE_BOOST = ["results", "table", "accuracy", "score", "outperforms", "compared", "evaluation"]


def gather_evidence(claim: str, paper_id: str) -> list[dict]:
    """Retrieve the chunks most likely to contain evidence for a claim,
    biased toward results/tables via boost terms."""
    try:
        return hybrid_retrieve(
            claim, paper_id, top_k=EVIDENCE_PER_CLAIM, boost_terms=_EVIDENCE_BOOST,
        )
    except Exception as exc:
        print(f"[claim_auditor] Evidence retrieval failed for a claim: {exc}")
        return []


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — Verdicts (batched, conservative)
# ───────────────────────────────────────────────────────────────────────────

_VERDICT_SYSTEM_PROMPT = """You are a meticulous, conservative peer reviewer auditing whether a paper's
claims are backed by its own evidence.

For each claim you are given numbered EVIDENCE chunks retrieved from the SAME
paper (results prose and linearized tables, where a table cell looks like
"BERT — SQuAD F1: 0.79"). Decide one verdict per claim:

  GROUNDED        — The evidence supports the claim AS STATED, including its
                    strength and scope. The numbers/results back it up.
  OVERCLAIM       — The evidence exists but is WEAKER than the claim's wording.
                    e.g. claim says "significantly outperforms" but the table
                    shows a tiny gap, or there are no error bars / no
                    significance test to justify a strength word.
  SCOPE_MISMATCH  — The claim's SCOPE exceeds what was tested. e.g. "works on
                    long documents" but evidence only covers short inputs; or a
                    dataset/setting the claim names is absent from the evidence.
  UNVERIFIABLE    — No evidence chunk supports the claim at all.

CRITICAL RULES — bias toward GROUNDED, flag only on concrete contradiction:
  - To return OVERCLAIM or SCOPE_MISMATCH you MUST cite the evidence chunk
    number that contradicts the claim's strength/scope in "evidence_ref".
    If you cannot point to a specific chunk, do NOT flag — choose GROUNDED
    (if anything supports it) or UNVERIFIABLE (if nothing does).
  - Do not flag OVERCLAIM merely because a strength word is present; flag only
    when the cited numbers are actually weak (small gap, no significance/error
    bars where the claim demands them).
  - When genuinely unsure between GROUNDED and a flag, choose GROUNDED.
  - "why" must be ONE sentence naming the concrete evidence (a number, a
    dataset, an absence) — not a generic statement.

Return ONLY a JSON array, one object per claim, in the SAME ORDER as the claims:

[
  {
    "index": 1,
    "verdict": "GROUNDED | OVERCLAIM | SCOPE_MISMATCH | UNVERIFIABLE",
    "severity": "low | med | high",
    "why": "one concrete sentence",
    "evidence_ref": <chunk number you relied on, or null>
  }
]
"""


def _format_evidence(chunks: list) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        section = meta.get("section", "?")
        page    = meta.get("page_num", "?")
        kind    = "TABLE" if meta.get("section_type") == "table" else "TEXT"
        text    = c.get("text", "")[:700]
        lines.append(f"[Chunk {i} | {kind} | {section} | p.{page}]\n{text}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"


def _verdict_batch(batch: list[dict]) -> list[dict]:
    """Grade a batch of {claim, evidence} items. Returns verdict dicts aligned
    to batch order. Each claim gets its own numbered evidence block so the
    model's evidence_ref maps back to a real chunk."""
    blocks = []
    for i, item in enumerate(batch, 1):
        blocks.append(
            f"CLAIM {i} ({item['claim_obj']['type']}, from {item['claim_obj']['source_section']}):\n"
            f"\"{item['claim_obj']['claim']}\"\n\n"
            f"EVIDENCE for claim {i}:\n{_format_evidence(item['evidence'])}"
        )
    user_prompt = (
        "\n\n========================================\n\n".join(blocks)
        + "\n\nAudit every claim above. Return the JSON array (one object per claim, in order)."
    )

    raw = chat_completion(
        messages=[
            {"role": "system", "content": _VERDICT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1200,
        temperature=0.0,
    )
    verdicts = parse_llm_json(raw, expect="array")
    return verdicts if isinstance(verdicts, list) else []


def _attach_verdict(claim_obj: dict, evidence: list[dict], verdict: dict | None) -> dict:
    """Merge a raw model verdict with the real cited chunk's metadata, applying
    the conservative downgrade rule (a flag without a valid cited chunk becomes
    GROUNDED — we never accuse without showable evidence)."""
    v = (verdict or {}).get("verdict", "UNVERIFIABLE")
    if v not in _VALID_VERDICTS:
        v = "UNVERIFIABLE"

    severity = (verdict or {}).get("severity", "med")
    if severity not in ("low", "med", "high"):
        severity = "med"

    why = str((verdict or {}).get("why", "")).strip()

    # Resolve the cited evidence chunk (1-based) → real metadata for the UI jump.
    ref = (verdict or {}).get("evidence_ref")
    cited = None
    if isinstance(ref, int) and 1 <= ref <= len(evidence):
        c = evidence[ref - 1]
        meta = c.get("metadata", {})
        cited = {
            "section":      meta.get("section", "?"),
            "page":         meta.get("page_num"),
            "section_type": meta.get("section_type", "text"),
            "quote":        c.get("text", "")[:600],
        }

    # Conservative downgrade: a flag we can't show evidence for is not a flag.
    if v in _FLAG_VERDICTS and cited is None:
        v = "GROUNDED"
        if not why:
            why = "Evidence supports the claim."

    return {
        "claim":          claim_obj["claim"],
        "type":           claim_obj["type"],
        "source_section": claim_obj["source_section"],
        "hedge":          claim_obj["hedge"],
        "verdict":        v,
        "severity":       severity,
        "why":            why,
        "evidence":       cited,
    }


# ───────────────────────────────────────────────────────────────────────────
# Orchestrator
# ───────────────────────────────────────────────────────────────────────────

def _emit(on_progress, stage: str, message: str, **kw):
    if on_progress:
        try:
            on_progress({"stage": stage, "message": message, **kw})
        except Exception:
            pass


def _empty_report(paper_id: str, *, failed: bool, reason: str = "") -> dict:
    return {
        "paper_id":       paper_id,
        "claims_checked": 0,
        "grounded":       0,
        "flagged":        0,
        "trust_score":    None,
        "claims":         [],
        "audit_failed":   failed,
        "reason":         reason,
        "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def audit_paper(paper_id: str, on_progress=None) -> dict:
    """Run the full claim-audit pipeline for one paper and return a report dict.

    Never raises — returns an `audit_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    try:
        _emit(on_progress, "reading", "Reading the paper…")
        all_chunks = get_all_chunks(paper_id)
        if not all_chunks:
            return _empty_report(paper_id, failed=True, reason="No content found for this paper.")

        _emit(on_progress, "extracting", "Extracting the paper's claims…")
        claims = extract_claims(all_chunks)
        if not claims:
            return _empty_report(paper_id, failed=False, reason="No checkable claims found.")

        _emit(on_progress, "gathering", f"Tracing evidence for {len(claims)} claims…")
        items = [{"claim_obj": c, "evidence": gather_evidence(c["claim"], paper_id)} for c in claims]

        _emit(on_progress, "auditing", "Auditing claims against the evidence…")
        results: list[dict] = []
        for start in range(0, len(items), VERDICT_BATCH):
            batch = items[start:start + VERDICT_BATCH]
            try:
                verdicts = _verdict_batch(batch)
            except Exception as exc:
                print(f"[claim_auditor] Verdict batch failed: {exc}")
                verdicts = []
            # Align by order; index field is advisory only.
            for i, item in enumerate(batch):
                results.append(_attach_verdict(item["claim_obj"], item["evidence"],
                                               verdicts[i] if i < len(verdicts) else None))

        flagged  = sum(1 for r in results if r["verdict"] in _FLAG_VERDICTS or r["verdict"] == "UNVERIFIABLE")
        grounded = len(results) - flagged
        trust    = round(grounded / len(results), 3) if results else None

        # Flagged first (most actionable), then by severity, so the UI can show
        # the problems without re-sorting.
        sev_rank = {"high": 0, "med": 1, "low": 2}
        results.sort(key=lambda r: (
            0 if r["verdict"] != "GROUNDED" else 1,
            sev_rank.get(r["severity"], 1),
        ))

        return {
            "paper_id":       paper_id,
            "claims_checked": len(results),
            "grounded":       grounded,
            "flagged":        flagged,
            "trust_score":    trust,
            "claims":         results,
            "audit_failed":   False,
            "reason":         "",
            "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[claim_auditor] Audit failed: {exc}")
        return _empty_report(paper_id, failed=True, reason=str(exc))

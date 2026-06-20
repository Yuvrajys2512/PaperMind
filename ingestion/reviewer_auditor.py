"""
ingestion/reviewer_auditor.py

Automated reviewer / weakness audit — methodological completeness against the
norms of a generic ML/NLP venue.

This is the COMPLEMENT of claim_auditor. Where claim_auditor asks "does the
paper back up its OWN claims with its OWN evidence?" (internal grounding /
overclaim), this module asks "does the paper meet the methodological BAR a
reviewer would hold it to?" (external norms / completeness). It produces the
kind of weaknesses section a referee writes: missing baselines, absent
ablations, no error bars / significance tests, small N, unaddressed threats to
validity, thin related work.

The hard part — absence vs. presence
------------------------------------
claim_auditor detects the PRESENCE of a contradiction. A reviewer audit mostly
detects the ABSENCE of something ("there are no error bars"). Absence is far
more prone to false accusation: if retrieval merely *failed to surface* the
results table, we'd confidently report "no significance testing" when it's
right there in Table 3 — and that damages trust in *us*, not the paper.

So the design is coverage-first, not retrieval-first:
  1. We hand the model the paper's full SECTION MAP (every section name + the
     table count) so it can see the paper's structure at a glance and can't
     claim "no Related Work" when a Related Work section plainly exists.
  2. Per dimension we retrieve targeted evidence (biased to the right terms)
     so a present-but-buried ablation/baseline still surfaces.
  3. A deterministic guard (`_SECTION_HINTS`) downgrades a MISSING verdict to
     WEAK whenever a section whose name matches the dimension exists — we saw a
     section on-topic, so the thing is at worst inadequate, never truly absent.
  4. The prompt biases toward OK/WEAK over MISSING when unsure, and the user
     always sees the cited evidence and is the final judge.

Public API
----------
review_paper(paper_id, on_progress=None) -> dict
"""

from __future__ import annotations

import time

from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# NOTE: get_all_chunks (retriever) and hybrid_retrieve (hybrid_retriever) are
# imported lazily inside the functions that use them, not at module top. They
# pull in chromadb *and* the torch-backed embedder/reranker; importing that pair
# in one process trips a native DLL access violation on Windows during pytest
# collection. Deferring them keeps the import light for the endpoint wiring and
# the pure-function tests, with no runtime behaviour change.

# ── Tunables ───────────────────────────────────────────────────────────────
EVIDENCE_PER_DIM = 5   # chunks retrieved per rubric dimension
VERDICT_BATCH    = 3   # dimensions graded per verdict LLM call (cost compromise)

_VALID_VERDICTS = {"OK", "WEAK", "MISSING"}
_FLAG_VERDICTS  = {"WEAK", "MISSING"}


# ───────────────────────────────────────────────────────────────────────────
# The rubric — a generic ML/NLP reviewer's methodological checklist
# ───────────────────────────────────────────────────────────────────────────
#
# Each dimension carries the retrieval query + lexical boost terms that surface
# the section it lives in, and a `section_hints` set of lowercase substrings: if
# the paper has a section whose name matches one of these, a MISSING verdict is
# downgraded to WEAK (the topic is present, just possibly inadequate).

DIMENSIONS: list[dict] = [
    {
        "key":   "baselines",
        "label": "Baselines & comparisons",
        "query": "baselines compared against prior methods comparison table results",
        "boost": ["baseline", "compared", "prior", "outperforms", "table", "results", "vs"],
        "section_hints": ("experiment", "result", "evaluation", "comparison"),
        "ask":   "Does the paper compare against appropriate, recent baselines — "
                 "or only weak / dated / self-built ones?",
    },
    {
        "key":   "ablations",
        "label": "Ablation studies",
        "query": "ablation study removing components contribution of each part",
        "boost": ["ablation", "ablate", "without", "component", "remove", "variant", "w/o"],
        "section_hints": ("ablation",),
        "ask":   "Is there an ablation isolating the contribution of the paper's "
                 "individual components?",
    },
    {
        "key":   "statistical_rigor",
        "label": "Statistical rigor",
        "query": "error bars standard deviation significance test multiple seeds runs variance",
        "boost": ["error", "deviation", "significance", "p-value", "seed", "runs",
                  "confidence", "variance", "stddev", "std"],
        "section_hints": (),  # rigor lives inside results, not its own section
        "ask":   "Are results reported with error bars / variance over multiple "
                 "runs and significance tests — or single-point numbers?",
    },
    {
        "key":   "sample_size",
        "label": "Sample size & data scale",
        "query": "dataset size number of examples participants test set scale",
        "boost": ["dataset", "examples", "samples", "participants", "size", "N=",
                  "instances", "test set", "train"],
        "section_hints": ("data", "dataset", "setup", "experiment"),
        "ask":   "Is the evaluation data large/diverse enough to support the "
                 "claims, or is N small / single-domain?",
    },
    {
        "key":   "threats_to_validity",
        "label": "Threats to validity & limitations",
        "query": "limitations threats to validity assumptions failure cases generalization",
        "boost": ["limitation", "threat", "validity", "assume", "failure",
                  "caveat", "generaliz", "weakness"],
        "section_hints": ("limitation", "threat", "discussion", "conclusion"),
        "ask":   "Does the paper acknowledge its limitations / threats to "
                 "validity and where the method would fail?",
    },
    {
        "key":   "related_work",
        "label": "Related work coverage",
        "query": "related work prior literature positioning compared to existing",
        "boost": ["related", "prior", "literature", "previous", "existing", "et al"],
        "section_hints": ("related", "background", "literature"),
        "ask":   "Is prior work adequately covered and is the contribution "
                 "clearly positioned against it?",
    },
]


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — Paper structure + per-dimension evidence
# ───────────────────────────────────────────────────────────────────────────

def build_section_map(chunks: list) -> str:
    """A one-line inventory of the paper's structure: every distinct section
    name in reading order, plus the table count.

    This is the antidote to false "missing" accusations — the model sees up
    front that, say, a "Related Work" or "Ablation" section exists before it
    decides anything is absent.
    """
    seen: list[str] = []
    tables = 0
    for c in chunks:
        meta = c.get("metadata", {})
        if meta.get("section_type") == "table":
            tables += 1
        sec = str(meta.get("section", "") or "").strip()
        if sec and sec not in seen:
            seen.append(sec)
    sections = ", ".join(seen) if seen else "(no section labels detected)"
    return f"Sections in reading order: {sections}\nTables detected: {tables}"


def _present_section_hints(chunks: list) -> set[str]:
    """The lowercase section names actually present, for the MISSING→WEAK guard."""
    names = set()
    for c in chunks:
        sec = str(c.get("metadata", {}).get("section", "") or "").lower()
        if sec:
            names.add(sec)
    return names


def gather_evidence(dim: dict, paper_id: str) -> list[dict]:
    """Retrieve the chunks most likely to address a rubric dimension, biased
    toward the dimension's vocabulary so a present-but-buried section surfaces."""
    from ingestion.hybrid_retriever import hybrid_retrieve  # lazy: see module note
    try:
        return hybrid_retrieve(
            dim["query"], paper_id, top_k=EVIDENCE_PER_DIM, boost_terms=dim["boost"],
        )
    except Exception as exc:
        print(f"[reviewer_auditor] Evidence retrieval failed for {dim['key']}: {exc}")
        return []


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — Verdicts (batched, conservative)
# ───────────────────────────────────────────────────────────────────────────

_VERDICT_SYSTEM_PROMPT = """You are an experienced, fair peer reviewer for a top ML/NLP venue, writing the
"weaknesses" part of a review. You assess whether a paper meets the field's
methodological norms on specific dimensions — NOT whether its claims match its
own numbers (that is a different audit).

You are given the paper's SECTION MAP (its overall structure) and, for each
rubric dimension, numbered EVIDENCE chunks retrieved from the paper (prose and
linearized tables, where a table cell looks like "BERT — SQuAD F1: 0.79").

For each dimension return one verdict:
  OK       — The paper adequately satisfies this dimension. Cite the chunk that
             shows it.
  WEAK     — The dimension is addressed but inadequately (e.g. only weak/dated
             baselines, an ablation that omits the key component, single-run
             numbers with no variance, limitations mentioned only in passing).
             Cite the chunk that is inadequate.
  MISSING  — Nothing in the evidence OR the section map addresses this dimension
             at all.

CRITICAL RULES — be fair, never invent an absence:
  - Before returning MISSING, check the SECTION MAP. If a section plainly covers
    this dimension (e.g. a "Related Work" section for related-work coverage, an
    "Ablation" section for ablations), it is NOT missing — return OK or WEAK.
  - For OK or WEAK you MUST cite the evidence chunk number you relied on in
    "evidence_ref". For MISSING, "evidence_ref" is null.
  - Retrieval is imperfect — absence of evidence here is not proof of absence in
    the paper. When genuinely unsure between MISSING and WEAK, choose WEAK; when
    unsure between WEAK and OK, choose OK. Bias away from harsh verdicts.
  - "why" is ONE sentence naming the concrete thing (a number, a named baseline,
    an absent section) — not a generic statement.
  - "reviewer_question" is the single sharpest question a reviewer would ask the
    authors about this dimension (one sentence). Always provide it.

Return ONLY a JSON array, one object per dimension, in the SAME ORDER given:

[
  {
    "index": 1,
    "verdict": "OK | WEAK | MISSING",
    "severity": "low | med | high",
    "why": "one concrete sentence",
    "reviewer_question": "the sharpest question a reviewer would ask",
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
    return "\n\n".join(lines) if lines else "(no evidence retrieved for this dimension)"


def _verdict_batch(section_map: str, batch: list[dict]) -> list[dict]:
    """Grade a batch of {dim, evidence} items. Each dimension gets its own
    numbered evidence block so the model's evidence_ref maps back to a real
    chunk. The shared section map is prepended once."""
    blocks = []
    for i, item in enumerate(batch, 1):
        dim = item["dim"]
        blocks.append(
            f"DIMENSION {i} — {dim['label']}\n"
            f"What a reviewer checks: {dim['ask']}\n\n"
            f"EVIDENCE for dimension {i}:\n{_format_evidence(item['evidence'])}"
        )
    user_prompt = (
        f"PAPER SECTION MAP\n{section_map}\n\n"
        "========================================\n\n"
        + "\n\n========================================\n\n".join(blocks)
        + "\n\nReview every dimension above. Return the JSON array (one object per "
          "dimension, in order)."
    )

    raw = chat_completion(
        messages=[
            {"role": "system", "content": _VERDICT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1300,
        temperature=0.0,
    )
    verdicts = parse_llm_json(raw, expect="array")
    return verdicts if isinstance(verdicts, list) else []


def _attach_verdict(
    dim: dict, evidence: list[dict], verdict: dict | None, present_sections: set[str],
) -> dict:
    """Merge a raw model verdict with the real cited chunk's metadata, applying
    the conservative MISSING→WEAK guard (if a section on-topic exists, the
    dimension is at worst inadequate, never truly absent)."""
    v = (verdict or {}).get("verdict", "MISSING")
    if v not in _VALID_VERDICTS:
        v = "MISSING"

    severity = (verdict or {}).get("severity", "med")
    if severity not in ("low", "med", "high"):
        severity = "med"

    why = str((verdict or {}).get("why", "")).strip()
    question = str((verdict or {}).get("reviewer_question", "")).strip() or dim["ask"]

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

    # Conservative guard: never call a dimension MISSING when a section whose
    # name matches it actually exists in the paper — the topic is present, so at
    # worst it's inadequate (WEAK).
    if v == "MISSING" and dim["section_hints"]:
        if any(hint in name for name in present_sections for hint in dim["section_hints"]):
            v = "WEAK"
            if not why:
                why = "A section on this topic exists but may not fully meet the bar."

    return {
        "key":               dim["key"],
        "label":             dim["label"],
        "verdict":           v,
        "severity":          severity,
        "why":               why,
        "reviewer_question": question,
        "evidence":          cited,
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
        "paper_id":           paper_id,
        "dimensions_checked": 0,
        "ok":                 0,
        "weak":               0,
        "missing":            0,
        "review_score":       None,
        "dimensions":         [],
        "review_failed":      failed,
        "reason":             reason,
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def review_paper(paper_id: str, on_progress=None) -> dict:
    """Run the full reviewer/weakness audit for one paper and return a report.

    Never raises — returns a `review_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    try:
        from ingestion.retriever import get_all_chunks  # lazy: see module note

        _emit(on_progress, "reading", "Reading the paper…")
        all_chunks = get_all_chunks(paper_id)
        if not all_chunks:
            return _empty_report(paper_id, failed=True, reason="No content found for this paper.")

        section_map      = build_section_map(all_chunks)
        present_sections = _present_section_hints(all_chunks)

        _emit(on_progress, "gathering", "Tracing evidence for each review dimension…")
        items = [{"dim": d, "evidence": gather_evidence(d, paper_id)} for d in DIMENSIONS]

        _emit(on_progress, "reviewing", "Reviewing the paper against venue norms…")
        results: list[dict] = []
        for start in range(0, len(items), VERDICT_BATCH):
            batch = items[start:start + VERDICT_BATCH]
            try:
                verdicts = _verdict_batch(section_map, batch)
            except Exception as exc:
                print(f"[reviewer_auditor] Verdict batch failed: {exc}")
                verdicts = []
            # Align by order; index field is advisory only.
            for i, item in enumerate(batch):
                results.append(_attach_verdict(
                    item["dim"], item["evidence"],
                    verdicts[i] if i < len(verdicts) else None,
                    present_sections,
                ))

        missing = sum(1 for r in results if r["verdict"] == "MISSING")
        weak    = sum(1 for r in results if r["verdict"] == "WEAK")
        ok      = len(results) - missing - weak
        score   = round(ok / len(results), 3) if results else None

        # Worst verdicts first (MISSING > WEAK > OK), then by severity, so the
        # UI shows the sharpest weaknesses without re-sorting.
        verdict_rank = {"MISSING": 0, "WEAK": 1, "OK": 2}
        sev_rank     = {"high": 0, "med": 1, "low": 2}
        results.sort(key=lambda r: (
            verdict_rank.get(r["verdict"], 1),
            sev_rank.get(r["severity"], 1),
        ))

        return {
            "paper_id":           paper_id,
            "dimensions_checked": len(results),
            "ok":                 ok,
            "weak":               weak,
            "missing":            missing,
            "review_score":       score,
            "dimensions":         results,
            "review_failed":      False,
            "reason":             "",
            "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[reviewer_auditor] Review failed: {exc}")
        return _empty_report(paper_id, failed=True, reason=str(exc))

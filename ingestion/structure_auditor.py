"""
ingestion/structure_auditor.py

Venue-fit / structure check for a draft in "write mode".

Sibling of reviewer_auditor, but a different question. reviewer_auditor asks
"does the paper meet the methodological BAR?" (quality of baselines, ablations,
rigor). This asks the cheaper, more mechanical question authors actually forget:
"does the draft CONTAIN the structural components a target venue expects?" —
a Limitations section (mandatory at ACL since 2023), a Broader Impacts / ethics
statement (required at NeurIPS), a reproducibility statement (ICLR), etc.

Same coverage-first design as reviewer_auditor, for the same reason (absence is
easy to get wrong):
  1. Hand the model the paper's full SECTION MAP so it can't claim "no Related
     Work" when the section plainly exists.
  2. Per component retrieve targeted evidence, so a statement buried in another
     section (e.g. a code-availability line inside the intro, limitations folded
     into the conclusion) still surfaces.
  3. A deterministic guard downgrades a MISSING verdict to THIN whenever a
     section whose name matches the component exists.
  4. Severity is computed deterministically from whether the venue REQUIRES the
     component — a required component that's missing is the headline risk.

This is a checklist, not a judgement: the author sees exactly what's absent and
a one-line fix, and decides. We never fail a draft.

Public API
----------
check_structure(paper_id, venue="generic", on_progress=None) -> dict
list_venues() -> list[dict]   # {slug, label} for the UI selector
"""

from __future__ import annotations

import time

from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# NOTE: get_all_chunks (retriever) and hybrid_retrieve (hybrid_retriever) are
# imported lazily inside the functions that use them — they pull chromadb + the
# torch-backed embedder/reranker, whose paired import trips a native DLL access
# violation on Windows during test collection. Same rationale as reviewer_auditor.

# ── Tunables ───────────────────────────────────────────────────────────────
EVIDENCE_PER_COMPONENT = 4   # chunks retrieved per structural component
VERDICT_BATCH          = 4   # components graded per verdict LLM call

_VALID_VERDICTS = {"PRESENT", "THIN", "MISSING"}


# ───────────────────────────────────────────────────────────────────────────
# Component registry — the structural elements we can check for
# ───────────────────────────────────────────────────────────────────────────
#
# Each component carries the retrieval query + lexical boost terms that surface
# it, a `section_hints` set (lowercase substrings; a matching section name
# downgrades MISSING → THIN), the reviewer-facing `ask`, and a default `fix`.

_COMPONENTS: dict[str, dict] = {
    "related_work": {
        "label": "Related work",
        "query": "related work prior literature previous approaches positioning",
        "boost": ["related", "prior", "literature", "previous", "existing", "et al"],
        "section_hints": ("related", "background", "literature"),
        "ask":   "Is there a related-work / background section positioning the paper against prior literature?",
        "fix":   "Add a Related Work section situating the contribution against recent prior methods.",
    },
    "method": {
        "label": "Method / approach",
        "query": "method approach model architecture proposed algorithm we propose",
        "boost": ["method", "approach", "model", "architecture", "algorithm", "propose", "framework"],
        "section_hints": ("method", "approach", "model", "architecture", "framework", "proposed"),
        "ask":   "Is the proposed method / approach described in its own section?",
        "fix":   "Add a Method section describing the proposed approach in enough detail to reproduce.",
    },
    "experiments": {
        "label": "Experiments & results",
        "query": "experiments results evaluation benchmark datasets performance table",
        "boost": ["experiment", "result", "evaluation", "benchmark", "dataset", "table", "accuracy"],
        "section_hints": ("experiment", "result", "evaluation", "empirical"),
        "ask":   "Is there an experiments / results section with empirical evaluation?",
        "fix":   "Add an Experiments section evaluating the method on relevant benchmarks.",
    },
    "limitations": {
        "label": "Limitations",
        "query": "limitations weaknesses failure cases assumptions what does not work threats",
        "boost": ["limitation", "weakness", "failure", "assume", "caveat", "threat", "does not"],
        "section_hints": ("limitation", "threat", "weakness"),
        "ask":   "Is there an explicit Limitations section discussing where the method falls short?",
        "fix":   "Add a Limitations section naming concrete failure modes, assumptions, and scope boundaries.",
    },
    "ethics": {
        "label": "Ethics statement",
        "query": "ethics statement ethical considerations consent data privacy risks",
        "boost": ["ethic", "ethical", "consent", "privacy", "harm", "misuse", "risk"],
        "section_hints": ("ethic",),
        "ask":   "Is there an ethics statement / ethical-considerations discussion?",
        "fix":   "Add an Ethics Statement covering data provenance, consent, and potential for misuse.",
    },
    "broader_impacts": {
        "label": "Broader / societal impact",
        "query": "broader impact societal impact positive negative consequences deployment",
        "boost": ["broader", "impact", "societal", "society", "consequence", "deploy", "misuse"],
        "section_hints": ("impact", "societal", "broader"),
        "ask":   "Is there a broader-impacts / societal-impact discussion?",
        "fix":   "Add a Broader Impacts paragraph discussing societal benefits and risks of the work.",
    },
    "reproducibility": {
        "label": "Reproducibility (code & data)",
        "query": "reproducibility code available github data release implementation details hyperparameters",
        "boost": ["reproduc", "code", "github", "available", "release", "hyperparameter", "implementation", "appendix"],
        "section_hints": ("reproduc", "availability", "implementation detail"),
        "ask":   "Is there a reproducibility statement or code/data availability (link, hyperparameters, setup)?",
        "fix":   "Add a Reproducibility statement: code/data links, hyperparameters, and compute/setup details.",
    },
    "conclusion": {
        "label": "Conclusion",
        "query": "conclusion concluding remarks summary future work",
        "boost": ["conclusion", "conclude", "summary", "future work", "in summary"],
        "section_hints": ("conclusion", "concluding", "summary"),
        "ask":   "Is there a conclusion summarizing the contribution and future work?",
        "fix":   "Add a Conclusion summarizing findings and outlining future work.",
    },
}

# Canonical display order for whatever subset a venue selects.
_ORDER = ["related_work", "method", "experiments", "limitations",
          "ethics", "broader_impacts", "reproducibility", "conclusion"]


def _rubric(label: str, **required_by_key: bool) -> dict:
    """Build a venue rubric from {component_key: is_required}. Unknown keys are
    ignored; components render in canonical order."""
    comps = []
    for key in _ORDER:
        if key in required_by_key:
            comps.append({**_COMPONENTS[key], "key": key, "required": required_by_key[key]})
    return {"label": label, "components": comps}


# ───────────────────────────────────────────────────────────────────────────
# Venue rubrics — which components a venue expects, and which it REQUIRES.
# required=True  → its absence is a high-severity, headline gap.
# required=False → expected/recommended; absence is a nudge, not a blocker.
# ───────────────────────────────────────────────────────────────────────────

VENUES: dict[str, dict] = {
    "generic": _rubric(
        "Generic ML/CS",
        related_work=True, method=True, experiments=True,
        limitations=False, reproducibility=False, conclusion=False,
    ),
    "neurips": _rubric(
        "NeurIPS",
        related_work=True, method=True, experiments=True,
        limitations=True, broader_impacts=True, reproducibility=False, conclusion=False,
    ),
    "icml": _rubric(
        "ICML",
        related_work=True, method=True, experiments=True,
        limitations=True, reproducibility=False, conclusion=False,
    ),
    "iclr": _rubric(
        "ICLR",
        related_work=True, method=True, experiments=True,
        limitations=False, ethics=False, reproducibility=True, conclusion=False,
    ),
    "acl": _rubric(
        "ACL / ARR",
        related_work=True, method=True, experiments=True,
        limitations=True, ethics=False, conclusion=False,
    ),
    "emnlp": _rubric(
        "EMNLP",
        related_work=True, method=True, experiments=True,
        limitations=True, ethics=False, conclusion=False,
    ),
    "cvpr": _rubric(
        "CVPR / ICCV",
        related_work=True, method=True, experiments=True,
        limitations=False, conclusion=False,
    ),
}

_DEFAULT_VENUE = "generic"


def list_venues() -> list[dict]:
    """{slug, label} pairs for the UI selector, in a sensible display order."""
    return [{"slug": slug, "label": VENUES[slug]["label"]} for slug in VENUES]


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — section map + per-component evidence  (mirrors reviewer_auditor)
# ───────────────────────────────────────────────────────────────────────────

def build_section_map(chunks: list) -> str:
    """One-line inventory of the draft's structure — the antidote to false
    'missing' accusations (the model sees which sections exist up front)."""
    seen: list[str] = []
    for c in chunks:
        sec = str(c.get("metadata", {}).get("section", "") or "").strip()
        if sec and sec not in seen:
            seen.append(sec)
    sections = ", ".join(seen) if seen else "(no section labels detected)"
    return f"Sections in reading order: {sections}"


def _present_section_hints(chunks: list) -> set[str]:
    """Lowercase section names actually present, for the MISSING→THIN guard."""
    names = set()
    for c in chunks:
        sec = str(c.get("metadata", {}).get("section", "") or "").lower()
        if sec:
            names.add(sec)
    return names


def gather_evidence(comp: dict, paper_id: str) -> list[dict]:
    """Retrieve chunks most likely to contain a component, biased to its
    vocabulary so a buried statement (e.g. a code link in the intro) surfaces."""
    from ingestion.hybrid_retriever import hybrid_retrieve  # lazy: see module note
    try:
        return hybrid_retrieve(
            comp["query"], paper_id, top_k=EVIDENCE_PER_COMPONENT, boost_terms=comp["boost"],
        )
    except Exception as exc:
        print(f"[structure_auditor] Evidence retrieval failed for {comp['key']}: {exc}")
        return []


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — verdicts (batched, conservative)
# ───────────────────────────────────────────────────────────────────────────

_VERDICT_SYSTEM_PROMPT = """You check whether a paper draft CONTAINS the structural components expected in
a strong submission. This is a presence/completeness check — NOT a quality
judgement of the results.

You are given the draft's SECTION MAP (its overall structure) and, per component,
numbered EVIDENCE chunks retrieved from the draft. For each component decide:

  PRESENT — The component clearly exists and is developed (its own section, or a
            substantive dedicated passage). Cite the chunk that shows it.
  THIN    — The component is touched on but underdeveloped (e.g. limitations
            mentioned in a single passing sentence, a code link with no other
            reproducibility detail). Cite the chunk.
  MISSING — Nothing in the evidence OR the section map addresses this component.

CRITICAL RULES — never invent an absence:
  - Before MISSING, check the SECTION MAP and evidence. A component can live under
    a different name or inside another section (limitations folded into a
    Discussion, a code link in the intro). If it's there at all, it is PRESENT or
    THIN, never MISSING.
  - For PRESENT or THIN you MUST cite the evidence chunk number in "evidence_ref".
    For MISSING, "evidence_ref" is null.
  - Retrieval is imperfect — when unsure between MISSING and THIN, choose THIN.
  - "why" is ONE concrete sentence (name the section, or the absence).
  - "fix" is ONE actionable sentence telling the author what to add or expand.
    For PRESENT components, "fix" may be an empty string.

Return ONLY a JSON array, one object per component, in the SAME ORDER given:

[
  {
    "index": 1,
    "verdict": "PRESENT | THIN | MISSING",
    "why": "one concrete sentence",
    "fix": "one actionable sentence, or empty if PRESENT",
    "evidence_ref": <chunk number, or null>
  }
]
"""


def _format_evidence(chunks: list) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        section = meta.get("section", "?")
        page    = meta.get("page_num", "?")
        text    = c.get("text", "")[:600]
        lines.append(f"[Chunk {i} | {section} | p.{page}]\n{text}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved for this component)"


def _verdict_batch(section_map: str, batch: list[dict]) -> list[dict]:
    """Grade a batch of {comp, evidence} items. Each component gets its own
    numbered evidence block; the shared section map is prepended once."""
    blocks = []
    for i, item in enumerate(batch, 1):
        comp = item["comp"]
        blocks.append(
            f"COMPONENT {i} — {comp['label']}\n"
            f"What to look for: {comp['ask']}\n\n"
            f"EVIDENCE for component {i}:\n{_format_evidence(item['evidence'])}"
        )
    user_prompt = (
        f"PAPER SECTION MAP\n{section_map}\n\n"
        "========================================\n\n"
        + "\n\n========================================\n\n".join(blocks)
        + "\n\nCheck every component above. Return the JSON array (one object per "
          "component, in order)."
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


# Severity is a deterministic function of (required, verdict) — a required
# component that's absent is the sharpest gap; an optional nicety is a nudge.
def _severity(required: bool, verdict: str) -> str:
    if verdict == "MISSING":
        return "high" if required else "low"
    if verdict == "THIN":
        return "med" if required else "low"
    return "low"  # PRESENT


def _attach_verdict(
    comp: dict, evidence: list[dict], verdict: dict | None, present_sections: set[str],
) -> dict:
    """Merge a raw model verdict with the cited chunk's metadata, applying the
    conservative MISSING→THIN guard when a section on-topic exists."""
    v = (verdict or {}).get("verdict", "MISSING")
    if v not in _VALID_VERDICTS:
        v = "MISSING"

    why = str((verdict or {}).get("why", "")).strip()
    fix = str((verdict or {}).get("fix", "")).strip()

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

    # Guard: never call a component MISSING when a section whose name matches it
    # exists — the topic is present, so at worst it's underdeveloped (THIN).
    if v == "MISSING" and comp["section_hints"]:
        if any(hint in name for name in present_sections for hint in comp["section_hints"]):
            v = "THIN"
            if not why:
                why = "A section on this topic exists but may be underdeveloped."

    # A missing/thin component always needs an actionable fix; fall back to the
    # component's default when the model didn't supply one.
    if v != "PRESENT" and not fix:
        fix = comp["fix"]

    return {
        "key":      comp["key"],
        "label":    comp["label"],
        "required": comp["required"],
        "verdict":  v,
        "severity": _severity(comp["required"], v),
        "why":      why,
        "fix":      fix if v != "PRESENT" else "",
        "evidence": cited,
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


def _empty_report(paper_id: str, venue: str, *, failed: bool, reason: str = "") -> dict:
    return {
        "paper_id":           paper_id,
        "venue":              venue,
        "venue_label":        VENUES.get(venue, VENUES[_DEFAULT_VENUE])["label"],
        "components_checked": 0,
        "present":            0,
        "thin":               0,
        "missing":            0,
        "required_missing":   0,
        "completeness_score": None,
        "components":         [],
        "structure_failed":   failed,
        "reason":             reason,
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Worst-first ordering for the UI: required-missing, then by verdict, then severity.
_VERDICT_RANK = {"MISSING": 0, "THIN": 1, "PRESENT": 2}
_SEV_RANK     = {"high": 0, "med": 1, "low": 2}


def check_structure(paper_id: str, venue: str = _DEFAULT_VENUE, on_progress=None) -> dict:
    """Run the venue-fit / structure check for one draft and return a report.

    Never raises — returns a `structure_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    if venue not in VENUES:
        venue = _DEFAULT_VENUE
    rubric = VENUES[venue]

    try:
        from ingestion.retriever import get_all_chunks  # lazy: see module note

        _emit(on_progress, "reading", "Reading the draft…")
        all_chunks = get_all_chunks(paper_id)
        if not all_chunks:
            return _empty_report(paper_id, venue, failed=True, reason="No content found for this draft.")

        section_map      = build_section_map(all_chunks)
        present_sections = _present_section_hints(all_chunks)
        components       = rubric["components"]

        _emit(on_progress, "gathering", "Locating each expected component…")
        items = [{"comp": c, "evidence": gather_evidence(c, paper_id)} for c in components]

        _emit(on_progress, "checking", f"Checking structure against {rubric['label']} norms…")
        results: list[dict] = []
        for start in range(0, len(items), VERDICT_BATCH):
            batch = items[start:start + VERDICT_BATCH]
            try:
                verdicts = _verdict_batch(section_map, batch)
            except Exception as exc:
                print(f"[structure_auditor] Verdict batch failed: {exc}")
                verdicts = []
            for i, item in enumerate(batch):
                results.append(_attach_verdict(
                    item["comp"], item["evidence"],
                    verdicts[i] if i < len(verdicts) else None,
                    present_sections,
                ))

        missing = sum(1 for r in results if r["verdict"] == "MISSING")
        thin    = sum(1 for r in results if r["verdict"] == "THIN")
        present = len(results) - missing - thin
        required_missing = sum(1 for r in results if r["verdict"] == "MISSING" and r["required"])
        score   = round(present / len(results), 3) if results else None

        results.sort(key=lambda r: (
            0 if (r["verdict"] == "MISSING" and r["required"]) else 1,
            _VERDICT_RANK.get(r["verdict"], 1),
            _SEV_RANK.get(r["severity"], 1),
        ))

        return {
            "paper_id":           paper_id,
            "venue":              venue,
            "venue_label":        rubric["label"],
            "components_checked": len(results),
            "present":            present,
            "thin":               thin,
            "missing":            missing,
            "required_missing":   required_missing,
            "completeness_score": score,
            "components":         results,
            "structure_failed":   False,
            "reason":             "",
            "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[structure_auditor] Structure check failed: {exc}")
        return _empty_report(paper_id, venue, failed=True, reason=str(exc))

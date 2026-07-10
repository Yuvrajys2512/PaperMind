"""
ingestion/numbers_auditor.py

Numbers consistency check — do the figures in the abstract/intro match the
results?

A close cousin of claim_auditor, but a sharper, narrower question. claim_auditor
judges *qualitative* claims ("significantly outperforms") against the evidence.
This checks *exact figures*: the abstract says "92.4 F1 on SQuAD" — does the
results section / a table actually show 92.4? It catches a different, very common
error class that claim_auditor doesn't target:
  - a stale number left over from an earlier draft,
  - a copy-paste/transcription slip (28.4 → 82.4),
  - an abstract headline that quietly disagrees with Table 3.

Why table retrieval matters (same as claim_auditor): table_extractor linearizes
every cell into header-anchored text ("BERT — SQuAD F1: 0.79"), so a numeric
claim retrieves the actual cell — that's what lets us compare the abstract's
number to the results number rather than just topical prose.

Design stance (inherited from claim_auditor): a false accusation is the only
real failure — it damages trust in *us*, not the paper. So a MISMATCH requires
the model to cite the specific evidence chunk AND report the conflicting value it
found; if it can't, we downgrade to NOT_FOUND. When a number simply isn't in the
retrieved evidence we say NOT_FOUND (retrieval may have missed it), never
MISMATCH. The user always sees the cited evidence and is the final judge.

Public API
----------
audit_numbers(paper_id, on_progress=None) -> dict
"""

from __future__ import annotations

import time

from ingestion.retriever import get_all_chunks
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# ── Tunables ───────────────────────────────────────────────────────────────
MAX_NUMBERS         = 14   # cap extracted numeric claims (cost/latency bound)
EVIDENCE_PER_NUMBER = 6    # chunks retrieved per number
VERDICT_BATCH       = 4    # numbers graded per verdict LLM call
_EXTRACT_SECTION_CHARS = 6000

# Same claim-bearing sections claim_auditor keys on — headline numbers live in
# the abstract/intro (and are sometimes restated in the conclusion).
_NUMBER_SECTIONS = ("abstract", "introduction", "conclusion")

# Bias retrieval toward result-bearing chunks and linearized table cells.
_EVIDENCE_BOOST = ["results", "table", "accuracy", "score", "f1", "bleu",
                   "outperforms", "compared", "evaluation", "%"]

_VALID_VERDICTS = {"MATCH", "MISMATCH", "NOT_FOUND"}
_FLAG_VERDICTS  = {"MISMATCH"}  # require a cited chunk AND a found value


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — extract numeric claims
# ───────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM_PROMPT = """You extract the checkable NUMERIC claims a paper makes about its own results.

You are given the Abstract, Introduction, and/or Conclusion of a paper. Pull out
the specific quantitative results the authors state — the numbers a skeptical
reader would want to confirm against the results tables.

INCLUDE numbers of these kinds:
  metric      — "92.4 F1", "78.3% accuracy", "BLEU of 34.1", "0.89 AUC"
  improvement — "+3.2 points", "5% relative gain", "reduces error by 12%"
  speed/scale — "3.2x faster", "trained on 1.2M examples", "2.5B parameters"

EXCLUDE (these are NOT results to verify against tables):
  - years, dates, citation/reference numbers, section/figure/equation numbers,
  - page counts, hyperparameters given only as settings (learning rate, batch
    size, epochs) unless stated as a headline result.

For each numeric claim return:
  value          — the number exactly as written ("92.4", "3.2", "1.2M", "+5%")
  metric         — what the number measures ("F1", "accuracy", "speedup",
                   "training examples"), or "" if unclear
  dataset        — the dataset/benchmark/setting it applies to, or "" if unstated
  claim          — the full phrase as written ("achieves 92.4 F1 on SQuAD")
  source_section — Abstract | Introduction | Conclusion

Return ONLY a JSON array, at most __MAX_NUMBERS__ items, most important first.
No prose, no markdown.

[
  {"value": "92.4", "metric": "F1", "dataset": "SQuAD", "claim": "achieves 92.4 F1 on SQuAD", "source_section": "Abstract"}
]
""".replace("__MAX_NUMBERS__", str(MAX_NUMBERS))


def _gather_number_sections(chunks: list) -> str:
    """Join the abstract/intro/conclusion bodies for numeric-claim extraction.
    Falls back to the earliest non-table chunks when section detection labelled
    nothing (mirrors claim_auditor)."""
    picked = [
        c for c in chunks
        if any(s in str(c["metadata"].get("section", "")).lower() for s in _NUMBER_SECTIONS)
        and c["metadata"].get("section_type") != "table"
    ]
    if not picked:
        picked = [c for c in chunks if c["metadata"].get("section_type") != "table"][:6]

    out, total = [], 0
    for c in picked:
        text = c["text"].strip()
        section = c["metadata"].get("section", "?")
        block = f"[{section}]\n{text}"
        if total + len(block) > _EXTRACT_SECTION_CHARS:
            break
        out.append(block)
        total += len(block)
    return "\n\n---\n\n".join(out)


def extract_numbers(chunks: list) -> list[dict]:
    """Extract checkable numeric claims from a paper's opening/closing sections."""
    context = _gather_number_sections(chunks)
    if not context.strip():
        return []

    raw = chat_completion(
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Paper sections:\n\n{context}\n\nExtract the numeric claims as JSON."},
        ],
        max_tokens=1400,
        temperature=0.1,
    )
    try:
        nums = parse_llm_json(raw, expect="array")
    except Exception as exc:
        print(f"[numbers_auditor] Extraction parse failed: {exc}")
        return []
    if not isinstance(nums, list):
        return []

    cleaned = []
    for n in nums:
        if isinstance(n, dict) and str(n.get("value", "")).strip():
            cleaned.append({
                "value":          str(n["value"]).strip(),
                "metric":         str(n.get("metric", "") or "").strip(),
                "dataset":        str(n.get("dataset", "") or "").strip(),
                "claim":          str(n.get("claim", "") or "").strip() or str(n["value"]).strip(),
                "source_section": str(n.get("source_section", "?") or "?").strip(),
            })
    return cleaned[:MAX_NUMBERS]


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — evidence retrieval (biased to results/tables)
# ───────────────────────────────────────────────────────────────────────────

def gather_evidence(num: dict, paper_id: str) -> list[dict]:
    """Retrieve the chunks most likely to contain the results number, biased
    toward results prose and linearized table cells."""
    # Query on the metric + dataset + the value itself so the matching table cell
    # surfaces even when prose doesn't repeat the number.
    query = " ".join(x for x in (num["metric"], num["dataset"], num["value"], num["claim"]) if x)
    try:
        return hybrid_retrieve(
            query, paper_id, top_k=EVIDENCE_PER_NUMBER, boost_terms=_EVIDENCE_BOOST,
        )
    except Exception as exc:
        print(f"[numbers_auditor] Evidence retrieval failed for a number: {exc}")
        return []


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — verdicts (batched, conservative)
# ───────────────────────────────────────────────────────────────────────────

_VERDICT_SYSTEM_PROMPT = """You are a meticulous fact-checker verifying whether the numbers a paper states
in its abstract/intro match the numbers in its own results.

For each numeric CLAIM you are given numbered EVIDENCE chunks retrieved from the
SAME paper (results prose and linearized tables, where a table cell looks like
"BERT — SQuAD F1: 0.79"). Decide one verdict per claim:

  MATCH      — The SAME number for the SAME metric/dataset appears in the
               evidence. Allow trivial formatting/rounding differences
               (92.4 vs 92.40; 0.924 vs 92.4%; 1.2M vs 1,200,000).
  MISMATCH   — The evidence reports a DIFFERENT number for the same
               metric/dataset than the claim states (e.g. claim says 92.4 but the
               table shows 91.2). This is the important finding.
  NOT_FOUND  — No corresponding number for this metric/dataset appears in the
               evidence at all.

CRITICAL RULES — never accuse without proof:
  - To return MISMATCH you MUST (a) cite the evidence chunk number in
    "evidence_ref" and (b) put the conflicting number you found in "found_value".
    If you cannot do BOTH, return NOT_FOUND — not MISMATCH.
  - Do NOT treat a different metric or a different dataset as a mismatch. 92.4 F1
    on SQuAD vs 88.1 F1 on a DIFFERENT dataset is not a mismatch — it's NOT_FOUND
    for SQuAD. Only compare like-for-like.
  - Retrieval is imperfect: a number missing from the evidence is NOT_FOUND, not
    MISMATCH. When unsure between MATCH and MISMATCH, choose MATCH.
  - "why" is ONE concrete sentence naming the two numbers (claimed vs found) or
    the absence.

Return ONLY a JSON array, one object per claim, in the SAME ORDER:

[
  {
    "index": 1,
    "verdict": "MATCH | MISMATCH | NOT_FOUND",
    "found_value": "the number found in the evidence, or null",
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
    """Grade a batch of {num, evidence} items. Each number gets its own numbered
    evidence block so the model's evidence_ref maps back to a real chunk."""
    blocks = []
    for i, item in enumerate(batch, 1):
        n = item["num_obj"]
        label = f"{n['value']}" + (f" {n['metric']}" if n["metric"] else "") + (f" on {n['dataset']}" if n["dataset"] else "")
        blocks.append(
            f"CLAIM {i} (from {n['source_section']}): {label}\n"
            f"Full phrase: \"{n['claim']}\"\n\n"
            f"EVIDENCE for claim {i}:\n{_format_evidence(item['evidence'])}"
        )
    user_prompt = (
        "\n\n========================================\n\n".join(blocks)
        + "\n\nVerify every number above. Return the JSON array (one object per claim, in order)."
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


def _attach_verdict(num_obj: dict, evidence: list[dict], verdict: dict | None) -> dict:
    """Merge a raw model verdict with the cited chunk's metadata, applying the
    conservative downgrade: a MISMATCH without a cited chunk AND a found value is
    not showable, so it becomes NOT_FOUND (we never accuse without the numbers)."""
    v = (verdict or {}).get("verdict", "NOT_FOUND")
    if v not in _VALID_VERDICTS:
        v = "NOT_FOUND"

    severity = (verdict or {}).get("severity", "med")
    if severity not in ("low", "med", "high"):
        severity = "med"

    why = str((verdict or {}).get("why", "")).strip()
    found_value = (verdict or {}).get("found_value")
    found_value = str(found_value).strip() if found_value not in (None, "") else ""

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

    # Conservative downgrade: a mismatch we can't show (no cited chunk or no
    # concrete found value) is not a mismatch.
    if v == "MISMATCH" and (cited is None or not found_value):
        v = "NOT_FOUND"
        found_value = ""
        if not why:
            why = "Could not locate a matching number in the retrieved results."

    return {
        "value":          num_obj["value"],
        "metric":         num_obj["metric"],
        "dataset":        num_obj["dataset"],
        "claim":          num_obj["claim"],
        "source_section": num_obj["source_section"],
        "verdict":        v,
        "found_value":    found_value if v == "MISMATCH" else "",
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
        "paper_id":          paper_id,
        "numbers_checked":   0,
        "matched":           0,
        "mismatched":        0,
        "not_found":         0,
        "consistency_score": None,
        "numbers":           [],
        "audit_failed":      failed,
        "reason":            reason,
        "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Mismatches first (actionable), then not-found, then matches.
_VERDICT_RANK = {"MISMATCH": 0, "NOT_FOUND": 1, "MATCH": 2}
_SEV_RANK     = {"high": 0, "med": 1, "low": 2}


def audit_numbers(paper_id: str, on_progress=None) -> dict:
    """Run the full numbers-consistency check for one paper and return a report.

    Never raises — returns an `audit_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    try:
        _emit(on_progress, "reading", "Reading the paper…")
        all_chunks = get_all_chunks(paper_id)
        if not all_chunks:
            return _empty_report(paper_id, failed=True, reason="No content found for this paper.")

        _emit(on_progress, "extracting", "Extracting the numbers it reports…")
        numbers = extract_numbers(all_chunks)
        if not numbers:
            return _empty_report(paper_id, failed=False, reason="No checkable numbers found in the abstract/intro.")

        _emit(on_progress, "gathering", f"Locating results for {len(numbers)} numbers…")
        items = [{"num_obj": n, "evidence": gather_evidence(n, paper_id)} for n in numbers]

        _emit(on_progress, "checking", "Reconciling each number against the results…")
        results: list[dict] = []
        for start in range(0, len(items), VERDICT_BATCH):
            batch = items[start:start + VERDICT_BATCH]
            try:
                verdicts = _verdict_batch(batch)
            except Exception as exc:
                print(f"[numbers_auditor] Verdict batch failed: {exc}")
                verdicts = []
            for i, item in enumerate(batch):
                results.append(_attach_verdict(item["num_obj"], item["evidence"],
                                               verdicts[i] if i < len(verdicts) else None))

        matched    = sum(1 for r in results if r["verdict"] == "MATCH")
        mismatched = sum(1 for r in results if r["verdict"] == "MISMATCH")
        not_found  = sum(1 for r in results if r["verdict"] == "NOT_FOUND")
        # Score = of the numbers we could locate in the results, how many match.
        # NOT_FOUND is excluded (retrieval may have missed it — not the paper's fault).
        locatable  = matched + mismatched
        score      = round(matched / locatable, 3) if locatable else None

        results.sort(key=lambda r: (
            _VERDICT_RANK.get(r["verdict"], 1),
            _SEV_RANK.get(r["severity"], 1),
        ))

        return {
            "paper_id":          paper_id,
            "numbers_checked":   len(results),
            "matched":           matched,
            "mismatched":        mismatched,
            "not_found":         not_found,
            "consistency_score": score,
            "numbers":           results,
            "audit_failed":      False,
            "reason":            "",
            "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[numbers_auditor] Numbers check failed: {exc}")
        return _empty_report(paper_id, failed=True, reason=str(exc))

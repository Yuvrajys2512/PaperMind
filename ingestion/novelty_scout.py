"""
ingestion/novelty_scout.py

Novelty / related-work scan for a draft in "write mode".

Every other analysis in this app reasons over ONE uploaded PDF. This one is
different: it searches *across the literature* to find the prior work closest to
a draft, so an author can see — before submitting — what they'll be compared to
and what still looks novel.

Pipeline (mirrors claim_auditor's shape: a pure-sync engine that never raises
and returns a report dict, so it slots into the same run_in_executor + SSE
plumbing as the claim/reviewer audits):

  1. read      — pull the draft's abstract/intro from its Chroma collection.
  2. querying  — one LLM call distils the opening into 2-3 focused search
                 queries (a keyword search, the way a researcher would phrase it,
                 beats dumping a whole abstract at a relevance API).
  3. comparing — synchronous Semantic Scholar search per query, merge/dedup,
                 then a batched LLM call rates each candidate for how closely it
                 overlaps the draft and what the draft still does differently.
  4. synthesis — one LLM call turns the rated neighbours into an overall
                 novelty read + concrete positioning advice.

Design stance mirrors the audits: this is a *thinking prompt*, not a verdict.
We surface the closest work and let the author judge novelty — we never assert a
draft is/ isn't novel. Closeness is about topical overlap of abstracts, nothing
more, and the author always sees the real papers behind every rating.

No local embedding model is used (keeps torch out of this import path — see the
docling/torch import-order footgun — and keeps cold-starts cheap on free hosts).
S2 relevance + LLM re-ranking is the v1; a SPECTER2 embedding re-rank is a clean
later add.

Public API
----------
find_related_work(paper_id, on_progress=None) -> dict
"""

from __future__ import annotations

import os
import time

import httpx

from ingestion.retriever import get_all_chunks
from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# ── Tunables ───────────────────────────────────────────────────────────────
MAX_QUERIES        = 3     # distinct search queries fired at Semantic Scholar
RESULTS_PER_QUERY  = 10    # candidates requested per query (pre-dedup)
MAX_CANDIDATES     = 10    # candidates kept for LLM rating (cost/latency bound)
RATE_BATCH         = 5     # candidates rated per LLM call
_OPENING_CHARS     = 5000  # cap on the abstract/intro context we distil from
_ABSTRACT_CHARS    = 1200  # cap on each candidate abstract fed to the rater

# Same claim-bearing opening sections the claim auditor keys on — the draft's
# topic lives in the abstract/intro.
_OPENING_SECTIONS = ("abstract", "introduction")

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,abstract,citationCount,venue,externalIds,openAccessPdf"

_VALID_CLOSENESS = {"HIGH", "MEDIUM", "LOW"}


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — pull the draft's opening
# ───────────────────────────────────────────────────────────────────────────

def _gather_opening(chunks: list) -> str:
    """Join the draft's abstract/intro bodies — the topic-bearing text.

    Falls back to the earliest non-table chunks when section detection labelled
    nothing (messy PDFs), since a paper's opening is almost always abstract+intro.
    """
    picked = [
        c for c in chunks
        if any(s in str(c["metadata"].get("section", "")).lower() for s in _OPENING_SECTIONS)
        and c["metadata"].get("section_type") != "table"
    ]
    if not picked:
        picked = [c for c in chunks if c["metadata"].get("section_type") != "table"][:5]

    out, total = [], 0
    for c in picked:
        text = c["text"].strip()
        section = c["metadata"].get("section", "?")
        block = f"[{section}]\n{text}"
        if total + len(block) > _OPENING_CHARS:
            break
        out.append(block)
        total += len(block)
    return "\n\n---\n\n".join(out)


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — distil search queries
# ───────────────────────────────────────────────────────────────────────────

_QUERY_SYSTEM_PROMPT = """You turn a paper's opening into literature-search queries.

You are given the Abstract/Introduction of a research paper. Produce the search
queries a careful author would use to find the CLOSEST PRIOR WORK to this paper —
the papers a reviewer would expect them to compare against.

Write queries the way a keyword search engine wants them: 4-8 words of the core
technical concepts (methods + problem + domain), NOT a full sentence and NOT the
paper's own title verbatim. Vary them so they cover the paper's main angles
(e.g. the method, the task, the application) rather than restating one idea.

Return ONLY JSON:
{
  "topic": "one plain-language sentence naming what this paper is about",
  "queries": ["query one", "query two", "query three"]
}
At most __MAX_QUERIES__ queries. No prose, no markdown.
""".replace("__MAX_QUERIES__", str(MAX_QUERIES))


def distil_queries(opening: str) -> dict:
    """LLM-distil the opening into {topic, queries}. Returns {} on failure."""
    raw = chat_completion(
        messages=[
            {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Paper opening:\n\n{opening}\n\nReturn the JSON."},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    try:
        obj = parse_llm_json(raw, expect="object")
    except Exception as exc:
        print(f"[novelty_scout] Query distillation parse failed: {exc}")
        return {}
    if not isinstance(obj, dict):
        return {}

    queries = obj.get("queries")
    if not isinstance(queries, list):
        queries = []
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:MAX_QUERIES]
    return {"topic": str(obj.get("topic", "")).strip(), "queries": queries}


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — Semantic Scholar search (synchronous) + merge
# ───────────────────────────────────────────────────────────────────────────

def _s2_headers() -> dict:
    h = {"User-Agent": "PaperMind/1.0 (research tool)"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def _search_s2_sync(query: str, limit: int = RESULTS_PER_QUERY) -> list[dict]:
    """Synchronous sibling of discovery.sources.semantic_scholar.search_* — kept
    sync so the whole engine runs inside run_in_executor with no event loop.

    Returns raw candidate dicts (full abstract kept, unlike the discovery client
    which truncates for card display). Never raises — a failed/limited search is
    an empty list so one bad query can't sink the scan.
    """
    params = {"query": query, "fields": _S2_FIELDS, "limit": limit}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(_S2_SEARCH_URL, params=params, headers=_s2_headers())
            if resp.status_code == 429:
                time.sleep(3)
                resp = client.get(_S2_SEARCH_URL, params=params, headers=_s2_headers())
            if resp.status_code != 200:
                print(f"[novelty_scout] S2 HTTP {resp.status_code} for {query!r}")
                return []
            raw = resp.json().get("data", []) or []
    except Exception as exc:
        print(f"[novelty_scout] S2 search failed for {query!r}: {exc}")
        return []

    out = []
    for p in raw:
        s2_id = p.get("paperId") or ""
        abstract = (p.get("abstract") or "").strip()
        external = p.get("externalIds") or {}
        arxiv_id = external.get("ArXiv")
        out.append({
            "s2_id":     s2_id,
            "title":     (p.get("title") or "").strip(),
            "authors":   [a.get("name", "") for a in (p.get("authors") or [])][:4],
            "year":      p.get("year"),
            "venue":     p.get("venue") or None,
            "citations": p.get("citationCount"),
            "abstract":  abstract,
            "url":       f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else (
                f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
        })
    return out


def _collect_candidates(queries: list[str]) -> list[dict]:
    """Run every query, merge, dedup by s2_id/title, keep only papers with an
    abstract (needed to judge overlap), cap at MAX_CANDIDATES."""
    seen: set[str] = set()
    merged: list[dict] = []
    for q in queries:
        for cand in _search_s2_sync(q):
            if not cand["abstract"] or not cand["title"]:
                continue
            key = cand["s2_id"] or cand["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cand)
    # Most-cited first as a mild prior on relevance before the LLM re-ranks.
    merged.sort(key=lambda c: (c["citations"] or 0), reverse=True)
    return merged[:MAX_CANDIDATES]


# ───────────────────────────────────────────────────────────────────────────
# Step 4 — rate each candidate against the draft
# ───────────────────────────────────────────────────────────────────────────

_RATE_SYSTEM_PROMPT = """You compare a DRAFT paper against candidate prior-work papers.

You are given the draft's abstract, then numbered CANDIDATE papers (title +
abstract). For each candidate decide how close it is to the draft and articulate
the relationship, so the draft's author can position their work.

For each candidate return:
  closeness      — HIGH   : same problem AND same core method/approach; a direct
                            competitor the author must differentiate against.
                   MEDIUM : clearly related (same problem OR same method) but a
                            meaningful difference in setup, scope, or approach.
                   LOW    : same broad area but tackles a different problem/method;
                            background, not a direct comparison.
  overlap        — ONE sentence: what concretely the draft shares with this paper.
  differentiator — ONE sentence: what the draft appears to do differently, or the
                   gap it fills, relative to this paper. If you cannot tell from
                   the abstracts, say "unclear from abstracts" — do not invent one.

Judge only from the abstracts provided. Do not assume unstated results.

Return ONLY a JSON array, one object per candidate, in the SAME ORDER, at most
one per candidate:

[
  {"index": 1, "closeness": "HIGH | MEDIUM | LOW", "overlap": "...", "differentiator": "..."}
]
"""


def _rate_batch(draft_abstract: str, batch: list[dict]) -> list[dict]:
    """Rate one batch of candidates. Returns raw verdict dicts (order-aligned)."""
    blocks = []
    for i, c in enumerate(batch, 1):
        blocks.append(
            f"CANDIDATE {i}:\nTitle: {c['title']}\nAbstract: {c['abstract'][:_ABSTRACT_CHARS]}"
        )
    user_prompt = (
        f"DRAFT abstract:\n{draft_abstract[:_ABSTRACT_CHARS * 2]}\n\n"
        "========================================\n\n"
        + "\n\n----------------------------------------\n\n".join(blocks)
        + "\n\nRate every candidate above. Return the JSON array (one object per candidate, in order)."
    )
    raw = chat_completion(
        messages=[
            {"role": "system", "content": _RATE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1400,
        temperature=0.1,
    )
    verdicts = parse_llm_json(raw, expect="array")
    return verdicts if isinstance(verdicts, list) else []


def _attach_rating(cand: dict, verdict: dict | None) -> dict:
    """Merge a candidate's S2 metadata with its LLM rating, normalising fields."""
    v = verdict or {}
    closeness = str(v.get("closeness", "LOW")).upper().strip()
    if closeness not in _VALID_CLOSENESS:
        closeness = "LOW"
    return {
        "title":          cand["title"],
        "authors":        cand["authors"],
        "year":           cand["year"],
        "venue":          cand["venue"],
        "citations":      cand["citations"],
        "url":            cand["url"],
        "closeness":      closeness,
        "overlap":        str(v.get("overlap", "")).strip(),
        "differentiator": str(v.get("differentiator", "")).strip(),
    }


# ───────────────────────────────────────────────────────────────────────────
# Step 5 — overall novelty synthesis
# ───────────────────────────────────────────────────────────────────────────

_SYNTH_SYSTEM_PROMPT = """You advise an author on how their draft sits in the literature.

You are given the draft's topic and a list of the closest prior-work papers
found (each with a closeness rating and how it overlaps the draft). Write a short,
honest positioning read for the author.

Rules:
  - Be a helpful colleague, not a judge. NEVER declare the work "novel" or "not
    novel" — you have only abstracts. Frame everything as what the author should
    check or address before submitting.
  - Ground statements in the papers found. If the closest work is only LOW
    closeness, say the space looks relatively open around this specific angle.
  - No hype, no filler.

Return ONLY JSON:
{
  "summary": "2-4 sentences: how crowded this space looks, which papers are the
              closest comparisons, and what that means for the author.",
  "positioning": ["a concrete, actionable bullet the author should do to position
                   the work (e.g. 'Compare directly against X on benchmark Y')",
                  "..."]
}
At most 4 positioning bullets. No prose outside the JSON.
"""


def _synthesize(topic: str, related: list[dict]) -> dict:
    """Turn rated neighbours into {summary, positioning}. Returns {} on failure."""
    if not related:
        return {}
    lines = []
    for r in related:
        lines.append(
            f"- [{r['closeness']}] {r['title']} ({r['year'] or 'n.d.'}): {r['overlap'] or 'related work'}"
        )
    user_prompt = (
        f"Draft topic: {topic or '(not given)'}\n\n"
        f"Closest prior work found ({len(related)} papers):\n" + "\n".join(lines)
        + "\n\nReturn the JSON."
    )
    raw = chat_completion(
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0.3,
    )
    try:
        obj = parse_llm_json(raw, expect="object")
    except Exception as exc:
        print(f"[novelty_scout] Synthesis parse failed: {exc}")
        return {}
    if not isinstance(obj, dict):
        return {}
    positioning = obj.get("positioning")
    if not isinstance(positioning, list):
        positioning = []
    positioning = [str(p).strip() for p in positioning if str(p).strip()][:4]
    return {"summary": str(obj.get("summary", "")).strip(), "positioning": positioning}


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
        "paper_id":         paper_id,
        "topic":            "",
        "queries_used":     [],
        "candidates_found": 0,
        "related":          [],
        "summary":          "",
        "positioning":      [],
        "scan_failed":      failed,
        "reason":           reason,
        "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Closeness ordering for the final sort — the direct competitors first.
_CLOSE_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def find_related_work(paper_id: str, on_progress=None) -> dict:
    """Run the full novelty scan for one draft and return a report dict.

    Never raises — returns a `scan_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    try:
        _emit(on_progress, "reading", "Reading the draft…")
        all_chunks = get_all_chunks(paper_id)
        if not all_chunks:
            return _empty_report(paper_id, failed=True, reason="No content found for this draft.")

        opening = _gather_opening(all_chunks)
        if not opening.strip():
            return _empty_report(paper_id, failed=False, reason="Couldn't find an abstract/intro to search from.")

        _emit(on_progress, "querying", "Distilling search queries…")
        distilled = distil_queries(opening)
        queries = distilled.get("queries") or []
        if not queries:
            return _empty_report(paper_id, failed=False, reason="Couldn't derive a search query from the draft.")

        _emit(on_progress, "searching", f"Searching the literature ({len(queries)} queries)…")
        candidates = _collect_candidates(queries)
        if not candidates:
            report = _empty_report(paper_id, failed=False,
                                   reason="No closely related papers surfaced — the space around this angle looks open (or search was rate-limited).")
            report["topic"] = distilled.get("topic", "")
            report["queries_used"] = queries
            return report

        _emit(on_progress, "comparing", f"Comparing against {len(candidates)} related papers…")
        related: list[dict] = []
        for start in range(0, len(candidates), RATE_BATCH):
            batch = candidates[start:start + RATE_BATCH]
            try:
                verdicts = _rate_batch(opening, batch)
            except Exception as exc:
                print(f"[novelty_scout] Rate batch failed: {exc}")
                verdicts = []
            for i, cand in enumerate(batch):
                related.append(_attach_rating(cand, verdicts[i] if i < len(verdicts) else None))

        # Direct competitors first (HIGH closeness), then by citation weight.
        related.sort(key=lambda r: (_CLOSE_RANK.get(r["closeness"], 2), -(r["citations"] or 0)))

        _emit(on_progress, "synthesizing", "Summarizing where the draft sits…")
        synth = _synthesize(distilled.get("topic", ""), related)

        close_counts = {
            "high":   sum(1 for r in related if r["closeness"] == "HIGH"),
            "medium": sum(1 for r in related if r["closeness"] == "MEDIUM"),
            "low":    sum(1 for r in related if r["closeness"] == "LOW"),
        }

        return {
            "paper_id":         paper_id,
            "topic":            distilled.get("topic", ""),
            "queries_used":     queries,
            "candidates_found": len(related),
            "closeness_counts": close_counts,
            "related":          related,
            "summary":          synth.get("summary", ""),
            "positioning":      synth.get("positioning", []),
            "scan_failed":      False,
            "reason":           "",
            "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[novelty_scout] Novelty scan failed: {exc}")
        return _empty_report(paper_id, failed=True, reason=str(exc))

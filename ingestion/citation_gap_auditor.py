"""
ingestion/citation_gap_auditor.py

Citation gap check — statements that assert something citable but carry no
citation.

The sixth write-mode audit, and the mirror image of claim_auditor. claim_auditor
asks "does the paper's evidence back the paper's own claims?"; this asks the
*outward* question: "where does the draft assert something about the world —
prior work, an external statistic, an attributed method — with nothing pointing
at a source?" Those are the sentences a reviewer marks "[citation needed]", and
they're easy to lose track of in a draft that's been rewritten a few times.

Pipeline (same pure-sync, never-raises shape as the sibling auditors, so it
slots into the identical run_in_executor + SSE plumbing):

  1. reading     — pull the citation-dense body sections from Chroma.
  2. scanning    — split into sentences and drop, deterministically, every one
                   that already carries a citation marker ([12], (Smith, 2020),
                   "Vaswani et al. (2017)", …). This regex pass is the primary
                   precision guard: it is cheap, high-recall on markers, and it
                   means the LLM only ever sees genuinely uncited sentences.
  3. classifying — batched LLM calls decide which of the surviving sentences
                   actually NEED a citation (vs. common knowledge, vs. the
                   authors' own contribution), and emit search keywords for the
                   ones that do.
  4. searching   — for a bounded sample of the strongest gaps, Semantic Scholar
                   is queried for a plausible citation, and one batched LLM call
                   confirms each suggestion actually supports the statement
                   before we show it.

Design stance (inherited from claim_auditor/numbers_auditor): a false accusation
is the only real failure — "[citation needed]" on a sentence that plainly doesn't
need one trains the author to ignore us. So every uncertain case is downgraded to
OK, never up. Three deterministic downgrades run *after* the model:
  - a citation marker in the sentence OR the one immediately after it → OK
    (markers routinely trail the claim into the next sentence),
  - a first-person contribution sentence ("we propose…") → OK (that's the claim
    auditor's territory, not a missing reference),
  - a flag with no concrete reason → OK (we never flag without something to show).
Suggested citations are likewise gated: a suggestion is shown only when a second
LLM pass confirms the paper actually supports the statement.

Public API
----------
audit_citation_gaps(paper_id, on_progress=None) -> dict
"""

from __future__ import annotations

import os
import re
import time

import httpx

from ingestion.retriever import get_all_chunks
from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# ── Tunables ───────────────────────────────────────────────────────────────
MAX_CANDIDATES   = 32   # uncited sentences sent to the LLM (cost/latency bound)
CLASSIFY_BATCH   = 8    # sentences classified per LLM call
MAX_GAPS         = 12   # gaps kept in the report
S2_LOOKUPS       = 4    # Semantic Scholar searches per run (novelty scan does 3)
RESULTS_PER_LOOKUP = 5  # candidates requested per S2 search
_SECTION_CHARS   = 24000  # cap on the body text we scan
_MIN_SENTENCE    = 60   # chars — below this it's a heading/fragment, not a claim
_MAX_SENTENCE    = 400  # chars — above this the "sentence" is a split failure
_FINGERPRINT_CHARS = 4000  # draft text used to recognise the draft in S2 results
_SELF_MATCH_RATIO  = 0.65  # candidate-abstract overlap above this = the draft itself

# Section filtering is a DENY-list, not an allow-list. Real headings are
# numbered and idiosyncratic ("3 Model Architecture", "4 Why Self-Attention"), so
# an allow-list of expected names silently skips most of a paper — and citation
# gaps live everywhere, not just the intro ("we use the Adam optimizer" in the
# training section is exactly the attributed-method case).
#
# The abstract IS excluded: most venues discourage citations there, so scanning
# it would manufacture false positives. References/acknowledgements/appendices
# carry no prose claims to source.
_SKIP_SECTIONS = ("reference", "bibliograph", "acknowledg", "appendix", "abstract")

_VALID_VERDICTS = {"NEEDS_CITATION", "OK"}
_VALID_KINDS = {
    "prior_work", "external_stat", "attributed_method",
    "comparison", "established_fact",
}

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,abstract,citationCount,venue,externalIds"
_ABSTRACT_CHARS = 900   # cap on each candidate abstract fed to the verifier


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — gather the citation-dense body
# ───────────────────────────────────────────────────────────────────────────

def _gather_body(chunks: list) -> list[dict]:
    """Pick the prose body chunks, in reading order — everything except tables
    and the deny-listed sections."""
    def _named(c) -> str:
        return str(c["metadata"].get("section", "")).lower()

    picked = [
        c for c in chunks
        if c["metadata"].get("section_type") != "table"
        and not any(s in _named(c) for s in _SKIP_SECTIONS)
    ]

    out, total = [], 0
    for c in picked:
        total += len(c.get("text", ""))
        if total > _SECTION_CHARS:
            break
        out.append(c)
    return out


_WORD_RE = re.compile(r"[a-z]{4,}")


def _fingerprint(text: str) -> set[str]:
    """Content-word set used to recognise the draft in search results."""
    return set(_WORD_RE.findall(text.lower()))


def _draft_fingerprint(chunks: list) -> set[str]:
    """Fingerprint the draft's own opening (abstract + first prose)."""
    opening = " ".join(
        c.get("text", "") for c in chunks
        if c["metadata"].get("section_type") != "table"
    )[:_FINGERPRINT_CHARS]
    return _fingerprint(opening)


def _is_self_match(candidate: dict, draft_fp: set[str]) -> bool:
    """True when a search result is (a copy of) the draft itself.

    A draft is often derived from — or literally is — a paper already indexed by
    Semantic Scholar, and "you should cite this" pointing at the author's own
    paper is the most embarrassing output this feature could produce. Content
    words of the same paper's abstract sit almost entirely inside the draft;
    genuinely different papers on the same topic land far below the threshold.
    """
    cand_fp = _fingerprint(candidate.get("abstract", "") or candidate.get("title", ""))
    if len(cand_fp) < 12:  # too little text to judge — don't reject on noise
        return False
    return len(cand_fp & draft_fp) / len(cand_fp) >= _SELF_MATCH_RATIO


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — sentence split + deterministic citation-marker detection
# ───────────────────────────────────────────────────────────────────────────

# Abbreviations that end in a period and must not end a sentence.
_ABBREV = (
    "et al", "e.g", "i.e", "cf", "vs", "Fig", "Eq", "Sec", "Tab", "Ref",
    "approx", "resp", "etc", "Dr", "Prof", "Mr", "Ms", "St", "al",
)
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Citation markers, deliberately over-inclusive — a false "this is cited" only
# costs us a missed gap, while a false "uncited" is the failure mode we fear.
_CITATION_PATTERNS = [
    re.compile(r"\[\s*\d+\s*(?:[,;\-–]\s*\d+\s*)*\]"),                    # [12], [3, 4], [5-8]
    re.compile(r"\[[^\]]{0,80}\b(?:19|20)\d{2}[a-z]?\b[^\]]{0,80}\]"),    # [Smith, 2020]
    re.compile(r"\([^()]{0,120}\b(?:19|20)\d{2}[a-z]?\b[^()]{0,40}\)"),   # (Smith et al., 2020)
    re.compile(r"\bet\s+al\.?"),                                          # narrative "Smith et al."
    re.compile(r"\b[A-Z][A-Za-z\-']+\s+(?:and|&)\s+[A-Z][A-Za-z\-']+\s*\(?\s*(?:19|20)\d{2}"),
    re.compile(r"\[\s*(?:\d+\s*[,;]\s*)*\d+\s*\]"),                       # dense numeric groups
]


def _has_citation(text: str) -> bool:
    """True if the text carries anything that looks like a citation marker."""
    return any(p.search(text) for p in _CITATION_PATTERNS)


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences, protecting the abbreviations that show up in
    exactly the sentences we care about ('et al.', 'e.g.').

    PDF extraction wraps mid-sentence, so whitespace is collapsed first — both so
    the splitter sees real sentence boundaries and so the statement we quote back
    to the author reads as prose rather than as column fragments.
    """
    text = re.sub(r"\s+", " ", text)
    # Mask abbreviation periods so the splitter can't break on them.
    masked = _ABBREV_RE.sub(lambda m: m.group(1) + "\x00", text)
    parts = _SENT_SPLIT_RE.split(masked)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def collect_uncited(chunks: list) -> tuple[list[dict], int, int]:
    """Split the body into sentences and return the ones with NO citation marker.

    Returns (candidates, sentences_scanned, cited_sentences). A candidate carries
    the sentence plus the chunk metadata needed for the PDF jump, and the
    following sentence — the marker guard re-checks it later, because authors
    routinely place the citation in the sentence after the claim.
    """
    candidates: list[dict] = []
    scanned = cited = 0

    for c in chunks:
        section = c["metadata"].get("section", "?")
        sentences = _split_sentences(c.get("text", ""))
        for i, s in enumerate(sentences):
            if len(s) < _MIN_SENTENCE or len(s) > _MAX_SENTENCE:
                continue
            scanned += 1
            if _has_citation(s):
                cited += 1
                continue
            candidates.append({
                "statement":  s,
                "next":       sentences[i + 1] if i + 1 < len(sentences) else "",
                "section":    section,
                "page":       c["metadata"].get("page_num"),
                "quote":      c.get("text", "")[:600],
                "section_type": c["metadata"].get("section_type", "text"),
            })

    return _sample_across_sections(candidates), scanned, cited


def _sample_across_sections(candidates: list[dict]) -> list[dict]:
    """Trim to MAX_CANDIDATES while keeping every section represented.

    Taking the first N in reading order would spend the whole budget on the
    introduction, which is where uncited sentences are densest — and miss the
    attributed-method cases ("we use the Adam optimizer") that sit deep in the
    training/experiments sections. So we round-robin across sections instead,
    then restore reading order.
    """
    if len(candidates) <= MAX_CANDIDATES:
        return candidates

    by_section: dict[str, list[dict]] = {}
    for idx, cand in enumerate(candidates):
        by_section.setdefault(str(cand["section"]), []).append((idx, cand))

    picked: list[tuple[int, dict]] = []
    round_i = 0
    while len(picked) < MAX_CANDIDATES:
        added = False
        for bucket in by_section.values():
            if round_i < len(bucket):
                picked.append(bucket[round_i])
                added = True
                if len(picked) >= MAX_CANDIDATES:
                    break
        if not added:
            break
        round_i += 1

    picked.sort(key=lambda pair: pair[0])
    return [cand for _, cand in picked]


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — classify which uncited sentences actually need a citation
# ───────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM_PROMPT = """You are a meticulous, conservative reviewer looking for sentences in a paper
draft that assert something CITABLE but carry no citation.

Every sentence you are given has already been checked automatically and contains
NO citation marker. Your only job is to decide, for each one, whether a reviewer
would legitimately write "[citation needed]" next to it.

Return NEEDS_CITATION only when the sentence attributes something to the outside
world that the paper itself does not establish:
  prior_work        — describes what other work has done, shown, or proposed
                      ("previous approaches rely on X", "it has been shown that Y")
  external_stat     — a number, dataset property, or fact from outside this paper
                      ("X accounts for 40% of cases", "the corpus contains 2M docs")
  attributed_method — names a method, model, architecture, or dataset that someone
                      else introduced and that the paper builds on ("we fine-tune
                      BERT", "using the Adam optimizer")
  comparison        — asserts how something external behaves or performs
                      ("transformers outperform RNNs on long sequences")

Return OK for everything else. OK covers, and this is most sentences:
  established_fact  — textbook/common knowledge in the field, definitions, or
                      statements so standard that no reviewer would ask for a source
  - anything about THIS paper: "we propose", "our method", "in Section 3",
    "Table 2 shows", the authors' own results, setup, or motivation
  - vague or rhetorical framing with no factual assertion
  - sentences you cannot confidently judge out of context

CRITICAL RULES — bias hard toward OK:
  - When you are unsure whether a citation is genuinely expected, return OK.
    A wrong "[citation needed]" is far more damaging than a missed one.
  - NEVER flag a sentence about the authors' own contribution, method, or results.
  - NEVER flag on style alone. Flag only when the sentence makes a specific
    factual assertion that a source could support.
  - "why" must be ONE sentence naming the concrete thing that needs a source
    (the attributed fact, the named method, the statistic) — not generic advice.

For each sentence flagged NEEDS_CITATION also return "search_terms": 4-8 keywords
(not a sentence) that would find the paper that should be cited.

Return ONLY a JSON array, one object per sentence, in the SAME ORDER:

[
  {
    "index": 1,
    "verdict": "NEEDS_CITATION | OK",
    "kind": "prior_work | external_stat | attributed_method | comparison | established_fact",
    "severity": "low | med | high",
    "why": "one concrete sentence",
    "search_terms": "keywords for the missing citation, or \\"\\" when OK"
  }
]
"""

# Sentences about the authors' own work are never citation gaps — this is the
# claim auditor's territory. Matched as a deterministic post-guard because the
# model occasionally flags "we adopt X" as an attribution.
_FIRST_PERSON_RE = re.compile(
    r"\b(?:we|our|this paper|this work|this study|in this section|the present work)\b",
    re.IGNORECASE,
)
# …but an explicit attribution inside a first-person sentence IS still citable
# ("we fine-tune BERT"), so the guard only fires when nothing external is named.
_ATTRIBUTION_HINT_RE = re.compile(
    r"\b(?:prior|previous|existing|recent|earlier)\s+(?:work|approaches|methods|studies|models)\b"
    r"|\bhas\s+been\s+shown\b|\bare\s+known\s+to\b|\bintroduced\s+by\b|\bproposed\s+by\b",
    re.IGNORECASE,
)


def _classify_batch(batch: list[dict]) -> list[dict]:
    """Classify one batch of uncited sentences. Returns raw verdicts in order."""
    blocks = []
    for i, cand in enumerate(batch, 1):
        blocks.append(
            f"SENTENCE {i} (from {cand['section']}):\n\"{cand['statement']}\""
        )
    user_prompt = (
        "\n\n".join(blocks)
        + "\n\nJudge every sentence above. Return the JSON array (one object per sentence, in order)."
    )
    raw = chat_completion(
        messages=[
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1400,
        temperature=0.0,
    )
    verdicts = parse_llm_json(raw, expect="array")
    return verdicts if isinstance(verdicts, list) else []


def _attach_verdict(cand: dict, verdict: dict | None) -> dict:
    """Merge a raw model verdict with the sentence's metadata, applying the
    conservative downgrades. A gap we can't justify is not a gap."""
    v = str((verdict or {}).get("verdict", "OK")).upper().strip()
    if v not in _VALID_VERDICTS:
        v = "OK"

    kind = str((verdict or {}).get("kind", "") or "").lower().strip()
    if kind not in _VALID_KINDS:
        kind = "prior_work" if v == "NEEDS_CITATION" else "established_fact"

    severity = str((verdict or {}).get("severity", "med")).lower().strip()
    if severity not in ("low", "med", "high"):
        severity = "med"

    why = str((verdict or {}).get("why", "") or "").strip()
    terms = str((verdict or {}).get("search_terms", "") or "").strip()

    # Guard 1 — a marker in the FOLLOWING sentence still cites this claim.
    # Authors routinely write "X has been shown to work. See [12] for details."
    if v == "NEEDS_CITATION" and _has_citation(cand.get("next", "")):
        v = "OK"

    # Guard 2 — the authors' own contribution is not a missing reference, unless
    # the sentence explicitly attributes something outward.
    if (
        v == "NEEDS_CITATION"
        and _FIRST_PERSON_RE.search(cand["statement"])
        and not _ATTRIBUTION_HINT_RE.search(cand["statement"])
    ):
        v = "OK"

    # Guard 3 — never flag without a concrete reason to show the author.
    if v == "NEEDS_CITATION" and not why:
        v = "OK"

    return {
        "statement":    cand["statement"],
        "section":      cand["section"],
        "page":         cand["page"],
        "verdict":      v,
        "kind":         kind,
        "severity":     severity,
        "why":          why,
        "search_terms": terms if v == "NEEDS_CITATION" else "",
        "suggestion":   None,
        "evidence": {
            "section":      cand["section"],
            "page":         cand["page"],
            "section_type": cand.get("section_type", "text"),
            "quote":        cand.get("quote", ""),
        },
    }


# ───────────────────────────────────────────────────────────────────────────
# Step 4 — suggest a citation via Semantic Scholar (bounded)
# ───────────────────────────────────────────────────────────────────────────

def _s2_headers() -> dict:
    h = {"User-Agent": "PaperMind/1.0 (research tool)"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def _search_s2_sync(query: str, limit: int = RESULTS_PER_LOOKUP) -> list[dict]:
    """Synchronous S2 search — same shape/contract as novelty_scout's client
    (sync so the whole engine runs in run_in_executor with no event loop, and
    never raises so one throttled query can't sink the audit)."""
    params = {"query": query, "fields": _S2_FIELDS, "limit": limit}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(_S2_SEARCH_URL, params=params, headers=_s2_headers())
            if resp.status_code == 429:
                time.sleep(3)
                resp = client.get(_S2_SEARCH_URL, params=params, headers=_s2_headers())
            if resp.status_code != 200:
                print(f"[citation_gap_auditor] S2 HTTP {resp.status_code} for {query!r}")
                return []
            raw = resp.json().get("data", []) or []
    except Exception as exc:
        print(f"[citation_gap_auditor] S2 search failed for {query!r}: {exc}")
        return []

    out = []
    for p in raw:
        s2_id = p.get("paperId") or ""
        external = p.get("externalIds") or {}
        arxiv_id = external.get("ArXiv")
        out.append({
            "title":     (p.get("title") or "").strip(),
            "authors":   [a.get("name", "") for a in (p.get("authors") or [])][:4],
            "year":      p.get("year"),
            "venue":     p.get("venue") or None,
            "citations": p.get("citationCount"),
            "abstract":  (p.get("abstract") or "").strip(),
            "url":       f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else (
                f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
        })
    return [c for c in out if c["title"]]


_SUGGEST_SYSTEM_PROMPT = """You check whether a candidate paper could actually be cited for a statement.

You are given numbered CASES. Each has a STATEMENT from a paper draft that is
missing a citation, and one CANDIDATE paper (title + abstract) found by a
literature search.

For each case decide whether citing that candidate for that statement would be
defensible:
  SUPPORTS — the candidate's own content plainly covers what the statement
             asserts (it introduced the named method, it reports that result, it
             established that finding). An author could cite it and a reviewer
             would accept it.
  UNCLEAR  — merely same-topic, tangential, or you cannot tell from the abstract.

Be strict. A wrong suggestion is worse than none: authors trust suggestions and
mis-citation is a real academic error. When in any doubt, return UNCLEAR.

Return ONLY a JSON array, one object per case, in the SAME ORDER:

[
  {"index": 1, "relation": "SUPPORTS | UNCLEAR", "note": "one short sentence on why this paper fits"}
]
"""


def _verify_suggestions(pairs: list[tuple[dict, dict]]) -> list[dict]:
    """One batched LLM call confirming each (gap, candidate) pair is citable.
    Returns raw relation dicts aligned to pair order."""
    if not pairs:
        return []
    blocks = []
    for i, (gap, cand) in enumerate(pairs, 1):
        blocks.append(
            f"CASE {i}:\nSTATEMENT: \"{gap['statement']}\"\n"
            f"CANDIDATE: {cand['title']} ({cand['year'] or 'n.d.'})\n"
            f"Abstract: {cand['abstract'][:_ABSTRACT_CHARS] or '(no abstract)'}"
        )
    user_prompt = (
        "\n\n----------------------------------------\n\n".join(blocks)
        + "\n\nJudge every case above. Return the JSON array (one object per case, in order)."
    )
    raw = chat_completion(
        messages=[
            {"role": "system", "content": _SUGGEST_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=900,
        temperature=0.0,
    )
    rels = parse_llm_json(raw, expect="array")
    return rels if isinstance(rels, list) else []


def attach_suggestions(gaps: list[dict], draft_fp: set[str]) -> int:
    """Look up a plausible citation for the strongest gaps and attach the ones a
    verification pass confirms. Bounded to S2_LOOKUPS searches + 1 LLM call per
    run — same external-call budget as the novelty scan.

    Returns the number of suggestions actually attached.
    """
    targets = [g for g in gaps if g["search_terms"]][:S2_LOOKUPS]
    pairs: list[tuple[dict, dict]] = []
    for gap in targets:
        candidates = _search_s2_sync(gap["search_terms"])
        # Never suggest the draft itself back to its own author.
        candidates = [c for c in candidates if not _is_self_match(c, draft_fp)]
        # Most-cited first — for "which paper should be cited here", citation
        # weight is a decent prior on the canonical reference.
        candidates.sort(key=lambda c: (c["citations"] or 0), reverse=True)
        if candidates:
            pairs.append((gap, candidates[0]))

    try:
        relations = _verify_suggestions(pairs)
    except Exception as exc:
        print(f"[citation_gap_auditor] Suggestion verification failed: {exc}")
        relations = []

    attached = 0
    for i, (gap, cand) in enumerate(pairs):
        rel = relations[i] if i < len(relations) else None
        if str((rel or {}).get("relation", "")).upper().strip() != "SUPPORTS":
            continue  # conservative: only confirmed suggestions are shown
        gap["suggestion"] = {
            "title":     cand["title"],
            "authors":   cand["authors"],
            "year":      cand["year"],
            "venue":     cand["venue"],
            "citations": cand["citations"],
            "url":       cand["url"],
            "note":      str((rel or {}).get("note", "") or "").strip(),
        }
        attached += 1
    return attached


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
        "paper_id":            paper_id,
        "sentences_scanned":   0,
        "cited_sentences":     0,
        "statements_checked":  0,
        "gaps_found":          0,
        "suggestions_found":   0,
        "coverage_score":      None,
        "gaps":                [],
        "audit_failed":        failed,
        "reason":              reason,
        "generated_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_KIND_RANK = {"external_stat": 0, "prior_work": 1, "comparison": 2, "attributed_method": 3}
_SEV_RANK  = {"high": 0, "med": 1, "low": 2}


def audit_citation_gaps(paper_id: str, on_progress=None) -> dict:
    """Run the full citation-gap check for one draft and return a report dict.

    Never raises — returns an `audit_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    try:
        _emit(on_progress, "reading", "Reading the draft…")
        all_chunks = get_all_chunks(paper_id)
        if not all_chunks:
            return _empty_report(paper_id, failed=True, reason="No content found for this draft.")

        body = _gather_body(all_chunks)
        if not body:
            return _empty_report(paper_id, failed=False,
                                 reason="Couldn't find body sections to check for citations.")

        _emit(on_progress, "scanning", "Scanning sentences for citation markers…")
        candidates, scanned, cited = collect_uncited(body)
        if not candidates:
            report = _empty_report(paper_id, failed=False,
                                   reason="Every statement we could check already carries a citation.")
            report["sentences_scanned"] = scanned
            report["cited_sentences"]   = cited
            report["coverage_score"]    = 1.0 if cited else None
            return report

        _emit(on_progress, "classifying", f"Checking {len(candidates)} uncited statements…")
        results: list[dict] = []
        for start in range(0, len(candidates), CLASSIFY_BATCH):
            batch = candidates[start:start + CLASSIFY_BATCH]
            try:
                verdicts = _classify_batch(batch)
            except Exception as exc:
                print(f"[citation_gap_auditor] Classify batch failed: {exc}")
                verdicts = []
            # Align by order; the index field is advisory only.
            for i, cand in enumerate(batch):
                results.append(_attach_verdict(cand, verdicts[i] if i < len(verdicts) else None))

        gaps = [r for r in results if r["verdict"] == "NEEDS_CITATION"]
        # Strongest, most concrete gaps first — that's also the order the bounded
        # S2 lookup budget is spent in.
        gaps.sort(key=lambda g: (
            _SEV_RANK.get(g["severity"], 1),
            _KIND_RANK.get(g["kind"], 4),
        ))
        gaps = gaps[:MAX_GAPS]

        suggestions = 0
        if gaps:
            _emit(on_progress, "searching", "Looking for citations you may be missing…")
            suggestions = attach_suggestions(gaps, _draft_fingerprint(all_chunks))

        # Coverage = of the statements that appear to need support, how many
        # already carry a citation. Sentences with a marker are presumed citable
        # (they were cited *because* they needed it), so they're the numerator;
        # confirmed gaps are the only thing added to the denominator. Sentences
        # the model cleared as common knowledge are excluded from both — they
        # never needed a citation, so they'd only inflate the score.
        citable = cited + len(gaps)
        coverage = round(cited / citable, 3) if citable else None

        return {
            "paper_id":            paper_id,
            "sentences_scanned":   scanned,
            "cited_sentences":     cited,
            "statements_checked":  len(results),
            "gaps_found":          len(gaps),
            "suggestions_found":   suggestions,
            "coverage_score":      coverage,
            "gaps":                gaps,
            "audit_failed":        False,
            "reason":              "",
            "generated_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[citation_gap_auditor] Citation gap check failed: {exc}")
        return _empty_report(paper_id, failed=True, reason=str(exc))

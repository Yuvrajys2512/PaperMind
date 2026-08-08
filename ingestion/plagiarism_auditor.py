"""
ingestion/plagiarism_auditor.py

Overlap check — passages in a draft that also appear in the papers the author
already has in their PaperMind library.

The seventh write-mode audit. What it honestly is, and is not:

  It is NOT a Turnitin replacement. Real plagiarism detection needs a web-scale
  index of every published paper, thesis and web page, which no free stack can
  build. Claiming otherwise would be dishonest to the user.

  It IS a check against the corpus that actually matters for accidental reuse:
  the papers the author uploaded and has been reading, quoting and drafting
  from. Unattributed reuse overwhelmingly comes from the sources on your own
  desk — a definition pasted from a related-work paper while drafting, a method
  paragraph carried over from your own earlier submission. Those are exactly the
  documents in the library, and we have their FULL TEXT, so overlap here is
  provable rather than inferred.

Two signals, deliberately ranked:

  1. Shingles (primary). Word 8-grams are hashed into an index over the corpus;
     a hit is extended greedily into the longest contiguous run of identical
     words. A shingle match is *evidence* — we can show the author the exact
     shared string and where it came from. High precision, zero interpretation.

  2. Embeddings (secondary, lower confidence). Verbatim matching is blind to
     paraphrase, so a small bounded pass compares uncovered draft passages
     against the corpus by cosine similarity. Cosine alone is useless as a
     verdict — two papers on the same topic score high without sharing a
     sentence — so a match is only ever shown after one LLM pass confirms the
     two passages state the SAME CONTENT rather than merely sharing a topic.

Same pure-sync, never-raises shape as the sibling auditors, so it slots into
the identical run_in_executor + SSE plumbing.

Pipeline (stages: reading -> indexing -> matching -> paraphrase):
  1. reading    — pull the draft's prose body from Chroma.
  2. indexing   — pull each corpus paper and build the shingle index.
  3. matching   — greedy longest-run matching + the deterministic downgrades.
  4. paraphrase — bounded embedding probes, gated by one LLM confirmation call.

Design stance (inherited from the sibling auditors): a false accusation is the
only real failure, and here it is the *most* damaging one in the product —
telling a researcher they plagiarised when they didn't is not a bad suggestion,
it's an insult. So every deterministic guard runs one-way toward "not a flag":
  - a match whose surrounding text carries a citation marker or quotation marks
    is ATTRIBUTED, not a flag — the author already did the right thing,
  - a string found in 3+ different corpus papers is standard field phrasing
    ("we evaluate on the standard train/dev/test split"), not reuse — dropped,
  - a match too thin in content words (mostly stopwords, numbers, boilerplate)
    is dropped,
  - a corpus paper that overlaps the draft wholesale is the SAME DOCUMENT (the
    draft is also in the library), excluded with a note rather than reported as
    100% plagiarised,
  - references, bibliographies and tables are excluded on both sides — shared
    reference lists and shared benchmark numbers are not reuse.

Public API
----------
audit_overlap(paper_id, corpus, on_progress=None) -> dict
    corpus: [{"paper_id": str, "title": str}, ...] — the other papers to compare
    against. Passed IN rather than looked up so this engine stays free of any
    Postgres coupling, like every other audit engine.
"""

from __future__ import annotations

import re
import time

import numpy as np

from ingestion.llm_client import chat_completion
from ingestion.json_utils import parse_llm_json

# ingestion.retriever imports ingestion.models, which pulls in sentence-
# transformers/torch at import time — and importing torch in the same process as
# docling segfaults on Windows (see the note in write_mode_handoff.md §7). Every
# retriever/model import in this module is therefore deferred into the function
# that needs it, exactly as reviewer_auditor and structure_auditor do. Without
# this, merely collecting this module's tests alongside the table-extractor
# tests kills the interpreter.

# ── Tunables ───────────────────────────────────────────────────────────────
SHINGLE_W          = 8     # words per shingle — the unit of the index
MIN_MATCH_WORDS    = 12    # shortest run reported at all
LONG_MATCH_WORDS   = 25    # at/above this a run is "verbatim", not "near"
MIN_CONTENT_WORDS  = 6     # distinct non-stopword words a match must contain
COMMON_SOURCE_LIMIT = 3    # a string in this many papers is field boilerplate
MAX_POSTINGS       = 8     # postings kept per shingle (cap on hot boilerplate)
MAX_CORPUS_PAPERS  = 25    # corpus papers compared per run
MAX_CORPUS_TOKENS  = 24000 # words indexed per corpus paper
MAX_DRAFT_TOKENS   = 24000 # words scanned in the draft
MAX_MATCHES        = 20    # matches kept in the report
SAME_DOC_RATIO     = 0.45  # source covering this much of the draft = same doc

MAX_PROBES         = 8     # draft passages sent through the embedding pass
MIN_PROBE_CHARS    = 300   # shorter passages are too generic to compare
PROBE_CHARS        = 1500  # cap on the text embedded per probe
PARAPHRASE_SIM     = 0.92  # cosine floor before the LLM is even asked
_EXCERPT_CHARS     = 700   # cap on each passage shown to the LLM / the user
_CONTEXT_CHARS     = 220   # text either side of a match used for the guards

# Reference lists overlap heavily between any two papers in a field (same cited
# titles, verbatim), and result tables share benchmark names and numbers. Both
# would drown the real signal in false positives, so both are excluded from the
# draft AND from every corpus paper. The abstract stays IN — unlike the citation
# gap check, reused text in an abstract is exactly what we want to catch.
_SKIP_SECTIONS = ("reference", "bibliograph", "acknowledg")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Deliberately small: only words so common that a run made of them carries no
# evidence. Anything domain-specific stays a content word.
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
he her his how i if in into is it its may might more most must no not of on or
our out over said she should so some such than that the their them then there
these they this those to too us was we were what when where which while who
why will with would you your it's we've
""".split())

# Citation markers and quotation marks around a match mean the author already
# credited the source. Same over-inclusive stance as citation_gap_auditor: a
# false "this is attributed" costs us one missed flag, a false accusation costs
# the user's trust.
_CITATION_PATTERNS = [
    re.compile(r"\[\s*\d+\s*(?:[,;\-–]\s*\d+\s*)*\]"),                    # [12], [3, 4], [5-8]
    re.compile(r"\[[^\]]{0,80}\b(?:19|20)\d{2}[a-z]?\b[^\]]{0,80}\]"),    # [Smith, 2020]
    re.compile(r"\([^()]{0,120}\b(?:19|20)\d{2}[a-z]?\b[^()]{0,40}\)"),   # (Smith et al., 2020)
    re.compile(r"\bet\s+al\.?"),                                          # narrative "Smith et al."
    re.compile(r"\b[A-Z][A-Za-z\-']+\s+(?:and|&)\s+[A-Z][A-Za-z\-']+\s*\(?\s*(?:19|20)\d{2}"),
]
_QUOTE_RE = re.compile(r"[\"“”«»]|(?<!\w)'{1}(?=\w)")


def _has_citation(text: str) -> bool:
    """True if the text carries anything that looks like a citation marker."""
    return any(p.search(text) for p in _CITATION_PATTERNS)


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — tokenize a paper into a flat, position-addressable word stream
# ───────────────────────────────────────────────────────────────────────────

class _Doc:
    """A paper flattened into one word stream that still knows where every word
    came from.

    Shingle matching needs a single flat sequence (matches must run across chunk
    boundaries), but the report needs to quote the *original* text and jump to a
    page. So each token carries its chunk index and its character span inside
    that chunk, and excerpts are sliced out of the untouched chunk text rather
    than rebuilt from tokens — the author sees their own words, punctuation and
    casing intact.
    """

    __slots__ = ("paper_id", "title", "words", "chunk_of", "starts", "ends", "chunks")

    def __init__(self, paper_id: str, title: str):
        self.paper_id = paper_id
        self.title    = title
        self.words: list[str]    = []   # normalised (lowercase) words
        self.chunk_of: list[int] = []   # index into self.chunks
        self.starts: list[int]   = []   # char offset of the word in its chunk
        self.ends: list[int]     = []
        self.chunks: list[dict]  = []

    def add_chunk(self, chunk: dict, limit: int) -> bool:
        """Append one chunk's words. Returns False once the token budget is hit."""
        idx  = len(self.chunks)
        text = chunk.get("text", "") or ""
        self.chunks.append(chunk)
        for m in _TOKEN_RE.finditer(text):
            if len(self.words) >= limit:
                return False
            self.words.append(m.group(0).lower())
            self.chunk_of.append(idx)
            self.starts.append(m.start())
            self.ends.append(m.end())
        return True

    def _clip(self, i: int, j: int) -> tuple[int, int, str]:
        """Clip the word span [i, j) to the chunk it starts in.

        Matches can run across a chunk boundary, but character offsets are only
        meaningful inside their own chunk — slicing chunk A's text with chunk B's
        offsets would quote something the author never wrote. Clipping loses the
        tail of a rare cross-boundary match; a shortened true quote is a fine
        price for never showing a false one.
        """
        c = self.chunk_of[i]
        end_word = min(j, len(self.words)) - 1
        while end_word > i and self.chunk_of[end_word] != c:
            end_word -= 1
        return end_word, c, self.chunks[c].get("text", "") or ""

    def excerpt(self, i: int, j: int) -> str:
        """The original text spanning words [i, j) — sliced from the source text
        so punctuation and casing survive."""
        if i >= len(self.words):
            return ""
        end_word, _c, text = self._clip(i, j)
        return re.sub(r"\s+", " ", text[self.starts[i]:self.ends[end_word]]).strip()

    def context(self, i: int, j: int) -> tuple[str, str]:
        """(text around the match, the match itself) — what the guards read.

        The two are returned separately on purpose. A citation marker INSIDE the
        matched span was copied along with the text; it is the original author's
        citation, not this author's act of attribution, so it must not count as
        credit. Only markers the author wrote *around* the reused passage do.
        """
        if i >= len(self.words):
            return "", ""
        end_word, _c, text = self._clip(i, j)
        m_lo, m_hi = self.starts[i], self.ends[end_word]
        lo = max(0, m_lo - _CONTEXT_CHARS)
        hi = min(len(text), m_hi + _CONTEXT_CHARS)
        return text[lo:m_lo] + "  " + text[m_hi:hi], text[m_lo:m_hi]

    def locate(self, i: int) -> tuple[str, object]:
        """(section, page) for the word at position i — used for the PDF jump."""
        meta = self.chunks[self.chunk_of[i]]["metadata"]
        return str(meta.get("section", "?")), meta.get("page_num")


def _body_chunks(chunks: list) -> list[dict]:
    """Prose body only — no tables, no reference lists (see _SKIP_SECTIONS)."""
    def named(c) -> str:
        return str(c["metadata"].get("section", "")).lower()

    return [
        c for c in chunks
        if c["metadata"].get("section_type") != "table"
        and not any(s in named(c) for s in _SKIP_SECTIONS)
    ]


def _load_doc(paper_id: str, title: str, limit: int) -> _Doc | None:
    """Load one paper out of Chroma as a _Doc. None when it has no usable body
    (deleted collection, still ingesting, tables only) — never raises, because
    one bad corpus paper must not sink the whole check."""
    from ingestion.retriever import get_all_chunks  # lazy — see module header

    try:
        chunks = get_all_chunks(paper_id)
    except Exception as exc:
        print(f"[plagiarism_auditor] Could not read {paper_id}: {exc}")
        return None

    body = _body_chunks(chunks or [])
    if not body:
        return None

    doc = _Doc(paper_id, title)
    for c in body:
        if not doc.add_chunk(c, limit):
            break
    return doc if len(doc.words) >= SHINGLE_W else None


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — shingle index over the corpus
# ───────────────────────────────────────────────────────────────────────────

def _shingles(words: list[str]):
    """Yield (position, shingle) for every word n-gram in the stream."""
    for i in range(len(words) - SHINGLE_W + 1):
        yield i, " ".join(words[i:i + SHINGLE_W])


def build_index(sources: list[_Doc]) -> dict[str, list[tuple[int, int]]]:
    """Map each corpus shingle to the (source index, position) pairs holding it.

    Postings are capped at MAX_POSTINGS: a shingle appearing more often than
    that is boilerplate, and every extra posting only costs memory and match
    time without changing the outcome (the common-phrase guard drops it anyway).
    """
    index: dict[str, list[tuple[int, int]]] = {}
    for src_i, doc in enumerate(sources):
        for pos, sh in _shingles(doc.words):
            postings = index.setdefault(sh, [])
            if len(postings) < MAX_POSTINGS:
                postings.append((src_i, pos))
    return index


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — greedy longest-run matching
# ───────────────────────────────────────────────────────────────────────────

def _extend(draft: _Doc, src: _Doc, d_pos: int, s_pos: int) -> int:
    """Length in words of the identical run starting at these two positions."""
    n = SHINGLE_W
    dw, sw = draft.words, src.words
    while (d_pos + n < len(dw) and s_pos + n < len(sw)
           and dw[d_pos + n] == sw[s_pos + n]):
        n += 1
    return n


def _content_words(words: list[str]) -> int:
    """Distinct non-stopword, non-numeric words — how much a match actually says.

    "of the results reported in the previous section we observe that the" is
    twelve identical words and no evidence of anything. Requiring real content
    words is what separates reuse from two people writing English.
    """
    return len({w for w in words if w not in _STOPWORDS and not w.isdigit()})


def find_matches(draft: _Doc, sources: list[_Doc],
                 index: dict[str, list[tuple[int, int]]]) -> list[dict]:
    """Scan the draft against the index, returning raw (undowngraded) matches.

    Greedy and non-overlapping: at each hit the longest available run wins and
    the cursor jumps past it. Reporting every sub-run of one long match would
    bury the author in duplicates of the same finding.
    """
    matches: list[dict] = []
    i, limit = 0, len(draft.words) - SHINGLE_W + 1

    while i < limit:
        sh = " ".join(draft.words[i:i + SHINGLE_W])
        postings = index.get(sh)
        if not postings:
            i += 1
            continue

        # Guard — a string sitting in several different papers is how the field
        # writes, not something anyone copied.
        if len({src_i for src_i, _ in postings}) >= COMMON_SOURCE_LIMIT:
            i += 1
            continue

        best_len, best = 0, None
        for src_i, s_pos in postings:
            n = _extend(draft, sources[src_i], i, s_pos)
            if n > best_len:
                best_len, best = n, (src_i, s_pos)

        if best_len < MIN_MATCH_WORDS or best is None:
            i += 1
            continue

        src_i, s_pos = best
        if _content_words(draft.words[i:i + best_len]) < MIN_CONTENT_WORDS:
            i += 1
            continue

        matches.append({
            "source_i":    src_i,
            "draft_start": i,
            "draft_end":   i + best_len,
            "src_start":   s_pos,
            "src_end":     s_pos + best_len,
            "words":       best_len,
        })
        i += best_len

    return matches


def _classify(draft: _Doc, sources: list[_Doc], raw: dict) -> dict:
    """Turn a raw match into a report entry, applying the attribution downgrade."""
    src = sources[raw["source_i"]]
    d_i, d_j = raw["draft_start"], raw["draft_end"]

    around, matched = draft.context(d_i, d_j)
    # Credit counts only when the author gave it: a marker they wrote next to the
    # reused passage, or quotation marks wrapping it. Markers carried over inside
    # the copied text are the source's own citations and prove nothing — without
    # this split, any reused related-work paragraph (which is always dense with
    # citations) would be waved through.
    attributed = _has_citation(around) or bool(_QUOTE_RE.search(around + matched))

    if attributed:
        verdict, severity = "ATTRIBUTED", "low"
    elif raw["words"] >= LONG_MATCH_WORDS:
        verdict, severity = "VERBATIM", "high"
    else:
        verdict, severity = "NEAR_VERBATIM", "med"

    section, page = draft.locate(d_i)
    src_section, src_page = src.locate(raw["src_start"])

    return {
        "verdict":   verdict,
        "severity":  severity,
        "words":     raw["words"],
        "excerpt":   draft.excerpt(d_i, d_j)[:_EXCERPT_CHARS],
        "section":   section,
        "page":      page,
        "source": {
            "paper_id": src.paper_id,
            "title":    src.title,
            "section":  src_section,
            "page":     src_page,
            "excerpt":  src.excerpt(raw["src_start"], raw["src_end"])[:_EXCERPT_CHARS],
        },
        "note": "",
        "evidence": {
            "section":      section,
            "page":         page,
            "section_type": draft.chunks[draft.chunk_of[d_i]]["metadata"].get("section_type", "text"),
        },
    }


def _shingle_set(words: list[str]) -> set[str]:
    return {" ".join(words[i:i + SHINGLE_W]) for i in range(len(words) - SHINGLE_W + 1)}


def split_same_documents(draft: _Doc, sources: list[_Doc]
                         ) -> tuple[list[_Doc], list[dict]]:
    """Partition the corpus into papers worth comparing against, and papers that
    ARE the draft.

    A user who uploads their draft and also has the published version (or an
    earlier submission) in their library would otherwise be told their own paper
    is 90% plagiarised — technically true, useless, and alarming.

    This runs BEFORE indexing, and measures each source independently as the
    fraction of the draft's shingles it contains. Both properties are load-
    bearing, and the first browser run proved it by holding three copies of the
    same paper:

      - *Independently*, because the matcher is greedy: each draft span is
        assigned to one best source, so with duplicates in the corpus the first
        copy absorbs all the coverage and its twins score ~0. Measured after
        matching, only one of three identical copies was caught.
      - *Before indexing*, because three identical copies also trip the
        COMMON_SOURCE_LIMIT boilerplate guard — every real match would be
        silently discarded as "standard field phrasing".
    """
    draft_shingles = _shingle_set(draft.words)
    if not draft_shingles:
        return sources, []

    comparable: list[_Doc] = []
    excluded: list[dict] = []
    for src in sources:
        shared = len(draft_shingles & _shingle_set(src.words)) / len(draft_shingles)
        if shared >= SAME_DOC_RATIO:
            excluded.append({
                "paper_id": src.paper_id,
                "title":    src.title,
                "reason":   "Looks like the same document as this draft — excluded "
                            "so it doesn't swamp the report.",
            })
        else:
            comparable.append(src)
    return comparable, excluded


# ───────────────────────────────────────────────────────────────────────────
# Step 4 — paraphrase pass (embeddings, then an LLM gate)
# ───────────────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _probe_passages(draft: _Doc, covered: set[int]) -> list[dict]:
    """Substantial draft chunks that verbatim matching did NOT already explain.

    Re-probing text we've already flagged would just restate the same finding in
    a weaker form, so covered chunks are skipped. Longest first: a long passage
    carries enough signal for cosine to mean something.
    """
    seen: set[int] = set()
    out: list[dict] = []
    for pos, c_idx in enumerate(draft.chunk_of):
        if c_idx in seen or c_idx in covered:
            continue
        seen.add(c_idx)
        chunk = draft.chunks[c_idx]
        text  = re.sub(r"\s+", " ", chunk.get("text", "") or "").strip()
        if len(text) < MIN_PROBE_CHARS:
            continue
        meta = chunk["metadata"]
        out.append({
            "text":    text,
            "section": str(meta.get("section", "?")),
            "page":    meta.get("page_num"),
            "section_type": meta.get("section_type", "text"),
            "word_pos": pos,
        })
    out.sort(key=lambda p: len(p["text"]), reverse=True)
    return out[:MAX_PROBES]


def _nearest_in_source(vec, src: _Doc) -> tuple[float, dict | None]:
    """Closest passage to `vec` in one corpus paper, by true cosine.

    Chroma's stored distance depends on the collection's configured space and
    BGE vectors are not L2-normalised, so the raw distance is not a similarity.
    The candidate's own embedding is pulled back and the cosine computed here —
    unambiguous, and independent of how the collection was created.
    """
    from ingestion.retriever import client as chroma, collection_name  # lazy

    try:
        res = chroma.get_collection(name=collection_name(src.paper_id)).query(
            query_embeddings=[vec.tolist()],
            n_results=1,
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception as exc:
        print(f"[plagiarism_auditor] Probe against {src.paper_id} failed: {exc}")
        return 0.0, None

    docs = (res.get("documents") or [[]])[0]
    embs = (res.get("embeddings") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    if not docs or embs is None or len(embs) == 0:
        return 0.0, None

    meta = metas[0] or {}
    return _cosine(vec, np.asarray(embs[0])), {
        "text":    re.sub(r"\s+", " ", docs[0] or "").strip(),
        "section": str(meta.get("section", "?")),
        "page":    meta.get("page_num"),
    }


_PARAPHRASE_SYSTEM_PROMPT = """You compare two passages of academic writing and decide whether one is a
rewording of the other.

You are given numbered CASES. Each has a DRAFT passage and a SOURCE passage from
a different paper. They were paired because they are semantically close — that
alone means nothing, because papers on the same topic are always close.

For each case return:
  SAME_CONTENT — the draft passage states the same specific things as the source
                 in different words: the same argument with the same structure,
                 the same definition, the same described procedure, the same
                 findings in the same order. A reviewer reading both would say
                 one was rewritten from the other.
  SAME_TOPIC   — they are simply about the same subject, share terminology, or
                 make similar standard statements a field makes constantly. This
                 includes two independent descriptions of the same well-known
                 method, dataset, or background fact.

Be strict. Accusing an author of reusing text they wrote themselves is the worst
thing this tool can do. Shared technical vocabulary, shared background, shared
benchmark names, and standard phrasing are ALL SAME_TOPIC. Return SAME_CONTENT
only when the correspondence is specific and sustained, not topical. When in any
doubt at all, return SAME_TOPIC.

"note" must, for SAME_CONTENT, name the specific correspondence in one sentence
(what is being said the same way in both). Never generic.

Return ONLY a JSON array, one object per case, in the SAME ORDER:

[
  {"index": 1, "relation": "SAME_CONTENT | SAME_TOPIC", "note": "one short sentence"}
]
"""


def _confirm_paraphrases(pairs: list[tuple[dict, dict, _Doc, float]]) -> list[dict]:
    """One batched LLM call gating the embedding candidates. Raw relations, in
    pair order; an empty list on any failure (which shows nothing — the safe
    direction)."""
    if not pairs:
        return []
    blocks = []
    for i, (probe, cand, src, _sim) in enumerate(pairs, 1):
        blocks.append(
            f"CASE {i}:\nDRAFT PASSAGE:\n\"{probe['text'][:_EXCERPT_CHARS]}\"\n\n"
            f"SOURCE PASSAGE (from \"{src.title}\"):\n\"{cand['text'][:_EXCERPT_CHARS]}\""
        )
    user_prompt = (
        "\n\n----------------------------------------\n\n".join(blocks)
        + "\n\nJudge every case above. Return the JSON array (one object per case, in order)."
    )
    raw = chat_completion(
        messages=[
            {"role": "system", "content": _PARAPHRASE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=900,
        temperature=0.0,
    )
    rels = parse_llm_json(raw, expect="array")
    return rels if isinstance(rels, list) else []


def find_paraphrases(draft: _Doc, sources: list[_Doc], covered: set[int]) -> list[dict]:
    """Bounded embedding pass over the passages verbatim matching didn't explain.

    Budget: MAX_PROBES embeddings + MAX_PROBES x len(sources) local Chroma
    queries + exactly one LLM call. Only pairs the LLM calls SAME_CONTENT are
    returned, always at low severity — this tier is a prompt to look, never a
    finding.
    """
    probes = _probe_passages(draft, covered)
    if not probes or not sources:
        return []

    from ingestion.models import embed_passages  # lazy — see module header

    try:
        # Passage-side embedding on both sides: the stored chunk vectors were
        # made with embed_passages, and BGE is asymmetric — using the query
        # prefix here would compare vectors from two different spaces.
        vectors = embed_passages([p["text"][:PROBE_CHARS] for p in probes])
    except Exception as exc:
        print(f"[plagiarism_auditor] Probe embedding failed: {exc}")
        return []

    # A collection whose vector index is missing or corrupt on disk fails for
    # every probe, not just the first. Drop it after one failure so the log
    # carries one line per broken paper rather than one per probe per paper.
    unusable: set[str] = set()

    pairs: list[tuple[dict, dict, _Doc, float]] = []
    for probe, vec in zip(probes, vectors):
        best_sim, best_cand, best_src = 0.0, None, None
        for src in sources:
            if src.paper_id in unusable:
                continue
            sim, cand = _nearest_in_source(np.asarray(vec), src)
            if cand is None:
                unusable.add(src.paper_id)
                continue
            if sim > best_sim:
                best_sim, best_cand, best_src = sim, cand, src
        if best_cand and best_sim >= PARAPHRASE_SIM:
            pairs.append((probe, best_cand, best_src, best_sim))

    try:
        relations = _confirm_paraphrases(pairs)
    except Exception as exc:
        print(f"[plagiarism_auditor] Paraphrase confirmation failed: {exc}")
        return []

    out = []
    for i, (probe, cand, src, sim) in enumerate(pairs):
        rel = relations[i] if i < len(relations) else None
        if str((rel or {}).get("relation", "")).upper().strip() != "SAME_CONTENT":
            continue  # conservative: unconfirmed similarity is never shown
        out.append({
            "verdict":   "PARAPHRASE",
            "severity":  "low",
            "words":     0,          # no verbatim run to count
            "similarity": round(sim, 3),
            "excerpt":   probe["text"][:_EXCERPT_CHARS],
            "section":   probe["section"],
            "page":      probe["page"],
            "source": {
                "paper_id": src.paper_id,
                "title":    src.title,
                "section":  cand["section"],
                "page":     cand["page"],
                "excerpt":  cand["text"][:_EXCERPT_CHARS],
            },
            "note": str((rel or {}).get("note", "") or "").strip(),
            "evidence": {
                "section":      probe["section"],
                "page":         probe["page"],
                "section_type": probe["section_type"],
            },
        })
    return out


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
        "corpus_size":       0,
        "sources_matched":   0,
        "words_scanned":     0,
        "overlap_words":     0,
        "overlap_score":     0.0,
        "originality_score": None,
        "matches_found":     0,
        "attributed_count":  0,
        "paraphrase_count":  0,
        "matches":           [],
        "excluded_sources":  [],
        "audit_failed":      failed,
        "reason":            reason,
        "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_VERDICT_RANK = {"VERBATIM": 0, "NEAR_VERBATIM": 1, "PARAPHRASE": 2, "ATTRIBUTED": 3}


def audit_overlap(paper_id: str, corpus: list[dict], on_progress=None) -> dict:
    """Run the full overlap check for one draft and return a report dict.

    Never raises — returns an `audit_failed` report on any hard failure so the
    endpoint/frontend always get a consistent shape.
    """
    try:
        _emit(on_progress, "reading", "Reading the draft…")
        draft = _load_doc(paper_id, "this draft", MAX_DRAFT_TOKENS)
        if draft is None:
            return _empty_report(paper_id, failed=True,
                                 reason="No readable body text found for this draft.")

        # The draft can legitimately appear in its own corpus list (it lives in
        # the same library); comparing it to itself is meaningless.
        others = [c for c in (corpus or []) if c.get("paper_id") != paper_id][:MAX_CORPUS_PAPERS]
        if not others:
            report = _empty_report(paper_id, failed=False, reason=(
                "There are no other papers in your library to compare against. "
                "Upload the papers you're drawing on, then re-run this check."))
            report["words_scanned"] = len(draft.words)
            return report

        _emit(on_progress, "indexing", f"Indexing {len(others)} paper(s) from your library…")
        sources = [
            doc for doc in (
                _load_doc(c["paper_id"], c.get("title") or "Untitled", MAX_CORPUS_TOKENS)
                for c in others
            ) if doc is not None
        ]
        if not sources:
            report = _empty_report(paper_id, failed=False, reason=(
                "None of the other papers in your library are indexed yet, so "
                "there was nothing to compare against."))
            report["words_scanned"] = len(draft.words)
            return report

        # Copies of the draft itself come out of the corpus before anything else
        # touches them — indexing, matching and the paraphrase pass all work on
        # `comparable` only. See split_same_documents for why this must happen
        # here and not after matching.
        comparable, excluded = split_same_documents(draft, sources)
        if not comparable:
            report = _empty_report(paper_id, failed=False, reason=(
                "The only other copies of this paper in your library are the same "
                "document as this draft, so there was nothing to compare against."))
            report["words_scanned"]    = len(draft.words)
            report["excluded_sources"] = excluded
            return report

        index = build_index(comparable)

        _emit(on_progress, "matching", "Comparing passages against your library…")
        raw_matches = find_matches(draft, comparable, index)
        entries = [_classify(draft, comparable, m) for m in raw_matches]

        # Chunks the verbatim pass already explained — the paraphrase pass skips
        # them, so one passage is never reported twice under two labels.
        covered = {draft.chunk_of[m["draft_start"]] for m in raw_matches}

        # Overlap counts only UNATTRIBUTED verbatim runs. Quoted and cited reuse
        # is correctly-done scholarship — counting it against the author would
        # punish exactly the behaviour we want.
        flagged = [e for e in entries if e["verdict"] != "ATTRIBUTED"]
        overlap_words = sum(e["words"] for e in flagged)
        scanned = max(len(draft.words), 1)
        overlap_ratio = min(overlap_words / scanned, 1.0)

        _emit(on_progress, "paraphrase", "Checking for reworded passages…")
        paraphrases = find_paraphrases(draft, comparable, covered)

        all_matches = entries + paraphrases
        all_matches.sort(key=lambda m: (
            _VERDICT_RANK.get(m["verdict"], 4),
            -m.get("words", 0),
        ))
        all_matches = all_matches[:MAX_MATCHES]

        matched_titles = {m["source"]["title"] for m in all_matches
                          if m["verdict"] != "ATTRIBUTED"}

        return {
            "paper_id":          paper_id,
            # What was actually compared — excluded copies of the draft are not
            # "papers compared", and counting them would overstate the check.
            "corpus_size":       len(comparable),
            "sources_matched":   len(matched_titles),
            "words_scanned":     len(draft.words),
            "overlap_words":     overlap_words,
            "overlap_score":     round(overlap_ratio, 4),
            "originality_score": round(1.0 - overlap_ratio, 4),
            "matches_found":     len(flagged),
            "attributed_count":  len(entries) - len(flagged),
            "paraphrase_count":  len(paraphrases),
            "matches":           all_matches,
            "excluded_sources":  excluded,
            "audit_failed":      False,
            "reason":            "",
            "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    except Exception as exc:
        print(f"[plagiarism_auditor] Overlap check failed: {exc}")
        return _empty_report(paper_id, failed=True, reason=str(exc))

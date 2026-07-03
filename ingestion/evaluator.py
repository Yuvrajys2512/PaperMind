"""
ingestion/evaluator.py

Evaluates LLM-generated answers for faithfulness and answer relevancy
using local embedding-similarity scoring (all-MiniLM-L6-v2).

No API calls required.

Public API
----------
evaluate_answer(query, answer, chunks) → {faithfulness, answer_relevancy, method}
compute_confidence(faithfulness, answer_relevancy) → float (0-100)
"""

import re
import numpy as np
from ingestion.models import embed_query, embed_passages

_SUPPORT_THRESHOLD = 0.55


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _extract_chunk_texts(chunks: list) -> list[str]:
    texts = []
    for c in chunks:
        if isinstance(c, dict):
            texts.append(c.get("text", ""))
        else:
            texts.append(str(c))
    return [t for t in texts if t.strip()]


def _split_into_sentences(text: str) -> list[str]:
    raw = text.replace("\n", ". ").split(". ")
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _score_faithfulness(answer: str, chunk_texts: list[str]) -> float:
    sentences = _split_into_sentences(answer)
    if not sentences:
        return 0.0

    # Both sides are passage-like (chunks vs answer sentences), so no
    # query prefix on either — that's how BGE was trained for
    # symmetric similarity.
    chunk_embs    = embed_passages(chunk_texts)
    sentence_embs = embed_passages(sentences)

    supported = sum(
        1 for sent_emb in sentence_embs
        if max(_cosine_similarity(sent_emb, ce) for ce in chunk_embs) >= _SUPPORT_THRESHOLD
    )
    return supported / len(sentences)


def _score_answer_relevancy(query: str, answer: str) -> float:
    # Asymmetric: the query gets the BGE instruction prefix, the answer
    # (a passage) does not.
    query_emb  = embed_query(query)
    answer_emb = embed_passages([answer])[0]
    return _cosine_similarity(query_emb, answer_emb)


def evaluate_answer(query: str, answer: str, chunks: list) -> dict:
    chunk_texts = _extract_chunk_texts(chunks)

    if not chunk_texts:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "method": "error_no_chunks"}

    if not answer or not answer.strip():
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "method": "error_no_answer"}

    return {
        "faithfulness":     _score_faithfulness(answer, chunk_texts),
        "answer_relevancy": _score_answer_relevancy(query, answer),
        "method":           "local",
    }


def compute_confidence(faithfulness_score: float, answer_relevancy_score: float) -> float:
    """(faithfulness × 0.7 + answer_relevancy × 0.3) × 100"""
    return round(((faithfulness_score * 0.7) + (answer_relevancy_score * 0.3)) * 100, 2)


# Above this query↔chunk cosine, a passage is "strongly relevant". BGE
# query-passage similarities for genuinely on-topic academic text sit ~0.5-0.75;
# unrelated text floors out around ~0.25-0.35. 0.5 is the practical boundary.
_STRONG_CHUNK_SIM = 0.5


def compute_retrieval_quality(query: str, chunks: list) -> dict | None:
    """
    Score how relevant the *retrieved context* is to the query — the third leg
    of the RAG triad that faithfulness/relevancy don't capture.

    Faithfulness asks "is the answer grounded in the chunks?" but says nothing
    about whether those chunks were any good. A high-faithfulness answer built
    on weak evidence is the failure mode this metric exposes.

    Why NOT the reranker score
    --------------------------
    The CrossEncoder reranker (ms-marco-MiniLM) stamps a `rerank_score` on each
    chunk, but it is a *ranking* logit, not a calibrated relevance probability.
    It's brutally bimodal: a near-verbatim answer span scores ~+10, while ANY
    other passage — even genuinely on-topic supporting context — scores ~-11.
    For summarization-style answers (synthesized across several context chunks
    with no single answer span) every chunk scores ~0, so a sigmoid of those
    logits reports "0% retrieval" on a perfectly grounded answer. It conflates
    "broad query" with "bad retrieval".

    What we use instead
    -------------------
    The BGE bi-encoder cosine similarity between the query and each chunk — the
    same calibrated embedding space `answer_relevancy` already lives in. It's
    smooth, defined for every chunk, and doesn't punish synthesis.

    Returns None when there are no usable chunk texts (e.g. error paths or older
    persisted answers) so the UI can hide the metric rather than fabricate one.

    Returns
    -------
    dict | None:
        score          : float  Mean query↔chunk similarity × 100 (headline).
        top_relevance  : float  Best single chunk's similarity (0-1).
        strong_chunks  : int    Chunks with similarity ≥ _STRONG_CHUNK_SIM.
        chunks_used    : int    Total chunks the answer was built from.
    """
    chunk_texts = _extract_chunk_texts(chunks)
    if not chunk_texts:
        return None

    query_emb  = embed_query(query)
    chunk_embs = embed_passages(chunk_texts)

    sims         = [_cosine_similarity(query_emb, ce) for ce in chunk_embs]
    mean_sim     = float(np.mean(sims))
    top_sim      = float(np.max(sims))
    strong_count = sum(1 for s in sims if s >= _STRONG_CHUNK_SIM)

    return {
        "score":         round(max(0.0, mean_sim) * 100, 1),
        "top_relevance": round(max(0.0, top_sim), 2),
        "strong_chunks": strong_count,
        "chunks_used":   len(sims),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Numeric grounding
# ─────────────────────────────────────────────────────────────────────────────
# A number token: optional thousands separators + optional decimals. The
# negative-lookbehind keeps us from starting mid-number (e.g. the "4" in "v4").
_NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")

# Structural cross-references like "Section 7", "Figure 3", "Table 2". These are
# navigation, not claims about the paper's findings, so their numbers must not be
# treated as ungrounded just because the referenced label doesn't recur verbatim
# in the retrieved chunks. Matches an optional abbreviating dot ("Fig. 3").
_REF_RE = re.compile(
    r"\b(?:section|sec|subsection|figure|fig|table|tab|equation|eq|eqn|"
    r"appendix|app|chapter|ch|step|algorithm|alg|theorem|lemma|line)\.?\s*\d+",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[tuple[float, int, str]]:
    """
    Pull numeric tokens out of free text as (value, decimal_places, display).

    Strips three sources of *system / navigational* numbers first so they don't
    get mistaken for unsupported claims:
      - citation scaffolding like "[Section: 4, Page: 9]"
      - URLs (e.g. .../tensor2tensor)
      - inline cross-references like "Section 7", "Figure 3", "Table 2"
    """
    text = re.sub(r"\[[^\]]*\]", " ", text)        # citation brackets
    text = re.sub(r"https?://\S+", " ", text)      # URLs
    text = re.sub(r"www\.\S+", " ", text)
    text = _REF_RE.sub(" ", text)                  # inline cross-references

    out: list[tuple[float, int, str]] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((val, decimals, m.group(1)))
    return out


def compute_numeric_grounding(answer: str, chunks: list) -> dict | None:
    """
    Check that the numbers in the answer actually appear in the source chunks.

    Numbers (BLEU scores, p-values, percentages, dataset sizes) are exactly
    where hallucination does the most damage in a paper, and exactly where
    embedding-based faithfulness is weakest — a 0.92 cosine won't notice "28.4"
    silently becoming "82.4". This is a targeted check where the other metrics
    are blind.

    Each answer number is grounded if some chunk contains a number that matches
    when both are rounded to the answer number's precision (so "28.4" matches
    "28.43" and "28.40", while years/integers must match exactly — no loose
    tolerance that would let 2014 match 2020).

    Returns None when the answer contains no numbers, so the UI hides the metric
    rather than showing a vacuous 100%.

    Returns
    -------
    dict | None:
        score      : float  Grounded ÷ total × 100.
        grounded   : int    Distinct numbers found in the source.
        total      : int    Distinct numbers in the answer.
        ungrounded : list   Up to 8 answer numbers not located in any chunk.
    """
    answer_nums = _extract_numbers(answer or "")
    if not answer_nums:
        return None

    # Deduplicate by rounded value so "28.4" mentioned twice counts once, and
    # "28.4" vs "28.40" are treated as the same claim.
    seen: set = set()
    unique: list[tuple[float, int, str]] = []
    for val, decimals, disp in answer_nums:
        key = round(val, decimals)
        if key in seen:
            continue
        seen.add(key)
        unique.append((val, decimals, disp))

    chunk_text  = " ".join(_extract_chunk_texts(chunks))
    chunk_nums  = [v for v, _, _ in _extract_numbers(chunk_text)]

    grounded_count = 0
    ungrounded: list[str] = []
    for val, decimals, disp in unique:
        target = round(val, decimals)
        hit = any(round(cv, decimals) == target for cv in chunk_nums)
        if hit:
            grounded_count += 1
        else:
            ungrounded.append(disp)

    total = len(unique)
    return {
        "score":      round(grounded_count / total * 100, 1),
        "grounded":   grounded_count,
        "total":      total,
        "ungrounded": ungrounded[:8],
    }

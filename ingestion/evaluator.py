"""
ingestion/evaluator.py — Phase 3, Step 1

Evaluates LLM-generated answers for faithfulness and answer relevancy
using local embedding-similarity scoring (all-MiniLM-L6-v2).

No OpenAI key required. No RAGAS dependency.

Public API
----------
evaluate_answer(query, answer, chunks) → {faithfulness, answer_relevancy, method}
compute_confidence(faithfulness, answer_relevancy) → float (0-100)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


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


# A sentence is "supported" by a chunk if cosine similarity >= this threshold.
# Tuned for all-MiniLM-L6-v2 on academic text.
_SUPPORT_THRESHOLD = 0.55


def _score_faithfulness(answer: str, chunk_texts: list[str]) -> float:
    """
    Proportion of answer sentences that are semantically supported
    by at least one retrieved chunk. Range: 0.0 – 1.0.
    Typical good answer: 0.55 – 0.85 with this model.
    """
    sentences = _split_into_sentences(answer)
    if not sentences:
        return 0.0

    embedder      = _get_embedder()
    chunk_embs    = embedder.encode(chunk_texts)
    sentence_embs = embedder.encode(sentences)

    supported = 0
    for sent_emb in sentence_embs:
        sims = [_cosine_similarity(sent_emb, ce) for ce in chunk_embs]
        if max(sims) >= _SUPPORT_THRESHOLD:
            supported += 1

    return supported / len(sentences)


def _score_answer_relevancy(query: str, answer: str) -> float:
    """
    Cosine similarity between query and answer embeddings.
    Higher = more on-topic. Range: 0.0 – 1.0.
    Typical good answer: 0.40 – 0.75.
    """
    embedder   = _get_embedder()
    query_emb  = embedder.encode([query])[0]
    answer_emb = embedder.encode([answer])[0]
    return _cosine_similarity(query_emb, answer_emb)


def evaluate_answer(query: str, answer: str, chunks: list) -> dict:
    """
    Evaluate a generated answer for faithfulness and answer relevancy.
    No API calls. No external keys required.

    Parameters
    ----------
    query  : original user question
    answer : LLM-generated answer
    chunks : list of chunk dicts (with 'text' key) or plain strings

    Returns
    -------
    { faithfulness: float, answer_relevancy: float, method: "local" }
    """
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
    """
    (faithfulness × 0.7 + answer_relevancy × 0.3) × 100
    Returns a score between 0 and 100.
    """
    raw = (faithfulness_score * 0.7) + (answer_relevancy_score * 0.3)
    return round(raw * 100, 2)
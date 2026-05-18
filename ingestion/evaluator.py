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

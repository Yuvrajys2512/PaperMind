from sentence_transformers import CrossEncoder

_model = None

def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, chunks: list, top_k: int = 5):
    """
    Takes query and list of chunks (from RRF output),
    scores each (query, chunk_text) pair with CrossEncoder,
    returns top_k re-ranked results.
    """
    model = get_reranker()

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]
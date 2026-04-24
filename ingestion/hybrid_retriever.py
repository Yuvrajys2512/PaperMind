from ingestion.bm25_retriever import bm25_retrieve
from ingestion.retriever import retrieve  # your existing vector retriever

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    Merges BM25 and vector search results using RRF.
    Each chunk is identified by its text (used as unique key).
    Final score = sum of 1/(rank + k) across both lists.
    """
    scores = {}
    chunk_data = {}

    for rank, result in enumerate(bm25_results):
        key = result["text"]
        scores[key] = scores.get(key, 0) + 1 / (rank + 1 + k)
        chunk_data[key] = result

    for rank, result in enumerate(vector_results):
        key = result["text"]
        scores[key] = scores.get(key, 0) + 1 / (rank + 1 + k)
        if key not in chunk_data:
            chunk_data[key] = result

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for key in sorted_keys:
        entry = chunk_data[key].copy()
        entry["rrf_score"] = scores[key]
        merged.append(entry)

    return merged


def hybrid_retrieve(query: str, paper_name: str, top_k: int = 5):
    """
    Runs BM25 + vector search in parallel, merges with RRF,
    returns top_k results.
    """
    bm25_results = bm25_retrieve(query, paper_name, top_k=10)
    vector_results = retrieve(query, paper_name, top_k=10)

    merged = reciprocal_rank_fusion(bm25_results, vector_results)

    return merged[:top_k]
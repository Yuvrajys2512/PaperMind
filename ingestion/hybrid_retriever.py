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


def hybrid_retrieve(
    query: str,
    paper_name: str,
    top_k: int = 5,
    boost_terms: list[str] = None,
    hyde_text: str | None = None,
):
    """
    Runs BM25 + vector search in parallel, merges with RRF, returns top_k.

    Parameters
    ----------
    query       : User's question — used by BM25 (lexical signal).
    paper_name  : Paper identifier.
    top_k       : Number of final results to return after RRF.
    boost_terms : Extra terms appended to the BM25 query (key_concepts).
    hyde_text   : Optional HyDE pseudo-passage to use for *dense* retrieval
                  in place of the raw query. BM25 always uses the original
                  query so lexical hits aren't replaced by hallucinated
                  terminology.
    """
    if boost_terms:
        boosted_query = query + " " + " ".join(boost_terms)
    else:
        boosted_query = query

    bm25_results   = bm25_retrieve(boosted_query, paper_name, top_k=10)
    vector_query   = hyde_text if hyde_text else query
    vector_results = retrieve(vector_query, paper_name, top_k=10)

    merged = reciprocal_rank_fusion(bm25_results, vector_results)

    return merged[:top_k]
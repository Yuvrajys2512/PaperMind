"""
ingestion/compare_retriever.py  — Session 7

Retrieves chunks from two papers simultaneously for comparison queries.
Tags each chunk with its source paper label (A or B).

Public API
----------
compare_retrieve(query, paper_id_a, paper_id_b, top_k) -> list[dict]
"""

from ingestion.hybrid_retriever import hybrid_retrieve


def compare_retrieve(
    query: str,
    paper_id_a: str,
    paper_id_b: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top_k chunks from each paper and interleave them so both papers
    are always represented in the context sent to the generator.

    Each chunk gets two extra keys:
      paper_id    : the source paper's ID
      paper_label : "A" or "B"
    """
    chunks_a = hybrid_retrieve(query, paper_id_a, top_k=top_k)
    chunks_b = hybrid_retrieve(query, paper_id_b, top_k=top_k)

    for c in chunks_a:
        c["paper_id"]    = paper_id_a
        c["paper_label"] = "A"

    for c in chunks_b:
        c["paper_id"]    = paper_id_b
        c["paper_label"] = "B"

    # Interleave A1, B1, A2, B2, ... so both voices appear in every window
    merged = []
    for a, b in zip(chunks_a, chunks_b):
        merged.append(a)
        merged.append(b)
    longer = chunks_a if len(chunks_a) > len(chunks_b) else chunks_b
    merged.extend(longer[min(len(chunks_a), len(chunks_b)):])

    return merged

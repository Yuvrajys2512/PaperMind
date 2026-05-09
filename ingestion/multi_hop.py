"""
ingestion/multi_hop.py

Decomposes a complex query into 2-3 focused sub-questions,
then runs retrieval for each and merges the results.

This ensures the retriever pulls chunks from multiple relevant
sections of the paper rather than clustering around one.

Upgrade 1 change
----------------
multi_hop_retrieve() now accepts an optional sub_questions parameter.
When the query_router passes sub_questions from the Query Plan,
decompose_query() is skipped entirely — saving one LLM call per
multi-hop request.  If sub_questions is None (e.g. direct callers,
tests), the old decomposition behaviour is preserved as a fallback.
"""

import json
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.llm_client import chat_completion

DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition assistant for a research paper Q&A system.

Your job: break a complex question into 2-3 focused sub-questions that together cover everything needed to answer the original question.

Rules:
- Each sub-question must target a DIFFERENT aspect or concept
- Sub-questions must be specific enough to retrieve precise passages
- Use technical terms from the domain — don't paraphrase into vague language
- Return ONLY a JSON array of strings, nothing else
- Maximum 3 sub-questions. Minimum 2.

Example:
Question: "Why can the Transformer be trained faster than RNN-based models?"
Output: ["How do RNNs process sequences sequentially and what limits their parallelization?", "How does self-attention connect positions with constant operations?", "What training times and hardware did the Transformer use?"]"""


def decompose_query(query: str) -> list[str]:
    """
    Calls the LLM to break a complex query into 2-3 sub-questions.
    Falls back to [query] if decomposition fails for any reason.

    Note: this is only called when sub_questions are NOT supplied by
    the query_router (i.e. legacy callers or direct test calls).
    """
    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}"}
            ],
            max_tokens=200,
            temperature=0.1,
        )

        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        sub_questions = json.loads(raw)

        # Validate — must be a list of strings
        if (
            isinstance(sub_questions, list)
            and len(sub_questions) >= 2
            and all(isinstance(q, str) for q in sub_questions)
        ):
            print(f"[multi_hop] Decomposed into {len(sub_questions)} sub-questions:")
            for i, q in enumerate(sub_questions, 1):
                print(f"  {i}. {q}")
            return sub_questions

        print("[multi_hop] Decomposition returned unexpected structure, falling back.")
        return [query]

    except Exception as e:
        print(f"[multi_hop] Decomposition failed ({e}), falling back to original query.")
        return [query]


def multi_hop_retrieve(
    query: str,
    paper_name: str,
    retrieval_k: int,
    sub_questions: list[str] | None = None,   # ← Upgrade 1: supplied by query_router
    boost_terms:   list[str] | None = None,
) -> list:
    """
    Retrieves chunks for each sub-question and merges + deduplicates results.

    Parameters
    ----------
    query         : str            Original user question (always used as one retrieval pass).
    paper_name    : str            Paper ID scoping the vector-store lookup.
    retrieval_k   : int            How many chunks to retrieve per sub-question.
    sub_questions : list[str] | None
        Pre-computed sub-questions from the Query Plan (query_router passes these).
        If None, decompose_query() is called to generate them — preserving the
        original behaviour for any direct callers or existing tests.
    boost_terms   : list[str] | None
        Key concepts to boost during retrieval.

    Returns
    -------
    list  Merged, deduplicated chunks ready for reranking.
    """
    if sub_questions is not None:
        # ── Fast path: use plan's sub-questions, skip LLM decomposition ──
        print(f"[multi_hop] Using {len(sub_questions)} sub-questions from Query Plan.")
        for i, q in enumerate(sub_questions, 1):
            print(f"  {i}. {q}")
    else:
        # ── Fallback: derive sub-questions via LLM (legacy behaviour) ────
        print("[multi_hop] No sub_questions supplied — running decompose_query().")
        sub_questions = decompose_query(query)

    # Always include the original query as a retrieval pass
    all_queries = [query] + [q for q in sub_questions if q != query]

    seen_ids      = set()
    merged_chunks = []

    for q in all_queries:
        results = hybrid_retrieve(q, paper_name, top_k=retrieval_k, boost_terms=boost_terms)
        for chunk in results:
            chunk_id = chunk["metadata"].get("chunk_id")
            if chunk_id is None:
                chunk_id = hash(chunk["text"])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_chunks.append(chunk)

    print(
        f"[multi_hop] {len(merged_chunks)} unique chunks across "
        f"{len(all_queries)} retrieval queries."
    )
    return merged_chunks
from ingestion.intent_detector import detect_intent
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.reranker import rerank

# Configuration per intent
INTENT_CONFIG = {
    "factual":       {"retrieval_k": 10, "llm_k": 3},
    "summarization": {"retrieval_k": 15, "llm_k": 7},
    "critique":      {"retrieval_k": 10, "llm_k": 5},
    "comparison":    {"retrieval_k": 15, "llm_k": 6},
    "mechanism":     {"retrieval_k": 10, "llm_k": 5},
    "explanation":   {"retrieval_k": 10, "llm_k": 5},
    "hypothetical":  {"retrieval_k": 15, "llm_k": 6},
    "analysis":      {"retrieval_k": 15, "llm_k": 7},
}


def resolve_config(intents: list) -> dict:
    """
    Given a list of 1-2 intents, returns the merged retrieval config.
    Strategy: take the maximum value for each parameter across all intents.
    More context is always better — CrossEncoder filters down anyway.
    """
    retrieval_k = max(INTENT_CONFIG[i]["retrieval_k"] for i in intents)
    llm_k       = max(INTENT_CONFIG[i]["llm_k"]       for i in intents)
    return {"retrieval_k": retrieval_k, "llm_k": llm_k}


def route_query(query: str, paper_name: str) -> dict:
    """
    Full routing pipeline:
    1. Detect intent(s)
    2. Resolve retrieval config from intent(s)
    3. Run hybrid retrieval with intent-aware top_k
    4. Re-rank with CrossEncoder
    5. Return chunks + metadata for answer generation

    Returns:
    {
        "query": str,
        "intents": list,
        "config": dict,
        "chunks": list   ← top chunks ready for LLM
    }
    """
    # Step 1 — classify intent
    intents = detect_intent(query)

    # Step 2 — resolve config
    config = resolve_config(intents)

    # Step 3 — hybrid retrieval with intent-aware retrieval_k
    rrf_results = hybrid_retrieve(query, paper_name, top_k=config["retrieval_k"])

    # Step 4 — rerank, return only llm_k chunks
    chunks = rerank(query, rrf_results, top_k=config["llm_k"])

    return {
        "query":   query,
        "intents": intents,
        "config":  config,
        "chunks":  chunks
    }
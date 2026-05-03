from ingestion.intent_detector import detect_intent
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.reranker import rerank
from ingestion.multi_hop import multi_hop_retrieve

# Intents that benefit from multi-hop retrieval
# These require connecting ideas from multiple sections
MULTI_HOP_INTENTS = {"mechanism", "comparison", "analysis", "hypothetical", "explanation"}

# Configuration per intent
INTENT_CONFIG = {
    "factual":       {"retrieval_k": 10, "llm_k": 3},
    "summarization": {"retrieval_k": 15, "llm_k": 7},
    "critique":      {"retrieval_k": 10, "llm_k": 5},
    "comparison":    {"retrieval_k": 12, "llm_k": 6},
    "mechanism":     {"retrieval_k": 10, "llm_k": 5},
    "explanation":   {"retrieval_k": 10, "llm_k": 5},
    "hypothetical":  {"retrieval_k": 12, "llm_k": 6},
    "analysis":      {"retrieval_k": 12, "llm_k": 7},
}


def resolve_config(intents: list) -> dict:
    """
    Given a list of 1-2 intents, returns the merged retrieval config.
    Takes the maximum value for each parameter across all intents.
    """
    retrieval_k = max(INTENT_CONFIG[i]["retrieval_k"] for i in intents)
    llm_k       = max(INTENT_CONFIG[i]["llm_k"]       for i in intents)
    return {"retrieval_k": retrieval_k, "llm_k": llm_k}


def needs_multi_hop(intents: list) -> bool:
    """Returns True if any detected intent benefits from multi-hop retrieval."""
    return any(i in MULTI_HOP_INTENTS for i in intents)


def route_query(query: str, paper_name: str) -> dict:
    """
    Full routing pipeline:
    1. Detect intent(s)
    2. Resolve retrieval config
    3. Run retrieval — multi-hop for complex intents, single-pass for simple
    4. Rerank with CrossEncoder
    5. Return chunks + metadata for generation

    Returns:
    {
        "query":   str,
        "intents": list,
        "config":  dict,
        "chunks":  list
    }
    """
    # Step 1 — classify intent
    intents = detect_intent(query)

    # Step 2 — resolve config
    config = resolve_config(intents)

    # Step 3 — retrieval strategy
    if needs_multi_hop(intents):
        print(f"[router] Multi-hop retrieval for intents: {intents}")
        raw_chunks = multi_hop_retrieve(
            query       = query,
            paper_name  = paper_name,
            retrieval_k = config["retrieval_k"]
        )
    else:
        print(f"[router] Single-pass retrieval for intents: {intents}")
        raw_chunks = hybrid_retrieve(query, paper_name, top_k=config["retrieval_k"])

    # Step 4 — rerank the merged pool, keep top llm_k
    chunks = rerank(query, raw_chunks, top_k=config["llm_k"])

    return {
        "query":   query,
        "intents": intents,
        "config":  config,
        "chunks":  chunks
    }
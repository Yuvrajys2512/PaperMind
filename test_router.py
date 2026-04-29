from ingestion.query_router import route_query

PAPER = "attention-is-all-you-need"

test_queries = [
    "What optimizer was used?",
    "Summarize the attention mechanism",
    "How does scaled dot-product attention work?",
    "Why did the authors remove recurrence?",
    "What are the weaknesses of this approach?",
    "How does this compare to RNNs?",
    "What would happen if positional encoding was removed?",
    "What are the trade-offs of using self-attention?",
]

for q in test_queries:
    result = route_query(q, PAPER)
    print(f"\nQuery   : {q}")
    print(f"Intents : {result['intents']}")
    print(f"Config  : retrieval_k={result['config']['retrieval_k']} | llm_k={result['config']['llm_k']}")
    print(f"Chunks  : {len(result['chunks'])} returned")
    for i, c in enumerate(result['chunks']):
        print(f"  {i+1}. [{c['metadata']['section']}] p{c['metadata']['page_num']} | rerank: {c['rerank_score']:.4f}")
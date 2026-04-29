from ingestion.query_router import route_query
from ingestion.generator import generate_answer

PAPER = "attention-is-all-you-need"

test_queries = [
    "What optimizer was used for training?",
    "How does scaled dot-product attention work?",
    "Why did the authors remove recurrence?",
    "What are the weaknesses of this approach?",
    "What would happen if positional encoding was removed?",
]

for query in test_queries:
    print("\n" + "="*70)
    print(f"QUERY   : {query}")

    # Route — intent detection + retrieval
    routed = route_query(query, PAPER)

    print(f"INTENTS : {routed['intents']}")
    print(f"CHUNKS  : {routed['config']['llm_k']} sent to LLM")

    # Generate answer
    result = generate_answer(query, routed["chunks"], routed["intents"])

    print(f"\nANSWER  :\n{result['answer']}")
    print(f"\nSOURCES :")
    for s in result["sources"]:
        print(f"  - [{s['section']}] Page {s['page']}")
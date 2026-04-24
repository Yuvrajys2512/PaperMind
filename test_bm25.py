from ingestion.bm25_retriever import bm25_retrieve

results = bm25_retrieve(
    query="Adam optimizer beta",
    paper_name="attention-is-all-you-need",
    top_k=5
)

for i, r in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Section : {r['metadata']['section']}")
    print(f"Page    : {r['metadata']['page_num']}")
    print(f"Score   : {r['bm25_score']:.4f}")
    print(f"Text    : {r['text'][:300]}")
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.reranker import rerank

PAPER = "attention-is-all-you-need"
QUERY = "Adam optimizer beta"

print("=" * 60)
print("BEFORE RERANKING (RRF order)")
print("=" * 60)
rrf_results = hybrid_retrieve(QUERY, PAPER, top_k=10)
for i, r in enumerate(rrf_results):
    print(f"{i+1}. [{r['metadata']['section']}] p{r['metadata']['page_num']} | rrf_score: {r['rrf_score']:.6f}")

print("\n" + "=" * 60)
print("AFTER RERANKING (CrossEncoder order)")
print("=" * 60)
reranked = rerank(QUERY, rrf_results, top_k=5)
for i, r in enumerate(reranked):
    print(f"{i+1}. [{r['metadata']['section']}] p{r['metadata']['page_num']} | rerank_score: {r['rerank_score']:.4f}")
    print(f"   Text: {r['text'][:150]}\n")
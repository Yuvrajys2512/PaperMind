from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.bm25_retriever import bm25_retrieve
from ingestion.retriever import retrieve

PAPER = "attention-is-all-you-need"
QUERY = "Adam optimizer beta"

print("=" * 60)
print("BM25 ONLY")
print("=" * 60)
for i, r in enumerate(bm25_retrieve(QUERY, PAPER, top_k=5)):
    print(f"{i+1}. [{r['metadata']['section']}] p{r['metadata']['page_num']} | score: {r['bm25_score']:.4f}")

print("\n" + "=" * 60)
print("VECTOR ONLY")
print("=" * 60)
for i, r in enumerate(retrieve(QUERY, PAPER, top_k=5)):
    print(f"{i+1}. [{r['metadata']['section']}] p{r['metadata']['page_num']} | distance: {r['distance']:.4f}")

print("\n" + "=" * 60)
print("HYBRID (RRF)")
print("=" * 60)
for i, r in enumerate(hybrid_retrieve(QUERY, PAPER, top_k=5)):
    print(f"{i+1}. [{r['metadata']['section']}] p{r['metadata']['page_num']} | rrf_score: {r['rrf_score']:.6f}")
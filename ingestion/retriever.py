import chromadb
from ingestion.models import get_embedding_model

client = chromadb.PersistentClient(path="data/chroma_db")


def retrieve(query: str, paper_name: str, top_k: int = 5) -> list:
    clean_name = "".join(
        c if c.isalnum() or c == "-" else "-"
        for c in paper_name
    ).strip("-").lower()

    collection = client.get_collection(name=clean_name)

    query_embedding = get_embedding_model().encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return retrieved

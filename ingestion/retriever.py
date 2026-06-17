import chromadb
from ingestion.models import embed_query

client = chromadb.PersistentClient(path="data/chroma_db")


def collection_name(paper_name: str) -> str:
    """Deterministic Chroma collection name for a paper_id. Single source of
    truth — used by retrieval, deletion, and startup regeneration alike."""
    return "".join(
        c if c.isalnum() or c == "-" else "-"
        for c in paper_name
    ).strip("-").lower()


def collection_exists(paper_name: str) -> bool:
    """True if this paper already has a Chroma collection on disk."""
    name = collection_name(paper_name)
    return any(c.name == name for c in client.list_collections())


def retrieve(query: str, paper_name: str, top_k: int = 5) -> list:
    clean_name = collection_name(paper_name)

    collection = client.get_collection(name=clean_name)

    # BGE expects the query instruction prefix; `embed_query` adds it.
    query_embedding = embed_query(query).tolist()

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

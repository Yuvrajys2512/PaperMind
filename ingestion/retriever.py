# ingestion/retriever.py

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma_db")


def retrieve(query: str, paper_name: str, top_k: int = 5) -> list:
    """
    Converts a query to a vector and finds the most similar chunks.
    
    Why top_k=5 as default?
    We return the 5 most relevant chunks. The LLM generating
    the answer will receive all 5 as context. More than 5
    risks including irrelevant content. Fewer than 5 risks
    missing important context.
    
    Returns a list of dicts, each containing:
    - text: the chunk content
    - metadata: section, page, etc.
    - distance: how similar it was (lower = more similar)
    """
    # Clean name same way as embedder
    clean_name = "".join(
        c if c.isalnum() or c == "-" else "-"
        for c in paper_name
    ).strip("-").lower()
    
    collection = client.get_collection(name=clean_name)
    
    # Convert query to vector
    query_embedding = model.encode(query).tolist()
    
    # Search for most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results into clean dicts
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return retrieved
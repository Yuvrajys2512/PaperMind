from rank_bm25 import BM25Okapi
import chromadb
import os

def build_bm25_index(paper_name: str):
    """
    Loads all chunks stored in ChromaDB for a given paper,
    tokenizes their text, builds and returns a BM25 index
    along with the raw chunk data.
    """
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection(name=paper_name)

    results = collection.get(include=["documents", "metadatas"])

    documents = results["documents"]   # list of chunk texts
    metadatas = results["metadatas"]   # list of metadata dicts
    ids = results["ids"]

    tokenized_corpus = [doc.lower().split() for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    chunks = [
        {"id": ids[i], "text": documents[i], "metadata": metadatas[i]}
        for i in range(len(documents))
    ]

    return bm25, chunks


def bm25_retrieve(query: str, paper_name: str, top_k: int = 5):
    """
    Runs BM25 retrieval for a query against a paper's stored chunks.
    Returns top_k results sorted by BM25 score descending.
    """
    bm25, chunks = build_bm25_index(paper_name)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = [
        {"text": chunks[i]["text"], "metadata": chunks[i]["metadata"], "bm25_score": scores[i]}
        for i in range(len(chunks))
    ]

    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)

    return scored_chunks[:top_k]
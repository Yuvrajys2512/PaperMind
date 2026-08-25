"""
ingestion/chroma_snapshot.py

Export/restore a paper's Chroma collection as a plain dict, so it can be
cached in R2 and restored on an ephemeral-disk host without re-running the
LLM ingestion pipeline (Launch Checklist 1.4).

No sentence-transformers/torch import here — restoring a snapshot re-inserts
already-computed embeddings, it never re-embeds anything.

Public API
----------
export_collection(paper_id) -> dict          {ids, embeddings, documents, metadatas}
restore_collection(paper_id, snapshot) -> bool   True if it restored, False if
                                                  the collection already existed
"""

from api.concurrency import get_chroma_client
from ingestion.retriever import collection_name


def export_collection(paper_id: str) -> dict:
    """Reads every id/embedding/document/metadata out of paper_id's Chroma
    collection. Raises (like retriever.retrieve) if the collection doesn't
    exist — call only on a paper that's actually 'ready'."""
    collection = get_chroma_client().get_collection(name=collection_name(paper_id))
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = data["embeddings"]
    return {
        "ids": data["ids"],
        # Chroma returns a numpy array; snapshots must be JSON-serializable.
        "embeddings": [[float(x) for x in row] for row in embeddings],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
    }


def restore_collection(paper_id: str, snapshot: dict) -> bool:
    """Recreates paper_id's Chroma collection from an exported snapshot —
    zero embedding-model or LLM calls, just re-inserting precomputed vectors.
    Idempotent: a no-op (returns False) if the collection already exists, same
    semantics as api/ingestion_runner.regenerate_missing_collections."""
    name = collection_name(paper_id)
    client = get_chroma_client()
    if any(c.name == name for c in client.list_collections()):
        return False

    collection = client.get_or_create_collection(name=name, metadata={"paper": paper_id})
    if snapshot["ids"]:
        collection.add(
            ids=snapshot["ids"],
            embeddings=snapshot["embeddings"],
            documents=snapshot["documents"],
            metadatas=snapshot["metadatas"],
        )
    return True

import chromadb

# ingestion.models pulls in sentence-transformers/torch, and importing torch in
# the same process as docling segfaults on Windows (write_mode_handoff.md §7).
# Only retrieve() needs the embedder — get_all_chunks/collection_name are pure
# store access — so the import is deferred into the one function that uses it.
# Keeping it at module scope meant merely importing this module dragged torch
# in, which is what made `pytest tests/` die at collection.

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
    from ingestion.models import embed_query  # lazy — see module header

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


def _reading_order_key(chunks: list):
    """Pick the best available sort key for restoring document order.

    `doc_index` is the chunk's ordinal in the original document, written by
    embed_and_store. It is exact, but only collections ingested after it was
    added carry it — so it is used only when *every* chunk has one. Mixing the
    two schemes within a collection would sort the annotated chunks as a block
    ahead of the rest, which is worse than either scheme alone.

    The fallback, (page, index-within-section), is approximate: several sections
    can start on one page and all report index 0. Python's sort is stable, so
    those ties keep the order the store returned rather than being shuffled.

    Caveat on tables: ingest_document appends table chunks after all prose, so
    their doc_index puts them at the end of the document rather than on the page
    they were printed on. Harmless today — every caller that cares about running
    prose (citation_gap_auditor, plagiarism_auditor) filters
    `section_type == "table"` out first, and the callers that want tables
    (numbers_auditor, claim_auditor) look them up by section rather than by
    position. Folding page_num into this key would place them better but would
    also let a missing page_num disorder the prose, which is the case that
    actually matters.
    """
    if chunks and all("doc_index" in c["metadata"] for c in chunks):
        return lambda c: (c["metadata"].get("doc_index") or 0,)
    return lambda c: (c["metadata"].get("page_num") or 0,
                      c["metadata"].get("chunk_index") or 0)


def get_all_chunks(paper_name: str) -> list:
    """Return every stored chunk for a paper as {"text", "metadata"} dicts,
    in the paper's reading order.

    Unlike retrieve(), this does no embedding/search — it pulls the whole
    collection so callers can scan by section metadata (e.g. the claim auditor
    grabbing the Abstract / Introduction / Conclusion bodies). Single-paper
    collections are small (typically <150 chunks), so a full get() is cheap.

    Reading order matters to more callers than it looks: citation_gap_auditor
    checks whether the *next* sentence carries the citation marker, and
    plagiarism_auditor flattens chunks into one word stream that matches run
    across chunk boundaries. Both quietly produce wrong answers on scrambled
    input.

    This used to sort on a `chunk_id` metadata key that the embedder never
    writes (it writes `chunk_index`), so `.get("chunk_id", 0)` returned 0 for
    every chunk and the sort was a no-op — callers were served Chroma's own
    ordering, which matched the document only by luck.
    """
    collection = client.get_collection(name=collection_name(paper_name))
    got = collection.get(include=["documents", "metadatas"])

    chunks = []
    for doc, meta in zip(got["documents"], got["metadatas"]):
        chunks.append({"text": doc, "metadata": meta or {}})

    chunks.sort(key=_reading_order_key(chunks))
    return chunks

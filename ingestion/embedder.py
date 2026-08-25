# ingestion/embedder.py

import hashlib

from api.concurrency import get_chroma_client
from ingestion.models import embed_passages


def get_or_create_collection(paper_name: str):
    """
    Gets an existing collection for this paper or creates a new one.
    
    Why use paper name as collection name?
    Each paper gets its own isolated vector space.
    Searching one paper never returns results from another.
    
    ChromaDB collection names must be alphanumeric with hyphens only
    so we clean the paper name first.
    """
    # Clean the name — remove spaces and special characters
    clean_name = "".join(
        c if c.isalnum() or c == "-" else "-" 
        for c in paper_name
    ).strip("-").lower()
    
    return get_chroma_client().get_or_create_collection(
        name=clean_name,
        metadata={"paper": paper_name}
    )


def embed_and_store(chunks: list, paper_name: str) -> None:
    """
    Converts chunks to vectors and stores them in ChromaDB.
    
    Why batch embedding?
    The embedding model can process multiple texts at once
    much faster than one at a time. We send all chunk texts
    together and get all vectors back in one call.
    
    Args:
        chunks: the list of chunk dicts from chunk_sections()
        paper_name: used to name the collection
    """
    collection = get_or_create_collection(paper_name)
    
    # Extract just the text from each chunk for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Embedding {len(texts)} chunks...")

    # Convert all texts to vectors in one batch call.
    # `embed_passages` uses no instruction prefix because BGE was trained
    # asymmetrically: only queries get the prefix, not chunks.
    embeddings = embed_passages(texts)
    
    # Prepare metadata for each chunk
    # ChromaDB metadata values must be strings, ints, or floats — not lists
    # `doc_index` is the chunk's position in the document as a whole, which
    # nothing else in the metadata records: `chunk_index` restarts at 0 in every
    # section, and several sections can share a page. Retrieval needs true
    # reading order (see retriever.get_all_chunks), and `chunks` arrives here in
    # document order, so the enumeration index is exactly that.
    metadatas = [
        {
            "section": chunk["section"],
            "section_type": chunk["section_type"],
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
            "doc_index": i,
            "total_chunks_in_section": chunk["total_chunks_in_section"],
            "token_count": chunk["token_count"]
        }
        for i, chunk in enumerate(chunks)
    ]
    
    # Generate unique IDs for each chunk. Hashing the text keeps re-ingesting
    # the same paper idempotent (identical chunks → identical IDs → upsert
    # overwrites). We prefix the chunk's ordinal position so two chunks with
    # *identical text* in one paper (e.g. a repeated header or table row) don't
    # collide — a bare text hash made ChromaDB reject the whole batch with
    # DuplicateIDError. Order is deterministic per PDF, so idempotency holds.
    ids = [
        hashlib.md5(f"{i}:{chunk['text']}".encode()).hexdigest()
        for i, chunk in enumerate(chunks)
    ]
    
    # Store everything in ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )

    from ingestion.bm25_retriever import invalidate_bm25_cache
    invalidate_bm25_cache(paper_name)

    print(f"Stored {len(chunks)} chunks in collection '{paper_name}'")
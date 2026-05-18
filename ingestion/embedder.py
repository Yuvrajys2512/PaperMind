# ingestion/embedder.py

import chromadb
import hashlib

from ingestion.models import embed_passages

# ChromaDB client — PersistentClient means data is saved to disk
# so you don't have to re-embed every time you restart the program
client = chromadb.PersistentClient(path="data/chroma_db")


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
    
    return client.get_or_create_collection(
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
    metadatas = [
        {
            "section": chunk["section"],
            "section_type": chunk["section_type"],
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks_in_section": chunk["total_chunks_in_section"],
            "token_count": chunk["token_count"]
        }
        for chunk in chunks
    ]
    
    # Generate unique IDs for each chunk
    # We use a hash of the text so re-running doesn't create duplicates
    ids = [
        hashlib.md5(chunk["text"].encode()).hexdigest()
        for chunk in chunks
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
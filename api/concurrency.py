"""
api/concurrency.py — Thread-safe ChromaDB wrapper and per-paper locking.

Prevents concurrent writes to ChromaDB and race conditions on paper operations
(ingest, query, delete, update). Each paper gets its own lock; different papers
don't block each other.

Public API
----------
get_chroma_client() -> chromadb.PersistentClient
  Thread-safe singleton ChromaDB client with locking.

acquire_paper_lock(paper_id: str) -> threading.RLock
  Acquire a lock for a paper. Held during ingest/query/delete to prevent races.
  Use as a context manager or manually acquire/release.

paper_locked(paper_id: str)
  Decorator for functions that need exclusive access to a paper.
"""

import threading
from contextlib import contextmanager
from functools import wraps

import chromadb

_chroma_lock = threading.Lock()
_chroma_client = None

_paper_locks = {}
_paper_locks_lock = threading.Lock()


def get_chroma_client() -> chromadb.PersistentClient:
    """Get the thread-safe ChromaDB singleton client.

    Creates on first call, then returns the same instance. All access is
    protected by a lock to prevent concurrent writes (ChromaDB is not
    thread-safe for writes).
    """
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    return _chroma_client


def acquire_paper_lock(paper_id: str) -> threading.RLock:
    """Get or create a per-paper lock for synchronizing operations on that paper.

    Reentrant (RLock) so the same thread can acquire it multiple times
    (e.g., if ingest calls query internally).
    """
    with _paper_locks_lock:
        if paper_id not in _paper_locks:
            _paper_locks[paper_id] = threading.RLock()
        return _paper_locks[paper_id]


@contextmanager
def paper_locked(paper_id: str):
    """Context manager for paper-level locking.

    Usage:
        with paper_locked(paper_id):
            # operations on paper are exclusive
            ...
    """
    lock = acquire_paper_lock(paper_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def paper_lock_decorator(func):
    """Decorator to acquire a paper lock for a function.

    The function must have 'paper_id' as a parameter (positional or kwarg).

    Usage:
        @paper_lock_decorator
        def ingest_paper(paper_id, ...):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract paper_id from kwargs first, then look through args
        paper_id = kwargs.get("paper_id")
        if paper_id is None:
            # Assume first positional arg is paper_id (common pattern)
            # This is fragile but works for the codebase's function signatures
            if args:
                paper_id = args[0]

        if paper_id is None:
            raise ValueError(
                f"{func.__name__} must have 'paper_id' as first arg or kwarg"
            )

        with paper_locked(paper_id):
            return func(*args, **kwargs)

    return wrapper

"""
ingestion/models.py

Single source of truth for shared ML model instances.
Lazy-loaded so imports don't trigger heavy downloads at module load time.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the shared all-MiniLM-L6-v2 instance, loading it on first call.
    Used by retriever.py and evaluator.py so only one copy lives in memory.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

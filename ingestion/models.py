"""
ingestion/models.py

Single source of truth for shared ML model instances.
Lazy-loaded so imports don't trigger heavy downloads at module load time.

Embedder
--------
`BAAI/bge-small-en-v1.5` (384-dim, same dim as the previous MiniLM model).
BGE is an asymmetric retrieval model: queries should be prefixed with an
instruction, passages should not. Use `embed_query()` and
`embed_passages()` to apply the right convention automatically.

NOTE: BGE and MiniLM live in different embedding spaces. Any ChromaDB
collection ingested under MiniLM must be re-ingested before it can be
searched with BGE — they are not vector-compatible.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

# Recommended query instruction from the BGE model card.
# It is only prepended to *query* text — never to passages — because BGE
# was fine-tuned with this asymmetric pattern.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Return the shared BGE-small instance, loading it on first call.
    Prefer ``embed_query`` / ``embed_passages`` over raw ``.encode(...)`` so
    the BGE query instruction is applied where appropriate.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedding_model


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. Applies the BGE instruction prefix."""
    return get_embedding_model().encode(_BGE_QUERY_INSTRUCTION + text)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed one or more passages/chunks. No prefix — BGE was trained that way."""
    return get_embedding_model().encode(texts, show_progress_bar=False)

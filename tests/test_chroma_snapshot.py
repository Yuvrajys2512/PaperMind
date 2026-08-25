"""
Tests for ingestion/chroma_snapshot.py — export/restore a Chroma collection
without any embedding-model or LLM calls (Launch Checklist 1.4).

Uses a temp-directory PersistentClient (monkeypatched onto the shared
api.concurrency accessor — see Launch Checklist 2.9) rather than the real
data/chroma_db, so this suite never touches dev data.
"""

import json

import chromadb
import pytest

from api import concurrency
from ingestion import chroma_snapshot
from ingestion.retriever import collection_name


@pytest.fixture
def temp_client(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(concurrency, "_chroma_client", client)
    return client


def _seed(client, paper_id: str):
    collection = client.create_collection(name=collection_name(paper_id))
    collection.add(
        ids=["c0", "c1"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        documents=["first chunk", "second chunk"],
        metadatas=[{"section": "intro", "page_num": 1}, {"section": "body", "page_num": 2}],
    )


def test_export_returns_json_serializable_floats(temp_client):
    _seed(temp_client, "paper-1")
    snapshot = chroma_snapshot.export_collection("paper-1")

    # numpy floats/arrays would blow up json.dumps — this is the exact bug
    # the [float(x) for x in row] conversion guards against.
    json.dumps(snapshot)

    assert snapshot["ids"] == ["c0", "c1"]
    assert snapshot["documents"] == ["first chunk", "second chunk"]
    assert all(isinstance(x, float) for row in snapshot["embeddings"] for x in row)


def test_restore_recreates_an_equivalent_collection(temp_client):
    _seed(temp_client, "paper-2")
    snapshot = chroma_snapshot.export_collection("paper-2")

    # Simulate the ephemeral-disk wipe: drop the collection, then restore
    # purely from the exported snapshot dict.
    temp_client.delete_collection(name=collection_name("paper-2"))
    restored = chroma_snapshot.restore_collection("paper-2", snapshot)
    assert restored is True

    collection = temp_client.get_collection(name=collection_name("paper-2"))
    data = collection.get(include=["documents", "metadatas"])
    assert sorted(data["documents"]) == ["first chunk", "second chunk"]


def test_restore_is_a_noop_when_collection_already_exists(temp_client):
    _seed(temp_client, "paper-3")
    snapshot = chroma_snapshot.export_collection("paper-3")

    # Collection still exists (never deleted) — restoring must not touch it,
    # matching regenerate_missing_collections' "skip if present" semantics.
    restored = chroma_snapshot.restore_collection("paper-3", snapshot)
    assert restored is False


def test_restore_handles_an_empty_collection(temp_client):
    """A paper that ingested to zero chunks (shouldn't normally happen, but
    the export/restore path must not crash on it) round-trips cleanly."""
    name = collection_name("paper-4")
    temp_client.create_collection(name=name)
    snapshot = chroma_snapshot.export_collection("paper-4")
    assert snapshot["ids"] == []

    temp_client.delete_collection(name=name)
    restored = chroma_snapshot.restore_collection("paper-4", snapshot)
    assert restored is True
    assert temp_client.get_collection(name=name).count() == 0

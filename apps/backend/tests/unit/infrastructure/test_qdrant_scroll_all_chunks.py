"""
Tests for QdrantChunkStore.scroll_all_chunks — the method BM25Retriever now
calls at startup to rebuild its in-memory index from Qdrant (the durable
source of truth), fixing the "reload wipes sparse search" bug.

We mock the qdrant_client.QdrantClient entirely so these tests don't need a
real Qdrant instance (embedded or remote) — they only exercise our
pagination and Chunk-reconstruction logic.
"""

from unittest.mock import MagicMock, patch

from rag_harness.config.settings import EmbeddingSettings, EnvironmentSettings
from rag_harness.infrastructure.vector_stores.qdrant_chunk_store import QdrantChunkStore


def _fake_point(point_id: str, doc_id: str = "doc1"):
    point = MagicMock()
    point.id = point_id
    point.payload = {
        "raw_text": f"raw {point_id}",
        "contextual_text": f"contextual {point_id}",
        "doc_id": doc_id,
        "doc_version": "v1",
        "source_type": "pdf",
    }
    return point


def _make_store():
    with patch("rag_harness.infrastructure.vector_stores.qdrant_chunk_store.QdrantClient") as MockClient:
        client = MockClient.return_value
        client.get_collections.return_value = MagicMock(collections=[])
        environment_settings = EnvironmentSettings()
        store = QdrantChunkStore(EmbeddingSettings(), environment_settings)
        return store, client


def test_scroll_all_chunks_single_page():
    store, client = _make_store()
    client.scroll.return_value = ([_fake_point("c1"), _fake_point("c2")], None)

    chunks = store.scroll_all_chunks(batch_size=256)

    assert len(chunks) == 2
    assert {c.chunk_id for c in chunks} == {"c1", "c2"}
    assert chunks[0].raw_text == "raw c1"
    client.scroll.assert_called_once()


def test_scroll_all_chunks_follows_pagination_offset_until_none():
    store, client = _make_store()
    # Simulate 3 pages: first two return a next-offset, last returns None.
    client.scroll.side_effect = [
        ([_fake_point("c1")], "offset-1"),
        ([_fake_point("c2")], "offset-2"),
        ([_fake_point("c3")], None),
    ]

    chunks = store.scroll_all_chunks(batch_size=1)

    assert {c.chunk_id for c in chunks} == {"c1", "c2", "c3"}
    assert client.scroll.call_count == 3


def test_scroll_all_chunks_on_empty_collection_returns_empty_list():
    store, client = _make_store()
    client.scroll.return_value = ([], None)

    chunks = store.scroll_all_chunks()

    assert chunks == []
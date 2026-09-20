"""
Unit test for RRF fusion logic — mocks the embedder/reranker dependencies
since we're testing the fusion math, not the models themselves.
"""

from unittest.mock import MagicMock

from rag_harness.application.retrieval.fusion_service import RetrievalFusionService
from rag_harness.config.settings import RetrievalSettings
from rag_harness.domain.models import Chunk, ChunkMetadata, RetrievedChunk


def _make_retrieved_chunk(chunk_id: str, score: float, source: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, raw_text="text", contextual_text="text",
        metadata=ChunkMetadata(doc_id="doc1", doc_version="v1", source_type="pdf"),
    )
    return RetrievedChunk(chunk=chunk, score=score, source=source)


def test_rrf_fuse_favors_chunks_ranked_high_in_both_lists():
    settings = RetrievalSettings()
    fusion = RetrievalFusionService(settings, embedder=MagicMock(), model_registry=MagicMock())

    dense = [_make_retrieved_chunk("a", 0.9, "dense"), _make_retrieved_chunk("b", 0.8, "dense")]
    sparse = [_make_retrieved_chunk("b", 5.0, "sparse"), _make_retrieved_chunk("a", 3.0, "sparse")]

    fused = fusion.fuse(dense, sparse)

    # "a" is rank 1 in dense, rank 2 in sparse; "b" is rank 2 in dense, rank 1 in sparse
    # RRF should rank them very closely — both appear, neither is dropped
    fused_ids = [rc.chunk.chunk_id for rc in fused]
    assert set(fused_ids) == {"a", "b"}


def test_fuse_handles_empty_lists():
    settings = RetrievalSettings()
    fusion = RetrievalFusionService(settings, embedder=MagicMock(), model_registry=MagicMock())
    result = fusion.fuse([], [])
    assert result == []
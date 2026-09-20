"""
Unit tests for domain models — no infrastructure, no mocks needed since
these are pure data classes.
"""

from rag_harness.domain.models import Chunk, ChunkMetadata, Claim, RetrievedChunk


def test_chunk_metadata_requires_doc_id_and_version():
    metadata = ChunkMetadata(doc_id="doc1", doc_version="v1", source_type="pdf")
    assert metadata.doc_id == "doc1"
    assert metadata.page_num is None


def test_chunk_generates_unique_id_by_default():
    metadata = ChunkMetadata(doc_id="doc1", doc_version="v1", source_type="pdf")
    chunk_a = Chunk(raw_text="a", contextual_text="a", metadata=metadata)
    chunk_b = Chunk(raw_text="b", contextual_text="b", metadata=metadata)
    assert chunk_a.chunk_id != chunk_b.chunk_id


def test_retrieved_chunk_wraps_chunk_with_score():
    metadata = ChunkMetadata(doc_id="doc1", doc_version="v1", source_type="pdf")
    chunk = Chunk(raw_text="text", contextual_text="text", metadata=metadata)
    retrieved = RetrievedChunk(chunk=chunk, score=0.85, source="dense")
    assert retrieved.score == 0.85
    assert retrieved.source == "dense"


def test_claim_holds_citation_list():
    claim = Claim(claim_text="X is true", cited_chunk_ids=["c1", "c2"])
    assert len(claim.cited_chunk_ids) == 2
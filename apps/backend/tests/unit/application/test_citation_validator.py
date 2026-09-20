"""
Unit test for the citation validator — the cheapest, code-only verification
check. No LLM, no vector store; pure logic against a chunk registry.
"""

from rag_harness.application.verification.citation_validator import CitationValidator
from rag_harness.domain.models import Chunk, ChunkMetadata, Claim


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, raw_text="text", contextual_text="text",
        metadata=ChunkMetadata(doc_id="doc1", doc_version="v1", source_type="pdf"),
    )


def test_validate_passes_when_all_citations_exist():
    registry = {"c1": _make_chunk("c1"), "c2": _make_chunk("c2")}
    claims = [Claim(claim_text="A", cited_chunk_ids=["c1"]), Claim(claim_text="B", cited_chunk_ids=["c2"])]

    validator = CitationValidator()
    is_valid, hallucinated = validator.validate(claims, registry)

    assert is_valid is True
    assert hallucinated == []


def test_validate_flags_hallucinated_citation():
    registry = {"c1": _make_chunk("c1")}
    claims = [Claim(claim_text="A", cited_chunk_ids=["c1", "c_nonexistent"])]

    validator = CitationValidator()
    is_valid, hallucinated = validator.validate(claims, registry)

    assert is_valid is False
    assert hallucinated == ["c_nonexistent"]


def test_validate_passes_on_empty_claims():
    validator = CitationValidator()
    is_valid, hallucinated = validator.validate([], {})
    assert is_valid is True
    assert hallucinated == []
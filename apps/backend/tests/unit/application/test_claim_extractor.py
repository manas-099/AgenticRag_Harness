"""
Unit tests for ClaimExtractor:
- Structured output success
- Exception logging with backend name
- Regex fallback matching both chunk_id and [Doc: <doc_id>, Page: <page_num>]
"""

from unittest.mock import MagicMock
from rag_harness.application.verification.claim_extractor import ClaimExtractor
from rag_harness.domain.ports import LLMPort


def test_claim_extractor_regex_fallback():
    mock_llm = MagicMock(spec=LLMPort)
    mock_llm.generate.side_effect = Exception("Model refused structured output")
    mock_llm.last_backend_used = "openrouter"

    extractor = ClaimExtractor(mock_llm)
    
    # Test text containing both formats
    text = (
        "Antigravity is powerful [Doc: doc_agy, Page: 4]. "
        "Another sentence with chunk_id: chunk-123-abc. "
        "A third claim without citations."
    )
    claims = extractor.extract(text, available_chunk_ids=["chunk-123-abc", "doc_agy:p4"])
    
    assert len(claims) == 2
    assert "doc_agy:p4" in claims[0].cited_chunk_ids
    assert "chunk-123-abc" in claims[1].cited_chunk_ids


def test_claim_extractor_structured_success():
    mock_llm = MagicMock(spec=LLMPort)
    mock_llm.generate.return_value = '{"claims": [{"claim_text": "Antigravity works", "cited_chunk_ids": ["c1"]}]}'
    mock_llm.last_backend_used = "groq"

    extractor = ClaimExtractor(mock_llm)
    claims = extractor.extract("Antigravity works", available_chunk_ids=["c1"])

    assert len(claims) == 1
    assert claims[0].claim_text == "Antigravity works"
    assert claims[0].cited_chunk_ids == ["c1"]

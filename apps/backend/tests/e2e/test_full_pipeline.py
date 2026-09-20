"""
End-to-end smoke test — requires a real (local, embedded) Qdrant instance and
a real LLM backend configured via .env. Skipped by default in CI unless
RUN_E2E_TESTS=1 is set, since it costs real API calls and needs a test file.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E_TESTS") != "1",
    reason="E2E tests require RUN_E2E_TESTS=1 and a configured .env",
)


def test_ingest_then_query_returns_grounded_answer():
    from rag_harness.api.dependencies import get_rag_pipeline

    pipeline = get_rag_pipeline()
    test_file = os.getenv("E2E_TEST_FILE", "tests/fixtures/sample_document.pdf")

    chunks = pipeline.ingest_document(test_file, doc_id="e2e_test_doc", doc_version="v1")
    assert len(chunks) > 0
    assert all(c.embedding is not None for c in chunks)

    result = pipeline.answer("What is this document about?")
    assert result.answer
    assert not result.from_cache


def test_out_of_scope_question_returns_insufficient():
    from rag_harness.api.dependencies import get_rag_pipeline

    pipeline = get_rag_pipeline()
    result = pipeline.answer("What is the boiling point of mercury on Mars?")
    assert result.is_insufficient or "don't have enough information" in result.answer.lower()
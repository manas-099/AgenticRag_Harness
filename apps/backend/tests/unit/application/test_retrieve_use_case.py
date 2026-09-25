"""
Tests for RetrieveUseCase's failure handling. Previously, RetrieveUseCase's
_run_dense and _run_sparse each swallowed ANY exception and returned [], so
a broken embedder, a missing Qdrant collection, or an unindexed BM25 store
all looked identical to "there were genuinely no relevant chunks" — the
caller (agent nodes.py) couldn't tell the difference and just degraded
silently with an HTTP 200.

Fixed behavior:
- One leg failing -> log a warning, continue with the other leg's results
  (a partial result is still useful).
- BOTH legs failing -> raise RetrievalError, so a genuinely broken pipeline
  is distinguishable from a query that legitimately has no matches.
"""

from unittest.mock import MagicMock

import pytest

from rag_harness.application.retrieval.fusion_service import RetrievalFusionService
from rag_harness.application.retrieval.retrieve_use_case import RetrieveUseCase
from rag_harness.config.settings import RetrievalSettings
from rag_harness.domain.exceptions import RetrievalError


def _use_case(embedder=None, sparse_retriever=None, vector_store=None):
    return RetrieveUseCase(
        vector_store=vector_store or MagicMock(),
        sparse_retriever=sparse_retriever or MagicMock(),
        fusion_service=RetrievalFusionService(
            RetrievalSettings(), embedder=MagicMock(), model_registry=MagicMock()
        ),
        embedder=embedder or MagicMock(),
        settings=RetrievalSettings(),
    )


def test_dense_failure_alone_degrades_gracefully_without_raising():
    """One leg failing is NOT fatal — we log a warning and continue with
    whatever the other leg found (empty here), since a single leg failing
    must not raise RetrievalError."""
    broken_embedder = MagicMock()
    broken_embedder.encode.side_effect = RuntimeError("embedding model not loaded")

    sparse_retriever = MagicMock()
    sparse_retriever.search.return_value = []

    use_case = _use_case(embedder=broken_embedder, sparse_retriever=sparse_retriever)

    result = use_case.execute("what is agentic rag?")
    assert result == []


def test_sparse_failure_alone_degrades_gracefully_without_raising():
    sparse_retriever = MagicMock()
    sparse_retriever.search.side_effect = RuntimeError("bm25 index not built")

    vector_store = MagicMock()
    vector_store.dense_search.return_value = []

    use_case = _use_case(sparse_retriever=sparse_retriever, vector_store=vector_store)

    result = use_case.execute("what is agentic rag?")
    assert result == []


def test_both_legs_failing_now_raises_retrieval_error_instead_of_silent_empty_list():
    """
    THE FIX: previously this scenario returned [] silently (documented as
    a bug). Now a genuine infra failure on both legs surfaces as a
    RetrievalError instead of being indistinguishable from "no relevant
    content in the corpus".
    """
    broken_embedder = MagicMock()
    broken_embedder.encode.side_effect = RuntimeError("dimension mismatch: expected 768, got 384")

    sparse_retriever = MagicMock()
    sparse_retriever.search.side_effect = RuntimeError("bm25 index not built")

    use_case = _use_case(embedder=broken_embedder, sparse_retriever=sparse_retriever)

    with pytest.raises(RetrievalError, match="Both dense and sparse retrieval failed"):
        use_case.execute("what is agentic rag?")


def test_both_legs_genuinely_empty_still_returns_empty_list_not_an_error():
    """Distinguishes 'both legs ran fine and found nothing' (a real,
    legitimate empty result) from 'both legs broke' (an error) — this
    must still return [] cleanly, not raise."""
    embedder = MagicMock()
    embedder.encode.return_value = [[0.0] * 768]

    vector_store = MagicMock()
    vector_store.dense_search.return_value = []

    sparse_retriever = MagicMock()
    sparse_retriever.search.return_value = []

    use_case = _use_case(embedder=embedder, sparse_retriever=sparse_retriever, vector_store=vector_store)

    result = use_case.execute("a query with genuinely no matches")
    assert result == []
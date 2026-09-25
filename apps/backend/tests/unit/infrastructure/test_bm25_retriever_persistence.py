
"""
Reproduces the "reload wipes sparse index" bug: BM25Retriever keeps its
entire corpus and index in plain Python instance state (self.corpus_chunks,
self.bm25). Nothing persists it to disk or to Qdrant. Under `uvicorn --reload`
(WatchFiles), any source-file save recreates the process and this object,
so chunks ingested before a reload become invisible to sparse search even
though the dense vectors are still safely in Qdrant.
"""

from unittest.mock import MagicMock

from rag_harness.domain.models import Chunk, ChunkMetadata
from rag_harness.infrastructure.sparse_search import BM25Retriever


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        raw_text=text,
        contextual_text=text,
        metadata=ChunkMetadata(doc_id="doc1", doc_version="v1", source_type="pdf"),
    )


def test_bm25_returns_results_after_indexing():
    retriever = BM25Retriever()
    retriever.index([_chunk("c1", "agentic rag uses an agent to control retrieval")])

    results = retriever.search("agentic rag", top_k=5)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"


def test_bm25_matches_survive_on_a_tiny_single_document_corpus():
    """
    Regression test for the real bug found while testing: BM25Okapi's IDF
    term goes negative when a query term appears in (nearly) every document
    in a small corpus, so an exact match on a 1-document corpus previously
    scored negative and was filtered out by `if s > 0` in search().
    On a tiny/early-stage corpus this reproduced the exact "no chunks
    retrieved" symptom even though the document plainly matched the query.
    """
    retriever = BM25Retriever()
    retriever.index([_chunk("c1", "agentic rag uses an agent to control retrieval")])

    raw_scores = retriever.bm25.get_scores("agentic rag".lower().split())
    assert raw_scores[0] < 0, (
        "sanity check: this only reproduces the bug if the underlying BM25 "
        "score is actually negative for this corpus size — if rank_bm25's "
        "IDF formula changes, this assertion (not the fix) needs revisiting"
    )

    results = retriever.search("agentic rag", top_k=5)

    assert len(results) == 1, "a genuine term match must not be dropped just because its raw BM25 score is <= 0"
    assert results[0].chunk.chunk_id == "c1"


def test_bm25_still_excludes_chunks_with_no_query_term_overlap():
    """The fix must not turn search() into 'return everything' — chunks
    that share zero terms with the query should still be excluded."""
    retriever = BM25Retriever()
    retriever.index([
        _chunk("relevant", "agentic rag uses an agent to control retrieval"),
        _chunk("irrelevant", "the weather today is sunny and warm"),
    ])

    results = retriever.search("agentic rag retrieval", top_k=5)

    result_ids = {rc.chunk.chunk_id for rc in results}
    assert "relevant" in result_ids
    assert "irrelevant" not in result_ids


def test_rebuild_from_vector_store_repopulates_a_fresh_retriever():
    """
    This is the actual fix for the reload-wipes-state bug: a brand-new
    BM25Retriever (as constructed after every `--reload`/process restart)
    should be repopulated from the vector store (Qdrant), which is the
    durable source of truth, instead of starting empty.
    """
    fake_vector_store = MagicMock()
    fake_vector_store.scroll_all_chunks.return_value = [
        _chunk("c1", "agentic rag architecture"),
        _chunk("c2", "standard rag is a one-shot pipeline"),
    ]

    retriever_after_reload = BM25Retriever()
    assert retriever_after_reload.is_indexed is False  # starts empty, as before

    retriever_after_reload.rebuild_from_vector_store(fake_vector_store)

    assert retriever_after_reload.is_indexed is True
    assert retriever_after_reload.corpus_size == 2
    assert retriever_after_reload.search("agentic rag", top_k=5) != []


def test_rebuild_from_vector_store_on_empty_store_leaves_retriever_unindexed_not_broken():
    """A genuinely empty corpus (first-ever startup, nothing ingested yet)
    should not raise — it should just leave the retriever gracefully
    unindexed, same as before any documents exist."""
    fake_vector_store = MagicMock()
    fake_vector_store.scroll_all_chunks.return_value = []

    retriever = BM25Retriever()
    retriever.rebuild_from_vector_store(fake_vector_store)

    assert retriever.is_indexed is False
    assert retriever.search("anything", top_k=5) == []


def test_rebuild_from_vector_store_swallows_qdrant_errors_without_crashing_startup():
    """If Qdrant is briefly unreachable at startup, rebuild should log and
    leave the retriever empty rather than crashing the whole app boot —
    matches the existing 'fail soft, log loud' pattern used elsewhere in
    this codebase (e.g. _resync_sections)."""
    fake_vector_store = MagicMock()
    fake_vector_store.scroll_all_chunks.side_effect = RuntimeError("qdrant unreachable")

    retriever = BM25Retriever()
    retriever.rebuild_from_vector_store(fake_vector_store)  # must not raise

    assert retriever.is_indexed is False


def test_bm25_search_before_any_index_call_returns_empty_not_error():
    """
    Matches the observed 'no_chunks_retrieved' symptom: a brand-new
    BM25Retriever (e.g. right after a reload) silently returns [] rather
    than raising, so callers get no signal that the index is missing.
    """
    retriever = BM25Retriever()
    assert retriever.is_indexed is False

    results = retriever.search("agentic rag", top_k=5)

    assert results == []


def test_simulated_reload_loses_previously_ingested_chunks_without_the_fix():
    """
    Documents what happens WITHOUT calling rebuild_from_vector_store():
    a fresh BM25Retriever after reload starts empty even though the same
    document is still sitting in Qdrant. See
    test_rebuild_from_vector_store_repopulates_a_fresh_retriever for the fix.
    """
    # --- process A: ingest ---
    retriever_before_reload = BM25Retriever()
    retriever_before_reload.corpus_chunks.append(_chunk("c1", "agentic rag architecture"))
    retriever_before_reload.index(retriever_before_reload.corpus_chunks)
    assert retriever_before_reload.corpus_size == 1
    assert retriever_before_reload.search("agentic rag architecture", top_k=5) != []

    # --- reload happens: DI re-instantiates BM25Retriever from scratch,
    #     and (in this scenario) rebuild_from_vector_store is NOT called ---
    retriever_after_reload = BM25Retriever()

    # --- process B: query for the same content ---
    results_after_reload = retriever_after_reload.search("agentic rag", top_k=5)

    # Documents the bug: the chunk ingested before reload is gone.
    assert retriever_after_reload.corpus_size == 0
    assert results_after_reload == []
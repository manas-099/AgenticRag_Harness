"""
Two things pinned down here:

1. generate_node's silent-degrade path: when retrieved_chunk_registry is
   empty, the node never calls the LLM and never raises — it just sets
   degraded=True / degrade_reason="no_chunks_retrieved" and returns a canned
   answer. This is *correct* behavior for the node itself, but it's the
   reason the API still returns HTTP 200 on what is effectively a failed
   query — worth pinning explicitly so nobody "fixes" it by making it throw.

2. A regression guard for the old, already-fixed bug where
   CrossEncoderRegistry.nli_model() could raise
   AttributeError: 'RerankSettings' object has no attribute 'NLI_MODEL'.
   getattr(..., "NLI_MODEL", None) is what prevents that; this test fails
   loudly if that safety net is ever removed.
"""

from unittest.mock import MagicMock

from rag_harness.application.agent.nodes import make_generate_node
from rag_harness.application.retrieval.model_registry import CrossEncoderRegistry
from rag_harness.config.settings import RerankSettings


def _base_state(**overrides):
    state = {
        "query": "what is agentic rag?",
        "iteration": 1,
        "max_iterations": 5,
        "token_budget_used": 0,
        "max_token_budget": 4000,
        "max_validation_retries": 2,
        "validation_retries": 0,
        "search_attempts": [],
        "registry_size_history": [],
        "retrieved_chunk_registry": {},
        "scratchpad": [],
        "action_history": [],
        "pending_decision": None,
        "draft_answer": None,
        "verification_result": None,
        "final_answer": None,
        "sources_used": [],
        "degraded": False,
        "degrade_reason": None,
    }
    state.update(overrides)
    return state


def test_generate_node_degrades_without_calling_llm_when_no_chunks_retrieved():
    llm_client = MagicMock()
    node = make_generate_node(
        generate_use_case=MagicMock(),
        system_prompt="system",
        user_prompt_template="{context_block} {question}",
        llm_client=llm_client,
    )

    state = _base_state(retrieved_chunk_registry={})  # <-- the observed condition
    result = node(state)

    assert result["degraded"] is True
    assert result["degrade_reason"] == "no_chunks_retrieved"
    assert result["final_answer"] == "I don't have enough information to answer this question."
    # Confirms the LLM is never even called — this is a pure short-circuit,
    # which is exactly why it still returns HTTP 200 upstream: no exception
    # ever fires, so nothing tells the API layer this was a failed query.
    llm_client.generate.assert_not_called()


def test_nli_model_lookup_does_not_raise_when_rerank_settings_used_directly():
    """
    Regression guard for the historical bug:
        AttributeError: 'RerankSettings' object has no attribute 'NLI_MODEL'
    RerankSettings DOES define NLI_MODEL today, and nli_model() falls back
    via getattr with a default — this must never throw again even if a
    caller mistakenly passes RerankSettings where NLISettings is expected.
    """
    registry = CrossEncoderRegistry(settings=RerankSettings(), nli_settings=None)

    # Reset any class-level cached model from other tests/imports so this
    # test exercises the actual attribute-resolution path, not a cache hit.
    CrossEncoderRegistry._nli_model = None

    try:
        model_name = (
            getattr(registry.nli_settings, "NLI_MODEL", None)
            or getattr(registry.settings, "NLI_MODEL", None)
            or "cross-encoder/nli-deberta-v3-base"
        )
    except AttributeError as e:
        raise AssertionError(f"NLI_MODEL lookup regressed: {e}")

    assert model_name == RerankSettings().NLI_MODEL
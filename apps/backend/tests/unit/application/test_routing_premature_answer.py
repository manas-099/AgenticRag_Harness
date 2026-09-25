"""
Reproduces the real root cause behind the persistent "no_chunks_retrieved"
symptom seen in manual testing: the decision LLM chose action="answer" on
the very first iteration, with retrieved_chunk_registry still empty (no
search had ever run). route_after_dispatch previously routed "answer"
straight to generate_node unconditionally, so generate_node correctly (but
unhelpfully) degraded with no_chunks_retrieved — even though the corpus had
real, relevant, already-ingested content that was simply never queried.

Fix: route_after_dispatch now refuses to send "answer" to generate_node
when nothing has been retrieved yet, and instead loops back to
agent_decide (subject to the normal iteration/budget/search-attempt caps)
so the agent gets another chance to actually search first.
"""

from rag_harness.application.agent.routing import make_route_after_dispatch
from rag_harness.config.settings import HarnessLoopSettings


def _base_state(**overrides):
    state = {
        "query": "how is agentic rag different from normal rag",
        "iteration": 1,
        "max_iterations": 8,
        "token_budget_used": 0,
        "max_token_budget": 12000,
        "max_validation_retries": 2,
        "validation_retries": 0,
        "search_attempts": [],
        "registry_size_history": [],
        "retrieved_chunk_registry": {},
        "scratchpad": [],
        "action_history": ["answer"],
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


def test_answer_with_no_retrieved_chunks_does_not_go_straight_to_generate():
    """
    THE FIX: this exact scenario (screenshot: Reasoning Iterations=1,
    Sources Used=0, Degraded: no_chunks_retrieved) must no longer route
    "answer" -> "generate" when the registry is empty.
    """
    route = make_route_after_dispatch(HarnessLoopSettings())
    state = _base_state(action_history=["answer"], retrieved_chunk_registry={})

    destination = route(state)

    assert destination != "generate"
    assert destination == "agent_decide"


def test_answer_with_no_retrieved_chunks_adds_a_scratchpad_note():
    route = make_route_after_dispatch(HarnessLoopSettings())
    state = _base_state(action_history=["answer"], retrieved_chunk_registry={})

    route(state)

    assert any("forcing a search" in note.lower() for note in state["scratchpad"])


def test_answer_with_chunks_already_retrieved_still_goes_straight_to_generate():
    """Unchanged behavior: a legitimate "answer" after real retrieval must
    still short-circuit straight to generate_node, not loop unnecessarily."""
    route = make_route_after_dispatch(HarnessLoopSettings())
    state = _base_state(
        action_history=["search_documents", "answer"],
        retrieved_chunk_registry={"c1": object()},
    )

    destination = route(state)

    assert destination == "generate"


def test_answer_with_no_chunks_still_degrades_once_iteration_cap_is_hit():
    """The guard must not create an infinite loop — normal loop-limit
    checks still apply even when we're forcing a re-decision."""
    route = make_route_after_dispatch(HarnessLoopSettings())
    state = _base_state(
        action_history=["answer"],
        retrieved_chunk_registry={},
        iteration=8,
        max_iterations=8,
    )

    destination = route(state)

    assert destination == "degrade"
    assert state["degrade_reason"] == "max_iterations_exceeded"


def test_answer_with_no_chunks_still_degrades_once_search_attempts_exhausted():
    settings = HarnessLoopSettings()
    route = make_route_after_dispatch(settings)
    state = _base_state(
        action_history=["answer"],
        retrieved_chunk_registry={},
        search_attempts=["q1", "q2", "q3", "q4"],  # == MAX_SEARCH_ATTEMPTS default
    )

    destination = route(state)

    assert destination == "degrade"
    assert state["degrade_reason"] == "max_search_attempts_exceeded"
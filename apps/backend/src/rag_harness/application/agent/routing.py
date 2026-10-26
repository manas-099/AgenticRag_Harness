"""
Loop control — the agent's decision logic between nodes, including the
stuck-loop detector (registry size unchanged over N searches -> force degrade).
"""

from __future__ import annotations

import logging

from rag_harness.config.settings import HarnessLoopSettings
from rag_harness.domain.models import AgentState

logger = logging.getLogger("rag_harness.agent_routing")


def _is_stuck_in_loop(state: AgentState, window: int) -> bool:
    history = state["registry_size_history"]
    if len(history) < window:
        return False
    return len(set(history[-window:])) == 1


def make_route_after_dispatch(settings: HarnessLoopSettings):
    def route_after_dispatch(state: AgentState) -> str:
        last_action = state["action_history"][-1] if state["action_history"] else None

        if last_action == "answer":
            if not state["retrieved_chunk_registry"]:
                # Defensive guard: never let "answer" through to generate_node
                # with nothing retrieved. This happens when the decision LLM
                # (especially a smaller/local model) jumps straight to
                # "answer" on the very first turn without ever searching —
                # generate_node would then correctly but unhelpfully degrade
                # with no_chunks_retrieved despite the corpus having real,
                # relevant content that was simply never queried.
                # Force at least one real search attempt before allowing
                # a straight-to-answer exit, unless loop limits are already
                # exhausted (checked below, same as any other path).
                state["scratchpad"] = [
                    "System: 'answer' was chosen with no retrieved chunks — "
                    "forcing a search before allowing an answer."
                ]
            else:
                return "generate"

        has_chunks = bool(state.get("retrieved_chunk_registry"))

        if state["iteration"] >= state["max_iterations"]:
            if has_chunks:
                logger.info("Max iterations reached with retrieved chunks — proceeding to generate final answer")
                return "generate"
            state["degrade_reason"] = "max_iterations_exceeded"
            return "degrade"

        if state["token_budget_used"] >= state["max_token_budget"]:
            if has_chunks:
                logger.info("Token budget exceeded with retrieved chunks — proceeding to generate final answer")
                return "generate"
            state["degrade_reason"] = "token_budget_exceeded"
            return "degrade"

        if len(state["search_attempts"]) >= settings.MAX_SEARCH_ATTEMPTS:
            if has_chunks:
                logger.info("Max search attempts reached with retrieved chunks — proceeding to generate final answer")
                return "generate"
            state["degrade_reason"] = "max_search_attempts_exceeded"
            return "degrade"

        if _is_stuck_in_loop(state, settings.STUCK_LOOP_WINDOW):
            if has_chunks:
                logger.info("Stuck loop detected with retrieved chunks — proceeding to generate final answer")
                return "generate"
            state["degrade_reason"] = "stuck_loop_no_new_chunks"
            return "degrade"

        return "agent_decide"
    return route_after_dispatch


def route_after_verify(state: AgentState) -> str:
    # If generate_node already marked degraded (e.g. LLM timeout, no chunks),
    # don't loop back — go straight to degrade.
    if state.get("degraded"):
        return "degrade"
    vr = state.get("verification_result")
    if state.get("final_answer") and vr and vr.passed:
        return "end"
    if state["validation_retries"] >= state["max_validation_retries"]:
        state["degrade_reason"] = "max_validation_retries_exceeded"
        return "degrade"
    return "generate"
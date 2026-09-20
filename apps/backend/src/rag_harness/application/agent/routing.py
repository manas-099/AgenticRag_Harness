"""
Loop control — the agent's decision logic between nodes, including the
stuck-loop detector (registry size unchanged over N searches -> force degrade).
"""

from __future__ import annotations

from rag_harness.config.settings import HarnessLoopSettings
from rag_harness.domain.models import AgentState


def _is_stuck_in_loop(state: AgentState, window: int) -> bool:
    history = state["registry_size_history"]
    if len(history) < window:
        return False
    return len(set(history[-window:])) == 1


def make_route_after_dispatch(settings: HarnessLoopSettings):
    def route_after_dispatch(state: AgentState) -> str:
        last_action = state["action_history"][-1] if state["action_history"] else None
        if last_action == "answer":
            return "generate"

        if state["iteration"] >= state["max_iterations"]:
            state["degrade_reason"] = "max_iterations_exceeded"
            return "degrade"
        if state["token_budget_used"] >= state["max_token_budget"]:
            state["degrade_reason"] = "token_budget_exceeded"
            return "degrade"
        if len(state["search_attempts"]) >= settings.MAX_SEARCH_ATTEMPTS:
            state["degrade_reason"] = "max_search_attempts_exceeded"
            return "degrade"
        if _is_stuck_in_loop(state, settings.STUCK_LOOP_WINDOW):
            state["degrade_reason"] = "stuck_loop_no_new_chunks"
            return "degrade"

        return "agent_decide"
    return route_after_dispatch


def route_after_verify(state: AgentState) -> str:
    vr = state.get("verification_result")
    if state.get("final_answer") and vr and vr.passed:
        return "end"
    if state["validation_retries"] >= state["max_validation_retries"]:
        state["degrade_reason"] = "max_validation_retries_exceeded"
        return "degrade"
    return "generate"
"""
Builds and compiles the LangGraph state machine wiring all agent nodes
together, and the top-level entry point that runs a query through it.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from rag_harness.application.agent.decision import AgentDecisionEngine
from rag_harness.application.agent.nodes import (
    degrade_node,
    make_agent_decide_node,
    make_generate_node,
    make_tool_dispatch_node,
    make_verify_node,
)
from rag_harness.application.agent.routing import make_route_after_dispatch, route_after_verify
from rag_harness.application.agent.tools import AgentTools
from rag_harness.application.generation.generate_answer_use_case import GenerateAnswerUseCase
from rag_harness.application.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from rag_harness.application.verification.verify_answer_use_case import VerifyAnswerUseCase
from rag_harness.config.settings import HarnessLoopSettings
from rag_harness.domain.models import AgentState
from rag_harness.domain.ports import LLMPort

logger = logging.getLogger("rag_harness.agent_graph")


def build_harness_graph(
    agent_tools: AgentTools,
    decision_engine: AgentDecisionEngine,
    generate_use_case: GenerateAnswerUseCase,
    verify_use_case: VerifyAnswerUseCase,
    llm_client: LLMPort,
    loop_settings: HarnessLoopSettings,
):
    graph = StateGraph(AgentState)
    graph.add_node("agent_decide", make_agent_decide_node(decision_engine))
    graph.add_node("tool_dispatch", make_tool_dispatch_node(agent_tools))
    graph.add_node("generate", make_generate_node(generate_use_case, RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE, llm_client))
    graph.add_node("verify", make_verify_node(verify_use_case))
    graph.add_node("degrade", degrade_node)

    graph.set_entry_point("agent_decide")
    graph.add_edge("agent_decide", "tool_dispatch")
    graph.add_conditional_edges("tool_dispatch", make_route_after_dispatch(loop_settings), {
        "agent_decide": "agent_decide", "generate": "generate", "degrade": "degrade",
    })
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {
        "generate": "generate", "degrade": "degrade", "end": END,
    })
    graph.add_edge("degrade", END)

    logger.info("ReAct agent harness graph compiled")
    return graph.compile()


class RunAgenticQueryUseCase:
    def __init__(self, compiled_graph, loop_settings: HarnessLoopSettings):
        self.graph = compiled_graph
        self.loop_settings = loop_settings

    def execute(self, query: str) -> dict:
        logger.info(f"=== Agentic query: '{query}' ===")
        initial_state: AgentState = {
            "query": query,
            "iteration": 0,
            "max_iterations": self.loop_settings.MAX_ITERATIONS,
            "token_budget_used": 0,
            "max_token_budget": self.loop_settings.MAX_TOKEN_BUDGET,
            "max_validation_retries": self.loop_settings.MAX_VALIDATION_RETRIES,
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
        try:
            result = self.graph.invoke(initial_state)
            logger.info(f"=== Agentic query complete. Degraded={result.get('degraded', False)} ===")
            return {
                "answer": result["final_answer"],
                "sources_used": result.get("sources_used", []),
                "degraded": result.get("degraded", False),
                "degrade_reason": result.get("degrade_reason"),
                "iterations_used": result["iteration"],
                "validation_retries": result["validation_retries"],
                "action_history": result.get("action_history", []),
            }
        except Exception as e:
            logger.critical(f"Agentic query failed unexpectedly: {e}")
            return {"answer": "An unexpected error occurred.", "sources_used": [], "degraded": True, "degrade_reason": str(e)}
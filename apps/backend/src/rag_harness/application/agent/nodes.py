"""
LangGraph node functions: agent_decide -> tool_dispatch -> generate -> verify
-> (loop or degrade). Each node is a plain (AgentState) -> AgentState function.
"""

from __future__ import annotations

import logging

from rag_harness.application.agent.decision import AgentDecisionEngine
from rag_harness.application.agent.tools import AgentTools
from rag_harness.application.generation.generate_answer_use_case import GenerateAnswerUseCase
from rag_harness.application.verification.verify_answer_use_case import VerifyAnswerUseCase
from rag_harness.domain.exceptions import HarnessError
from rag_harness.domain.models import AgentState

logger = logging.getLogger("rag_harness.agent_nodes")


def make_agent_decide_node(decision_engine: AgentDecisionEngine):
    def agent_decide_node(state: AgentState) -> AgentState:
        logger.info(f"[agent_decide_node] iteration={state['iteration']}")
        decision = decision_engine.decide(state)
        state["scratchpad"] = [f"Thought: {decision.thought} | Action: {decision.action}"]
        state["action_history"] = [decision.action]
        state["pending_decision"] = decision.model_dump()
        state["iteration"] += 1
        return state
    return agent_decide_node


def make_tool_dispatch_node(agent_tools: AgentTools):
    def tool_dispatch_node(state: AgentState) -> AgentState:
        decision = state["pending_decision"]
        action = decision["action"]
        logger.info(f"[tool_dispatch_node] executing action={action}")

        try:
            if action == "search_documents":
                results = agent_tools.search_documents(decision.get("query") or state["query"])
                for rc in results:
                    state["retrieved_chunk_registry"][rc.chunk.chunk_id] = rc.chunk
                state["search_attempts"] = [decision.get("query") or state["query"]]

            elif action == "search_documents_reformulated":
                results = agent_tools.search_documents_reformulated(
                    decision.get("query") or state["query"], decision.get("failed_attempt_reason") or "unspecified"
                )
                for rc in results:
                    state["retrieved_chunk_registry"][rc.chunk.chunk_id] = rc.chunk
                state["search_attempts"] = [decision.get("query") or state["query"]]

            elif action == "get_page":
                if decision.get("doc_id") and decision.get("page_num") is not None:
                    text = agent_tools.get_page(decision["doc_id"], decision["page_num"])
                    state["scratchpad"] = [f"get_page result: {text[:300]}"]

            elif action == "compare_chunks":
                if decision.get("chunk_id_a") and decision.get("chunk_id_b"):
                    comparison = agent_tools.compare_chunks(state["retrieved_chunk_registry"], decision["chunk_id_a"], decision["chunk_id_b"])
                    state["scratchpad"] = [f"compare_chunks result: {comparison[:300]}"]

            elif action == "check_document_freshness":
                if decision.get("doc_id"):
                    version = agent_tools.check_document_freshness(decision["doc_id"])
                    state["scratchpad"] = [f"Document {decision['doc_id']} version: {version}"]

        except HarnessError as e:
            logger.error(f"Tool execution failed: {e}")
            state["scratchpad"] = [f"Tool error: {e}"]

        state["registry_size_history"] = [frozenset(state["retrieved_chunk_registry"].keys())]
        return state
    return tool_dispatch_node


def make_generate_node(generate_use_case: GenerateAnswerUseCase, system_prompt: str, user_prompt_template: str, llm_client):
    def generate_node(state: AgentState) -> AgentState:
        logger.info(f"[generate_node] iteration={state['iteration']}")
        context_blocks = [
            f"[Doc: {c.metadata.doc_id}, Page: {c.metadata.page_num}, chunk_id: {cid}]\n{c.raw_text}"
            for cid, c in state["retrieved_chunk_registry"].items()
        ]
        context_str = "\n\n---\n\n".join(context_blocks)

        if not context_blocks:
            state["degraded"] = True
            state["degrade_reason"] = "no_chunks_retrieved"
            state["final_answer"] = "I don't have enough information to answer this question."
            return state

        retry_feedback = ""
        vr = state.get("verification_result")
        if vr and not vr.passed:
            retry_feedback = f"\n\nIMPORTANT — your previous answer had an issue: {vr.feedback_for_retry}"

        user_prompt = user_prompt_template.format(context_block=context_str, question=state["query"]) + retry_feedback

        try:
            answer = llm_client.generate(system_prompt, user_prompt, max_tokens=800, temperature=0.0)
            state["draft_answer"] = answer
            state["token_budget_used"] += len(user_prompt.split())
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["degraded"] = True
            state["degrade_reason"] = f"llm_generation_failed: {e}"
            state["final_answer"] = "I couldn't generate an answer due to a system error."
        return state
    return generate_node


def make_verify_node(verify_use_case: VerifyAnswerUseCase):
    def verify_node(state: AgentState) -> AgentState:
        logger.info(f"[verify_node] validation_retries={state['validation_retries']}")
        result = verify_use_case.execute(state["draft_answer"], state["retrieved_chunk_registry"])
        state["verification_result"] = result

        if result.passed:
            state["final_answer"] = state["draft_answer"]
            state["sources_used"] = list(state["retrieved_chunk_registry"].keys())
            logger.info("Verification PASSED")
        else:
            state["validation_retries"] += 1
            state["scratchpad"] = [f"Verification failed: {result.feedback_for_retry}"]
        return state
    return verify_node


def degrade_node(state: AgentState) -> AgentState:
    logger.warning(f"[degrade_node] reason={state.get('degrade_reason')}")
    state["degraded"] = True

    if not state.get("final_answer"):
        vr = state.get("verification_result")
        if vr and vr.groundedness_results:
            grounded_texts = [r.claim.claim_text for r in vr.groundedness_results if r.is_grounded]
            if grounded_texts:
                state["final_answer"] = " ".join(grounded_texts) + "\n\n(Note: some claims could not be fully verified and were removed.)"
        if not state.get("final_answer"):
            state["final_answer"] = "I couldn't produce a fully verified answer to this question. Please rephrase, or consult the source documents directly."
    return state
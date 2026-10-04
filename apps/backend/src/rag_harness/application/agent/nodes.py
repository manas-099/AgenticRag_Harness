"""
LangGraph node functions: agent_decide -> tool_dispatch -> generate -> verify
-> (loop or degrade). Each node is a plain (AgentState) -> AgentState function.

FIX APPLIED:
  tool_dispatch_node now writes a rich scratchpad entry after EVERY action,
  including search_documents — previously search results were never summarized
  into the scratchpad so the agent had no memory of what it found and repeated
  the same search every iteration.
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
        # operator.add appends — return only the NEW entry, not the full list
        return {
            "scratchpad": [f"Thought: {decision.thought} | Action: {decision.action}"],
            "action_history": [decision.action],
            "pending_decision": decision.model_dump(),
            "iteration": state["iteration"] + 1,
        }
    return agent_decide_node


def make_tool_dispatch_node(agent_tools: AgentTools):
    def tool_dispatch_node(state: AgentState) -> AgentState:
        decision = state["pending_decision"]
        action = decision["action"]
        logger.info(f"[tool_dispatch_node] executing action={action}")

        new_scratchpad_entries = []
        new_search_attempts = []
        new_registry_entries = dict(state["retrieved_chunk_registry"])

        try:
            if action == "search_documents":
                query = decision.get("query") or state["query"]
                results = agent_tools.search_documents(query)
                new_chunks = {rc.chunk.chunk_id: rc.chunk for rc in results}
                new_registry_entries.update(new_chunks)
                new_search_attempts.append(query)

                # KEY FIX: write what was found into scratchpad so agent
                # knows these chunks exist and what they contain
                if results:
                    summaries = []
                    for rc in results[:5]:  # top 5 summaries
                        preview = rc.chunk.raw_text[:120].replace("\n", " ")
                        summaries.append(f"  - [{rc.chunk.chunk_id[:8]}] score={rc.score:.2f}: {preview}...")
                    new_scratchpad_entries.append(
                        f"search_documents('{query}') → found {len(results)} chunks:\n" +
                        "\n".join(summaries)
                    )
                else:
                    new_scratchpad_entries.append(
                        f"search_documents('{query}') → NO chunks found. Try a different query."
                    )

            elif action == "search_documents_reformulated":
                query = decision.get("query") or state["query"]
                reason = decision.get("failed_attempt_reason") or "unspecified"
                results = agent_tools.search_documents_reformulated(query, reason)
                new_chunks = {rc.chunk.chunk_id: rc.chunk for rc in results}
                new_registry_entries.update(new_chunks)
                new_search_attempts.append(query)

                if results:
                    summaries = []
                    for rc in results[:5]:
                        preview = rc.chunk.raw_text[:120].replace("\n", " ")
                        summaries.append(f"  - [{rc.chunk.chunk_id[:8]}] score={rc.score:.2f}: {preview}...")
                    new_scratchpad_entries.append(
                        f"search_documents_reformulated('{query}') → found {len(results)} chunks:\n" +
                        "\n".join(summaries)
                    )
                else:
                    new_scratchpad_entries.append(
                        f"search_documents_reformulated('{query}') → NO chunks found."
                    )

            elif action == "get_page":
                if decision.get("doc_id") and decision.get("page_num") is not None:
                    text = agent_tools.get_page(decision["doc_id"], decision["page_num"])
                    new_scratchpad_entries.append(f"get_page result: {text[:300]}")

            elif action == "compare_chunks":
                if decision.get("chunk_id_a") and decision.get("chunk_id_b"):
                    comparison = agent_tools.compare_chunks(
                        state["retrieved_chunk_registry"],
                        decision["chunk_id_a"],
                        decision["chunk_id_b"]
                    )
                    new_scratchpad_entries.append(f"compare_chunks result: {comparison[:300]}")

            elif action == "check_document_freshness":
                if decision.get("doc_id"):
                    version = agent_tools.check_document_freshness(decision["doc_id"])
                    new_scratchpad_entries.append(
                        f"Document {decision['doc_id']} version: {version}"
                    )

        except HarnessError as e:
            logger.error(f"Tool execution failed: {e}")
            new_scratchpad_entries.append(f"Tool error: {e}")

        return {
            "retrieved_chunk_registry": new_registry_entries,
            "scratchpad": new_scratchpad_entries,
            "search_attempts": new_search_attempts,
            "registry_size_history": [frozenset(new_registry_entries.keys())],
        }
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
            return {
                "degraded": True,
                "degrade_reason": "no_chunks_retrieved",
                "final_answer": "I don't have enough information to answer this question.",
            }

        retry_feedback = ""
        vr = state.get("verification_result")
        if vr and not vr.passed:
            retry_feedback = (
                f"\n\nIMPORTANT — your previous answer had an issue: {vr.feedback_for_retry}"
            )

        user_prompt = (
            user_prompt_template.format(context_block=context_str, question=state["query"])
            + retry_feedback
        )

        try:
            answer = llm_client.generate(
                system_prompt, user_prompt, max_tokens=800, temperature=0.0
            )
            return {
                "draft_answer": answer,
                "token_budget_used": state["token_budget_used"] + len(user_prompt.split()),
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "degraded": True,
                "degrade_reason": f"llm_generation_failed: {e}",
                "final_answer": "I couldn't generate an answer due to a system error.",
            }
    return generate_node


def make_verify_node(verify_use_case: VerifyAnswerUseCase):
    def verify_node(state: AgentState) -> AgentState:
        logger.info(f"[verify_node] validation_retries={state['validation_retries']}")
        result = verify_use_case.execute(
            state["draft_answer"], state["retrieved_chunk_registry"]
        )

        if result.passed:
            return {
                "verification_result": result,
                "final_answer": state["draft_answer"],
                "sources_used": list(state["retrieved_chunk_registry"].keys()),
            }
        else:
            return {
                "verification_result": result,
                "validation_retries": state["validation_retries"] + 1,
                "scratchpad": [f"Verification failed: {result.feedback_for_retry}"],
            }
    return make_verify_node


def degrade_node(state: AgentState) -> AgentState:
    logger.warning(f"[degrade_node] reason={state.get('degrade_reason')}")

    if not state.get("final_answer"):
        vr = state.get("verification_result")
        if vr and vr.groundedness_results:
            grounded_texts = [
                r.claim.claim_text
                for r in vr.groundedness_results
                if r.is_grounded
            ]
            if grounded_texts:
                return {
                    "degraded": True,
                    "final_answer": (
                        " ".join(grounded_texts) +
                        "\n\n(Note: some claims could not be fully verified and were removed.)"
                    ),
                }
        return {
            "degraded": True,
            "final_answer": (
                "I couldn't produce a fully verified answer to this question. "
                "Please rephrase, or consult the source documents directly."
            ),
        }
    return {"degraded": True}
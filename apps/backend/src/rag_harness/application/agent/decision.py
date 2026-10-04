"""
The structured "Thought -> Action" step of the ReAct loop.

FIX APPLIED:
  User prompt now includes a CHUNK PREVIEW section — the first 200 chars of
  each retrieved chunk. Previously the agent only saw chunk_ids (UUIDs) with
  no content, so it had no basis for deciding "I have enough to answer" and
  kept searching. Now it sees actual content and can make an informed decision.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel

from rag_harness.application.generation.prompts import AGENT_SYSTEM_PROMPT
from rag_harness.domain.models import AgentState
from rag_harness.domain.ports import LLMPort
from rag_harness.infrastructure.llm.xgrammar import XGrammar

logger = logging.getLogger("rag_harness.agent_decision")


class AgentDecision(BaseModel):
    thought: str
    action: Literal[
        "search_documents",
        "search_documents_reformulated",
        "get_page",
        "compare_chunks",
        "check_document_freshness",
        "answer",
    ]
    query: Optional[str] = None
    failed_attempt_reason: Optional[str] = None
    doc_id: Optional[str] = None
    page_num: Optional[int] = None
    chunk_id_a: Optional[str] = None
    chunk_id_b: Optional[str] = None


class AgentDecisionEngine:
    def __init__(self, llm_client: LLMPort):
        self.llm_client = llm_client
        self.grammar = XGrammar(AgentDecision, name="agent_decision")

    def decide(self, state: AgentState) -> AgentDecision:
        # Last 10 scratchpad entries so agent remembers what it found
        scratchpad_str = "\n".join(state["scratchpad"][-10:]) or "(nothing yet)"

        # KEY FIX: show chunk content previews, not just UUIDs.
        # Agent needs to READ what was retrieved to decide if it's enough.
        registry = state["retrieved_chunk_registry"]
        if registry:
            chunk_previews = []
            for cid, chunk in list(registry.items())[:8]:  # max 8 previews
                preview = chunk.raw_text[:200].replace("\n", " ")
                chunk_previews.append(f"  [{cid[:8]}] {preview}...")
            registry_summary = (
                f"{len(registry)} chunks retrieved:\n" + "\n".join(chunk_previews)
            )
        else:
            registry_summary = "(empty — no chunks retrieved yet)"

        # Deduplicated past queries so agent sees what it already tried
        seen_queries = list(dict.fromkeys(state["search_attempts"]))
        past_queries_str = (
            "\n".join(f"  - {q}" for q in seen_queries) or "  (none yet)"
        )

        user_content = (
            f"Question: {state['query']}\n\n"
            f"Scratchpad (your memory of what you have done so far):\n"
            f"{scratchpad_str}\n\n"
            f"Retrieved chunks so far:\n{registry_summary}\n\n"
            f"Past search queries already tried:\n{past_queries_str}\n\n"
            f"Iterations used: {state['iteration']} / {state['max_iterations']}\n\n"
            f"What is your next action? "
            f"If you have retrieved enough relevant content above to answer the question, "
            f"choose 'answer'. If not, search with a DIFFERENT query than the ones above."
        )

        try:
            response = self.llm_client.generate(
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_content=user_content,
                response_format=self.grammar.build_response_format(),
                max_tokens=400,
                temperature=0.0,
            )
            clean_resp = response.strip()
            if clean_resp.startswith("```"):
                clean_resp = re.sub(r"^```(?:json)?\s*", "", clean_resp)
                clean_resp = re.sub(r"\s*```$", "", clean_resp)

            try:
                decision = AgentDecision.model_validate_json(clean_resp)
            except Exception:
                match = re.search(r"(\{.*\})", clean_resp, re.DOTALL)
                if match:
                    decision = AgentDecision.model_validate_json(match.group(1))
                else:
                    action = "search_documents"
                    if "answer" in clean_resp.lower() and "search" not in clean_resp.lower():
                        action = "answer"
                    decision = AgentDecision(
                        thought=clean_resp[:200],
                        action=action,
                        query=state.get("query"),
                    )

            # Safety guard: don't repeat an identical query
            # Force reformulation if the agent picked the same query again
            if (
                decision.action == "search_documents"
                and decision.query
                and decision.query in state["search_attempts"]
            ):
                logger.warning(
                    f"Agent tried to repeat query '{decision.query}' — "
                    f"forcing search_documents_reformulated"
                )
                decision.action = "search_documents_reformulated"
                decision.failed_attempt_reason = (
                    "Previous search with this exact query already ran — "
                    "must use different keywords"
                )

            logger.info(
                f"Agent decision: action={decision.action}, "
                f"thought='{decision.thought[:80]}...'"
            )
            return decision

        except Exception as e:
            logger.error(
                f"Agent decision failed ({e}) — defaulting to search or answer"
            )
            fallback_action = (
                "search_documents" if not state.get("retrieved_chunk_registry") else "answer"
            )
            return AgentDecision(
                thought=f"Decision engine error: {e}",
                action=fallback_action,
                query=state.get("query"),
            )
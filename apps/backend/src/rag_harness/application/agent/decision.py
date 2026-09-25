"""
The structured "Thought -> Action" step of the ReAct loop. The LLM chooses
ONE action via XGrammar-enforced structured output — never free text we'd
have to guess-parse.
"""

from __future__ import annotations

import logging
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
        "search_documents", "search_documents_reformulated", "get_page",
        "compare_chunks", "check_document_freshness", "answer",
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
        scratchpad_str = "\n".join(state["scratchpad"][-10:]) or "(nothing yet)"
        registry_summary = ", ".join(state["retrieved_chunk_registry"].keys()) or "(empty)"

        user_content = (
            f"Question: {state['query']}\n\n"
            f"Scratchpad so far:\n{scratchpad_str}\n\n"
            f"Retrieved chunk_ids so far: {registry_summary}\n\n"
            f"Past search queries tried: {state['search_attempts']}\n\n"
            f"What is your next action?"
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
                import re
                clean_resp = re.sub(r"^```(?:json)?\s*", "", clean_resp)
                clean_resp = re.sub(r"\s*```$", "", clean_resp)
            try:
                decision = AgentDecision.model_validate_json(clean_resp)
            except Exception:
                import re
                match = re.search(r"(\{.*\})", clean_resp, re.DOTALL)
                if match:
                    decision = AgentDecision.model_validate_json(match.group(1))
                else:
                    action = "search_documents"
                    if "action: answer" in clean_resp.lower() or ("answer" in clean_resp.lower() and "search" not in clean_resp.lower()):
                        action = "answer"
                    decision = AgentDecision(thought=clean_resp[:200], action=action, query=state.get("query"))
            logger.info(f"Agent decision: action={decision.action}, thought='{decision.thought[:80]}...'")
            return decision
        except Exception as e:
            logger.error(f"Agent decision failed ({e}) — defaulting to 'search_documents' or 'answer'")
            fallback_action = "search_documents" if not state.get("retrieved_chunk_registry") else "answer"
            return AgentDecision(thought=f"Decision engine error: {e}", action=fallback_action, query=state.get("query"))
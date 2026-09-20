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
            decision = AgentDecision.model_validate_json(response)
            logger.info(f"Agent decision: action={decision.action}, thought='{decision.thought[:80]}...'")
            return decision
        except Exception as e:
            logger.error(f"Agent decision failed ({e}) — defaulting to 'answer' with whatever we have")
            return AgentDecision(thought=f"Decision engine error: {e}", action="answer")
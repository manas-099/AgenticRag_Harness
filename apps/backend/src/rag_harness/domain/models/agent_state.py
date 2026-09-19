"""
The state object the ReAct agent loop carries through every LangGraph node.
This is a plain TypedDict (LangGraph's required shape for graph state), kept
in domain/ since it represents pure agent state, not any infrastructure
concern — no Qdrant/LLM client objects live inside it, only data.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from rag_harness.domain.models.chunk import Chunk
from rag_harness.domain.models.verification import VerificationResult


class AgentState(TypedDict):
    query: str
    iteration: int
    max_iterations: int
    token_budget_used: int
    max_token_budget: int
    max_validation_retries: int
    validation_retries: int

    search_attempts: Annotated[list[str], operator.add]
    registry_size_history: Annotated[list[int], operator.add]
    retrieved_chunk_registry: dict[str, Chunk]
    scratchpad: Annotated[list[str], operator.add]
    action_history: Annotated[list[str], operator.add]

    pending_decision: Optional[dict]  # serialized AgentDecision, set by agent_decide_node

    draft_answer: Optional[str]
    verification_result: Optional[VerificationResult]
    final_answer: Optional[str]
    sources_used: list[str]

    degraded: bool
    degrade_reason: Optional[str]
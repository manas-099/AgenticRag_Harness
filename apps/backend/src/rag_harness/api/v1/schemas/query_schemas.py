# =============================================================================
# FILE: src/rag_harness/api/v1/schemas/query_schemas.py
# TARGET PATH IN PROJECT: apps/backend/src/rag_harness/api/v1/schemas/query_schemas.py
# REPLACES: existing query_schemas.py (full replacement)
#
# WHAT CHANGED:
#   AgenticQueryResponse now includes:
#     - validation_retries (int): how many verify→regenerate loops happened
#     - token_budget_used (int): approx tokens consumed by this run
#     - action_history (list[ActionStep]): each step now carries BOTH the
#       raw action string AND the agent's thought, so the frontend debug panel
#       can show "Thought: X → Action: Y" per step instead of just action names.
#
# WHY:
#   The frontend Debug Panel needs to show per-step agent reasoning (thought)
#   without any LLM call — the agent_decide_node already produces thoughts,
#   they just weren't being serialized into the response. This change threads
#   them through.
#
# WHERE IT CONNECTS:
#   - graph_builder.py RunAgenticQueryUseCase.execute() returns action_history
#     as list[str] — that stays unchanged. The router (query.py) now builds
#     ActionStep objects from the AgentState before returning.
#   - Frontend: types/domain.ts ActionStep mirrors this shape.
# =============================================================================

from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    doc_version_key: str = "v1"


class QueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    is_insufficient: bool
    from_cache: bool


class ActionStep(BaseModel):
    """One step in the agent's ReAct loop — action chosen + the thought behind it."""
    action: str          # e.g. "search_documents", "answer"
    thought: str = ""    # agent's reasoning text (from AgentDecision.thought)
    iteration: int = 0   # which iteration number this was


class AgenticQueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    degraded: bool
    degrade_reason: Optional[str] = None
    iterations_used: Optional[int] = None
    validation_retries: int = 0
    token_budget_used: int = 0
    # Each entry is now a structured ActionStep (thought + action + iteration),
    # not just a raw action string.
    action_steps: list[ActionStep] = []
    # Keep the old flat list too for backwards compat with existing frontend code
    action_history: list[str] = []

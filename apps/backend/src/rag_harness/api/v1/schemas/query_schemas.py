# =============================================================================
# FILE: src/rag_harness/api/v1/schemas/query_schemas.py
# TARGET PATH: apps/backend/src/rag_harness/api/v1/schemas/query_schemas.py
# REPLACES: previous query_schemas.py
#
# WHAT CHANGED:
#   - from_cache: bool added to AgenticQueryResponse (default False)
#   - ActionStep model added (thought + action + iteration)
#   - validation_retries, token_budget_used added
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
    action: str
    thought: str = ""
    iteration: int = 0


class AgenticQueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    degraded: bool
    degrade_reason: Optional[str] = None
    iterations_used: Optional[int] = None
    validation_retries: int = 0
    token_budget_used: int = 0
    action_steps: list[ActionStep] = []
    action_history: list[str] = []
    from_cache: bool = False   # ← NEW: True when served from cache

# =============================================================================
# FILE: src/rag_harness/api/v1/routers/query.py
# TARGET PATH IN PROJECT: apps/backend/src/rag_harness/api/v1/routers/query.py
# REPLACES: existing query.py (full replacement)
#
# WHAT CHANGED:
#   1. query_agentic() now maps the raw result dict into the richer
#      AgenticQueryResponse: builds ActionStep list from action_history +
#      scratchpad (thought text), passes validation_retries + token_budget_used.
#
#   2. NEW endpoint: GET /v1/query/agentic/last_state
#      Returns the last completed AgentState snapshot as a plain dict.
#      The frontend Debug Panel polls this (or reads it from the agentic
#      response) to show live iteration/token/degrade info WITHOUT needing
#      SSE or websockets. A module-level variable `_last_agent_state` is set
#      by query_agentic() on each call — safe for single-worker dev use.
#      For multi-worker production: replace with Redis/memcache.
#
#   3. GET /v1/query/graph is kept unchanged.
#
# WHY:
#   The Debug Panel needs: per-step thoughts, iteration count, token budget,
#   degrade reason. All this data is already in the AgentState after execute()
#   — it just wasn't being returned. No application layer changes needed.
#
# WHERE IT CONNECTS:
#   - schemas/query_schemas.py: AgenticQueryResponse, ActionStep
#   - application/rag_pipeline.py: pipeline.answer_agentic() returns dict
#     with keys: answer, sources_used, degraded, degrade_reason,
#     iterations_used, validation_retries, action_history, (scratchpad added below)
#   - graph_builder.py: RunAgenticQueryUseCase.execute() — we add scratchpad
#     to its return dict so thoughts are accessible here.
# =============================================================================

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from rag_harness.api.dependencies import get_rag_pipeline
from rag_harness.api.v1.schemas import AgenticQueryResponse, QueryRequest, QueryResponse
from rag_harness.api.v1.schemas.query_schemas import ActionStep
from rag_harness.application.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])

# Module-level snapshot of the last completed agentic run's state.
# Used by GET /v1/query/agentic/last_state for the Debug Panel.
_last_agent_debug: dict = {}


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    result = pipeline.answer(request.question, doc_version_key=request.doc_version_key)
    return QueryResponse(
        answer=result.answer,
        sources_used=result.sources_used,
        is_insufficient=result.is_insufficient,
        from_cache=result.from_cache,
    )


@router.post("/agentic", response_model=AgenticQueryResponse)
def query_agentic(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    global _last_agent_debug
    result = pipeline.answer_agentic(request.question)

    # Build structured ActionStep list.
    # result["action_history"] is list[str] of action names.
    # result["scratchpad"] (if present) is list[str] like
    #   ["Thought: X | Action: Y", ...] — one entry per agent_decide_node call.
    raw_actions: list[str] = result.get("action_history", [])
    raw_scratchpad: list[str] = result.get("scratchpad", [])

    action_steps: list[ActionStep] = []
    for i, action in enumerate(raw_actions):
        thought = ""
        # scratchpad entries are "Thought: ... | Action: ..." strings
        if i < len(raw_scratchpad):
            entry = raw_scratchpad[i]
            if "Thought:" in entry:
                thought = entry.split("Thought:")[-1].split("|")[0].strip()
        action_steps.append(ActionStep(action=action, thought=thought, iteration=i + 1))

    # Save debug snapshot for the /last_state endpoint
    _last_agent_debug = {
        "query": request.question,
        "degraded": result.get("degraded", False),
        "degrade_reason": result.get("degrade_reason"),
        "iterations_used": result.get("iterations_used", 0),
        "validation_retries": result.get("validation_retries", 0),
        "token_budget_used": result.get("token_budget_used", 0),
        "sources_count": len(result.get("sources_used", [])),
        "action_steps": [s.model_dump() for s in action_steps],
    }

    return AgenticQueryResponse(
        answer=result["answer"],
        sources_used=result.get("sources_used", []),
        degraded=result.get("degraded", False),
        degrade_reason=result.get("degrade_reason"),
        iterations_used=result.get("iterations_used"),
        validation_retries=result.get("validation_retries", 0),
        token_budget_used=result.get("token_budget_used", 0),
        action_steps=action_steps,
        action_history=raw_actions,  # keep flat list for backwards compat
    )


@router.get("/agentic/last_state")
def get_last_agent_state():
    """Returns a debug snapshot of the most recently completed agentic run.
    Used by the frontend Debug Panel. No LLM call; just reads the cached dict.
    Safe for single-worker dev. For multi-worker prod: use Redis instead."""
    return JSONResponse(content=_last_agent_debug)


@router.get("/graph")
def query_graph(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """Mermaid source for the compiled agent graph."""
    compiled = pipeline.agentic_use_case.graph
    try:
        mermaid_src = compiled.get_graph().draw_mermaid()
    except Exception as e:
        mermaid_src = f"%% could not draw graph: {e}"
    return {"mermaid": mermaid_src}

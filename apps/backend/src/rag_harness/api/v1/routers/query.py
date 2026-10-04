# =============================================================================
# FILE: src/rag_harness/api/v1/routers/query.py
# TARGET PATH: apps/backend/src/rag_harness/api/v1/routers/query.py
# REPLACES: previous query.py
#
# WHAT CHANGED:
#   Agentic endpoint now checks the query cache BEFORE running the full
#   LangGraph agent, and writes to cache AFTER a successful (non-degraded) run.
#   Cache key: sha256(question + "::agentic::v1") — same pattern as standard.
#   If cache hit: response includes from_cache=True flag for the frontend.
#   Also returns from_cache in AgenticQueryResponse (new field added below).
#
# WHY CACHE WASN'T WORKING:
#   answer_agentic() in rag_pipeline.py calls agentic_use_case.execute()
#   which goes straight into the LangGraph graph — it never touches the
#   InMemoryCache. The standard pipeline hits cache in GenerateAnswerUseCase.
#   Fix: wrap the agentic call here at the router level using the same cache
#   instance (accessed via pipeline.cache).
# =============================================================================

import hashlib
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from rag_harness.api.dependencies import get_rag_pipeline
from rag_harness.api.v1.schemas import AgenticQueryResponse, QueryRequest, QueryResponse
from rag_harness.api.v1.schemas.query_schemas import ActionStep
from rag_harness.application.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])

_last_agent_debug: dict = {}


def _agentic_cache_key(question: str) -> str:
    return hashlib.sha256(f"{question}::agentic::v1".encode()).hexdigest()


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

    # --- Cache check (agentic) ---
    cache_key = _agentic_cache_key(request.question)
    cached_answer = pipeline.cache.query_cache.get(cache_key)
    if cached_answer is not None:
        # Rebuild a minimal response from cache
        cached_debug = _last_agent_debug if _last_agent_debug.get("query") == request.question else {}
        return AgenticQueryResponse(
            answer=cached_answer,
            sources_used=cached_debug.get("sources_used", []),
            degraded=False,
            degrade_reason=None,
            iterations_used=cached_debug.get("iterations_used", 0),
            validation_retries=0,
            token_budget_used=0,
            action_steps=[],
            action_history=[],
            from_cache=True,
        )

    # --- Full agent run ---
    result = pipeline.answer_agentic(request.question)

    raw_actions: list[str] = result.get("action_history", [])
    raw_scratchpad: list[str] = result.get("scratchpad", [])

    action_steps: list[ActionStep] = []
    for i, action in enumerate(raw_actions):
        thought = ""
        if i < len(raw_scratchpad):
            entry = raw_scratchpad[i]
            if "Thought:" in entry:
                thought = entry.split("Thought:")[-1].split("|")[0].strip()
        action_steps.append(ActionStep(action=action, thought=thought, iteration=i + 1))

    # --- Write to cache if not degraded ---
    if not result.get("degraded", False) and result.get("answer"):
        pipeline.cache.query_cache[cache_key] = result["answer"]

    _last_agent_debug = {
        "query": request.question,
        "degraded": result.get("degraded", False),
        "degrade_reason": result.get("degrade_reason"),
        "iterations_used": result.get("iterations_used", 0),
        "validation_retries": result.get("validation_retries", 0),
        "token_budget_used": result.get("token_budget_used", 0),
        "sources_count": len(result.get("sources_used", [])),
        "sources_used": result.get("sources_used", []),
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
        action_history=raw_actions,
        from_cache=False,
    )


@router.get("/agentic/last_state")
def get_last_agent_state():
    return JSONResponse(content=_last_agent_debug)


@router.get("/graph")
def query_graph(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    compiled = pipeline.agentic_use_case.graph
    try:
        mermaid_src = compiled.get_graph().draw_mermaid()
    except Exception as e:
        mermaid_src = f"%% could not draw graph: {e}"
    return {"mermaid": mermaid_src}

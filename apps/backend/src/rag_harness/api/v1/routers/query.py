from fastapi import APIRouter, Depends

from rag_harness.api.dependencies import get_rag_pipeline
from rag_harness.api.v1.schemas import AgenticQueryResponse, QueryRequest, QueryResponse
from rag_harness.application.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    result = pipeline.answer(request.question, doc_version_key=request.doc_version_key)
    return QueryResponse(
        answer=result.answer, sources_used=result.sources_used,
        is_insufficient=result.is_insufficient, from_cache=result.from_cache,
    )


@router.post("/agentic", response_model=AgenticQueryResponse)
def query_agentic(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    result = pipeline.answer_agentic(request.question)
    return AgenticQueryResponse(**result)


@router.get("/graph")
def query_graph(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """Mermaid source for the compiled agent graph — paste into
    https://mermaid.live or render directly in a markdown viewer/README."""
    compiled = pipeline.agentic_use_case.graph
    try:
        mermaid_src = compiled.get_graph().draw_mermaid()
    except Exception as e:
        mermaid_src = f"%% could not draw graph: {e}"
    return {"mermaid": mermaid_src}

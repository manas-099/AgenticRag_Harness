from fastapi import APIRouter, Depends

from rag_harness.api.dependencies import get_rag_pipeline
from rag_harness.application.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    return {"status": "ok", **pipeline.status()}
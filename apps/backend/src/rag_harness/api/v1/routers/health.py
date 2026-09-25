from fastapi import APIRouter, Depends

from rag_harness.api.dependencies import get_llm_chains, get_rag_pipeline
from rag_harness.application.rag_pipeline import RAGPipeline
from rag_harness.infrastructure.llm import LLMChain

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    return {"status": "ok", **pipeline.status()}


@router.get("/llm")
def health_llm(chains: tuple[LLMChain, LLMChain] = Depends(get_llm_chains)):
    """Per-role LLM chain status: configured backend order, which backend
    last succeeded, and whether MyLLM is currently reachable. Backend names
    only ("myllm"/"openrouter"/"groq") — no model names are ever exposed."""
    agent_chain, generate_chain = chains
    return {"agent": agent_chain.status(), "generate": generate_chain.status()}

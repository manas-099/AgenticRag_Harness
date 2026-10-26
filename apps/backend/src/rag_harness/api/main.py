
import logging
from contextlib import asynccontextmanager

# pyrefly: ignore[missing-import]
from fastapi import FastAPI
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware

from rag_harness.api.v1.routers import documents, health, query
from rag_harness.api.dependencies import get_rag_pipeline

logger = logging.getLogger("rag_harness.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # BM25 warm-up: rebuild sparse index from Qdrant on every startup.
    # Without this, BM25 is empty after restart and sparse leg returns []
    # silently for the entire session — hybrid retrieval becomes dense-only.
    try:
        pipeline = get_rag_pipeline()
        all_chunks = pipeline.vector_store.scroll_all_chunks()
        if all_chunks:
            pipeline.sparse_retriever.index(all_chunks)
            logger.info(f"BM25 warm-up complete: indexed {len(all_chunks)} chunks from Qdrant")
        else:
            logger.warning("BM25 warm-up: no chunks found in Qdrant — corpus may be empty")
    except Exception as e:
        logger.error(f"BM25 warm-up failed (non-fatal): {e}")
    yield


app = FastAPI(title="Agentic RAG Harness", version="0.1.0", lifespan=lifespan)

# Mount under both /v1 and /api/v1 so frontend, evals, and curl all work seamlessly
for prefix in ("/v1", "/api/v1"):
    app.include_router(documents.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)
    app.include_router(health.router, prefix=prefix)


@app.get("/health", tags=["health"])
def root_health():
    return {"status": "ok"}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-production-frontend-domain.com"],
    allow_methods=["*"],
    allow_headers=["*"],  # needed so X-LLM-* headers (BYOK) aren't blocked
)
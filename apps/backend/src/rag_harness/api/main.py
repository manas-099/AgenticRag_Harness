from fastapi import FastAPI

from rag_harness.api.v1.routers import documents, health, query

app = FastAPI(title="Agentic RAG Harness", version="0.1.0")

app.include_router(documents.router, prefix="/v1")
app.include_router(query.router, prefix="/v1")
app.include_router(health.router, prefix="/v1")
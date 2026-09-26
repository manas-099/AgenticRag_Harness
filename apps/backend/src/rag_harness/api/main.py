from fastapi import FastAPI

from rag_harness.api.v1.routers import documents, health, query
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Agentic RAG Harness", version="0.1.0")

app.include_router(documents.router, prefix="/v1")
app.include_router(query.router, prefix="/v1")
app.include_router(health.router, prefix="/v1")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-production-frontend-domain.com"],
    allow_methods=["*"],
    allow_headers=["*"],  # needed so X-LLM-* headers (BYOK) aren't blocked
)
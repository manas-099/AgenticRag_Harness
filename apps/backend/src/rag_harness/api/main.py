from fastapi import FastAPI

from rag_harness.api.v1.routers import documents, health, query
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Agentic RAG Harness", version="0.1.0")

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
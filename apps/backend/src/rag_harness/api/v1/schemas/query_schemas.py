from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    doc_version_key: str = "v1"


class QueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    is_insufficient: bool
    from_cache: bool


class AgenticQueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    degraded: bool
    degrade_reason: str | None = None
    iterations_used: int | None = None
    action_history: list[str] = []
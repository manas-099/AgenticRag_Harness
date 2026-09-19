"""
Models for the claim-extraction and verification pipeline.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Claim(BaseModel):
    claim_text: str
    cited_chunk_ids: list[str]


class ClaimList(BaseModel):
    claims: list[Claim]


class GroundednessResult(BaseModel):
    claim: Claim
    entailment_score: float
    contradiction_score: float
    is_grounded: bool
    reason: Optional[str] = None


class VerificationResult(BaseModel):
    passed: bool
    citation_valid: bool
    hallucinated_chunk_ids: list[str]
    groundedness_results: list[GroundednessResult]
    feedback_for_retry: Optional[str] = None


class AnswerResult(BaseModel):
    answer: str
    sources_used: list[str]
    is_insufficient: bool
    retrieved_chunks: list["RetrievedChunk"]
    from_cache: bool = False


# Deferred import to avoid a circular dependency between retrieval.py and
# verification.py (AnswerResult references RetrievedChunk).
from rag_harness.domain.models.retrieval import RetrievedChunk  # noqa: E402

AnswerResult.model_rebuild()
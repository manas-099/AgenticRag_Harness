"""
Core chunk-related domain models. Framework-agnostic — no Qdrant, no LangGraph,
no HTTP concerns. This is what every other layer imports as the shared
vocabulary for "a piece of a document."
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RouteDecision(str, Enum):
    ANYDOC = "anydoc"
    DOCLING = "docling"


class PageStrategy(str, Enum):
    WHOLE_PAGE_ANYDOC = "whole_page_anydoc"
    WHOLE_PAGE_DOCLING = "whole_page_docling"


class PageStrategyDecision(BaseModel):
    page_num: int
    strategy: PageStrategy
    reason: str


class ChunkMetadata(BaseModel):
    doc_id: str
    doc_version: str
    source_type: Literal["pdf", "md"]
    page_num: Optional[int] = None
    section_title: Optional[str] = None
    ingestion_route: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str
    contextual_text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None
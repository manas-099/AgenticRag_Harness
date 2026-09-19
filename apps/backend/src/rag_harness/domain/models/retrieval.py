"""
Models for retrieval results and fusion. Kept separate from chunk.py since
these represent a chunk *plus retrieval context* (score, source), not the
chunk itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from rag_harness.domain.models.chunk import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["dense", "sparse", "fused"]
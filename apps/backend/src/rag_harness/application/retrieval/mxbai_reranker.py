"""
MxbaiReranker — cross-encoder reranker using mixedbread-ai/mxbai-rerank-base-v1.

Replaces CrossEncoderRegistry.reranker() usage in retrieve_use_case.py.
Same interface: reranker.rerank(query, candidates) → list[RetrievedChunk].

No new dependency needed — uses sentence-transformers already installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder

from rag_harness.domain.models import RetrievedChunk

logger = logging.getLogger("rag_harness.rerank.mxbai")

_DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"


@dataclass
class MxbaiRerankerConfig:
    model_name: str = _DEFAULT_MODEL
    top_k: int = 12
    batch_size: int = 32
    show_progress_bar: bool = False
    min_score: float = -5.0
    _extra: dict = field(default_factory=dict, repr=False)


class MxbaiReranker:
    """
    Lazy-loading cross-encoder reranker backed by mxbai-rerank-base-v1.
    CrossEncoder is loaded on first rerank() call and cached as class state.
    """

    __slots__ = ("config",)

    _model: CrossEncoder | None = None
    _loaded_model_name: str | None = None

    def __init__(self, config: MxbaiRerankerConfig | None = None) -> None:
        self.config: MxbaiRerankerConfig = config or MxbaiRerankerConfig()

    @classmethod
    def _get_model(cls, model_name: str) -> CrossEncoder:
        if cls._model is None or cls._loaded_model_name != model_name:
            logger.info("Loading rerank model: %s", model_name)
            cls._model = CrossEncoder(model_name)
            cls._loaded_model_name = model_name
            logger.info("Rerank model loaded: %s", model_name)
        return cls._model

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Re-score and sort candidates. Returns top_k chunks by rerank score."""
        if not candidates:
            return []

        model = self._get_model(self.config.model_name)
        texts = [
            (c.chunk.contextual_text or c.chunk.raw_text or "")
            for c in candidates
        ]
        pairs = [(query, t) for t in texts]
        scores: list[float] = model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            convert_to_numpy=True,
        ).tolist()

        scored = [
            RetrievedChunk(chunk=rc.chunk, score=float(s), source=rc.source)
            for rc, s in zip(candidates, scores)
            if float(s) >= self.config.min_score
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        top = scored[: self.config.top_k] if self.config.top_k else scored

        logger.debug(
            "Reranked %d → %d chunks (top score=%.3f)",
            len(candidates), len(top),
            top[0].score if top else float("nan"),
        )
        return top

    def rerank_scores_only(self, query: str, texts: list[str]) -> list[float]:
        """Return raw rerank scores for (query, text) pairs. For eval scripts."""
        model = self._get_model(self.config.model_name)
        pairs = [(query, t) for t in texts]
        return model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            convert_to_numpy=True,
        ).tolist()

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def unload(cls) -> None:
        cls._model = None
        cls._loaded_model_name = None
        logger.info("MxbaiReranker model unloaded from cache")

    def __repr__(self) -> str:
        return (
            f"MxbaiReranker(model={self.config.model_name!r}, "
            f"top_k={self.config.top_k}, loaded={self.is_loaded()})"
        )

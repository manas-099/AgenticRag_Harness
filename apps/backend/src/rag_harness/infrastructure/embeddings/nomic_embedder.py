"""
NomicEmbedder — HTTP-based embedder for nomic-ai/nomic-embed-text-v1.5
served locally via Ollama or LM Studio.

Drop-in replacement for SentenceTransformerEmbedder: implements EmbedderPort
so the rest of the codebase (retrieve_use_case, ingest pipeline) needs zero
changes other than swapping which class is injected.

Nomic-embed requires task-type prefixes (unlike BGE which only needs them on
queries). Always pass `task` so vectors are in the right subspace:
  - Ingestion  → task="search_document"   → prefix "search_document: "
  - Query time → task="search_query"      → prefix "search_query: "

Ollama API:  POST http://localhost:11434/api/embeddings
LM Studio:   POST http://localhost:1234/v1/embeddings  (OpenAI-compat)

Set NOMIC_BACKEND="ollama" (default) or "lmstudio" in your .env.
Set NOMIC_BASE_URL if you run on a non-default port.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx

from rag_harness.domain.exceptions import IngestionError
from rag_harness.domain.ports import EmbedderPort

logger = logging.getLogger("rag_harness.embedding.nomic")

_TASK_PREFIXES: dict[str, str] = {
    "search_document": "search_document: ",
    "search_query":    "search_query: ",
    "clustering":      "clustering: ",
    "classification":  "classification: ",
}

TaskType = Literal["search_document", "search_query", "clustering", "classification"]


class NomicEmbedder(EmbedderPort):
    """
    Thin HTTP client that delegates embedding to a locally-running Ollama or
    LM Studio server. The model process stays alive outside Python, so FastAPI
    startup never pays the model-load cost.
    """

    __slots__ = ("_model", "_base_url", "_backend", "_timeout", "_client")

    _dim: int = 768

    def __init__(
        self,
        model: str = "nomic-embed-text",
        backend: Literal["ollama", "lmstudio"] = "ollama",
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._backend = backend
        self._base_url = base_url or (
            "http://localhost:11434" if backend == "ollama" else "http://localhost:1234"
        )
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "NomicEmbedder initialised | backend=%s | model=%s | base_url=%s",
            backend, model, self._base_url,
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], task: TaskType = "search_document", **_kwargs) -> list[list[float]]:
        """
        Encode a batch of texts.
        task="search_document" at index time, task="search_query" at query time.
        Do NOT add prefixes yourself — this method adds them.
        """
        if not texts:
            return []
        prefix = _TASK_PREFIXES.get(task, "")
        prefixed = [prefix + t for t in texts]
        try:
            if self._backend == "ollama":
                return self._encode_ollama(prefixed)
            else:
                return self._encode_lmstudio(prefixed)
        except httpx.ConnectError as exc:
            raise IngestionError(
                f"NomicEmbedder: cannot reach {self._backend} at {self._base_url}. "
                "Is the server running?"
            ) from exc
        except Exception as exc:
            raise IngestionError(f"NomicEmbedder encode failed: {exc}") from exc

    def _encode_ollama(self, prefixed_texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in prefixed_texts:
            resp = self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return vectors

    def _encode_lmstudio(self, prefixed_texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            "/v1/embeddings",
            json={"model": self._model, "input": prefixed_texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def close(self) -> None:
        self._client.close()
        logger.info("NomicEmbedder HTTP client closed")

    def __enter__(self) -> "NomicEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"NomicEmbedder(model={self._model!r}, backend={self._backend!r}, "
            f"base_url={self._base_url!r}, dim={self._dim})"
        )

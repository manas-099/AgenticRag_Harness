"""
Factory for constructing the right LLMPort implementation by name. Callers
(application/, api/dependencies.py) depend on this factory and the LLMPort
interface — never on MyLLMClient/GroqFallbackClient directly.
"""

from __future__ import annotations

import logging

from rag_harness.config.settings import GroqSettings, MyLLMSettings
from rag_harness.domain.ports import LLMPort
from rag_harness.infrastructure.llm.groq_client import GroqFallbackClient
from rag_harness.infrastructure.llm.myllm_client import MyLLMClient

logger = logging.getLogger("rag_harness.llm.factory")


class LLMClientFactory:
    _registry = {
        "myllm": MyLLMClient,
        "groq": GroqFallbackClient,
    }

    @classmethod
    def create(cls, backend: str, settings: MyLLMSettings | GroqSettings) -> LLMPort:
        if backend not in cls._registry:
            raise ValueError(f"Unknown LLM backend '{backend}'. Available: {list(cls._registry)}")
        logger.info(f"Instantiating LLM backend: {backend}")
        return cls._registry[backend](settings)
"""
Abstract contract every LLM backend must fulfill. The application layer
depends ONLY on this interface, never on a concrete client — this is what
lets MyLLM, Groq, or any future backend be swapped without touching a single
use case.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMPort(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_content: str,
        response_format: Optional[dict] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        """Returns the model's text response. Implementations own their own
        retry/fallback logic and must raise LLMClientError on final failure."""
        raise NotImplementedError
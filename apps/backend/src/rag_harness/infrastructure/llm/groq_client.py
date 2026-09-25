"""
Groq-backed LLM clients, used for the cheap, high-volume contextual-chunking
preamble calls. GroqFallbackClient wraps a primary + fallback model pair:
retries the primary on rate limits, then switches models for the rest of the
call if the primary is exhausted.
"""

from __future__ import annotations

import logging
import time

from groq import APIStatusError, Groq, RateLimitError

from rag_harness.config.settings import GroqSettings
from rag_harness.domain.exceptions import LLMClientError
from rag_harness.domain.ports import LLMPort

logger = logging.getLogger("rag_harness.llm.groq")


class GroqLLMClient(LLMPort):
    """Wraps Groq's API for a single named model."""

    def __init__(self, api_key: str, model: str, timeout: int):
        self.model = model
        self.timeout = timeout
        self.client = Groq(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        response_format: dict | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        try:
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout,
            )
            # gpt-oss (reasoning model) burns tokens on internal "thinking" —
            # keep that minimal since this call only needs a short answer.
            if "gpt-oss" in self.model:
                kwargs["reasoning_effort"] = "low"

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if not content or not content.strip():
                finish_reason = response.choices[0].finish_reason
                logger.error(f"Empty content from model={self.model}, finish_reason={finish_reason}")
                raise LLMClientError(f"Empty content (finish_reason={finish_reason})")

            return content

        except (RateLimitError, APIStatusError):
            raise
        except LLMClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling model={self.model}: {e}")
            raise


class GroqFallbackClient(LLMPort):
    """Retries the primary model up to N times; on rate-limit or repeated
    failure, switches to the fallback model for that call."""

    def __init__(self, settings: GroqSettings):
        self.settings = settings
        self.primary = GroqLLMClient(settings.API_KEY, settings.PRIMARY_MODEL, settings.TIMEOUT_SECONDS)
        self.fallback = GroqLLMClient(settings.API_KEY, settings.FALLBACK_MODEL, settings.TIMEOUT_SECONDS)

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        response_format: dict | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        last_error = None

        for attempt in range(1, self.settings.MAX_RETRIES_PER_MODEL + 1):
            try:
                logger.info(f"Primary model attempt {attempt}/{self.settings.MAX_RETRIES_PER_MODEL}")
                return self.primary.generate(system_prompt, user_content, response_format, max_tokens, temperature)
            except RateLimitError as e:
                last_error = f"rate_limit: {e}"
                logger.warning(f"Primary rate-limited (attempt {attempt})")
                if attempt < self.settings.MAX_RETRIES_PER_MODEL:
                    time.sleep(self.settings.RETRY_DELAY_SECONDS)
            except Exception as e:
                last_error = f"error: {e}"
                logger.warning(f"Primary failed (attempt {attempt}): {e}")
                if attempt < self.settings.MAX_RETRIES_PER_MODEL:
                    time.sleep(self.settings.RETRY_DELAY_SECONDS)

        logger.warning(f"Primary exhausted (last_error={last_error}) — switching to fallback={self.settings.FALLBACK_MODEL}")
        try:
            return self.fallback.generate(system_prompt, user_content, response_format, max_tokens, temperature)
        except Exception as e:
            raise LLMClientError(
                f"Both primary ({self.settings.PRIMARY_MODEL}) and "
                f"fallback ({self.settings.FALLBACK_MODEL}) failed. "
                f"Primary: {last_error}. Fallback: {e}"
            )
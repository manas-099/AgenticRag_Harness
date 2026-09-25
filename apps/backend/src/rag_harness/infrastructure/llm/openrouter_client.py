"""
Concrete LLMPort implementation for OpenRouter. Sits as a middle tier in the
GENERATE chain (myllm -> openrouter -> groq): cheap/free text-only fallback,
automatically skipped for structured (JSON) calls if its configured model
doesn't support constrained output (see OpenRouterSettings.SUPPORTS_STRUCTURED_OUTPUT).
"""

from __future__ import annotations

import logging
import time

import requests

from rag_harness.config.settings import OpenRouterSettings
from rag_harness.domain.exceptions import LLMClientError
from rag_harness.domain.ports import LLMPort

logger = logging.getLogger("rag_harness.llm.openrouter")


class OpenRouterClient(LLMPort):
    def __init__(self, settings: OpenRouterSettings):
        self.settings = settings

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        response_format: dict | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.settings.MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.API_KEY}",
        }

        last_error = None
        for attempt in range(1, self.settings.MAX_RETRIES + 1):
            try:
                logger.info(f"Calling OpenRouter (attempt {attempt}/{self.settings.MAX_RETRIES})")
                resp = requests.post(
                    self.settings.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.TIMEOUT_SECONDS,
                )
                if resp.status_code >= 400:
                    last_error = f"http_{resp.status_code}: {resp.text[:500]}"
                    logger.warning(f"OpenRouter returned error: {last_error}")
                    if 400 <= resp.status_code < 500:
                        break
                    if attempt < self.settings.MAX_RETRIES:
                        time.sleep(self.settings.RETRY_DELAY_SECONDS)
                    continue

                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    last_error = "empty_content"
                    logger.warning("OpenRouter returned empty content")
                    if attempt < self.settings.MAX_RETRIES:
                        time.sleep(self.settings.RETRY_DELAY_SECONDS)
                    continue
                return content

            except requests.exceptions.RequestException as e:
                last_error = f"request_failed: {e}"
                logger.warning(last_error)
                if attempt < self.settings.MAX_RETRIES:
                    time.sleep(self.settings.RETRY_DELAY_SECONDS)
            except (KeyError, IndexError) as e:
                last_error = f"unexpected_response_shape: {e}"
                logger.error(last_error)
                if attempt < self.settings.MAX_RETRIES:
                    time.sleep(self.settings.RETRY_DELAY_SECONDS)

        logger.error(f"OpenRouterClient giving up after {self.settings.MAX_RETRIES} attempts: {last_error}")
        raise LLMClientError(f"OpenRouterClient giving up: {last_error}")

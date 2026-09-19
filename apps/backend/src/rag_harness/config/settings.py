"""
Concrete LLMPort implementation for the self-hosted MyLLM endpoint.
Used for text-only agents: agent decision-making, answer generation,
claim extraction, section-filter LLM calls.
"""

from __future__ import annotations

import logging
import time

import requests

from rag_harness.config.settings import MyLLMSettings
from rag_harness.domain.exceptions import LLMClientError
from rag_harness.domain.ports import LLMPort

logger = logging.getLogger("rag_harness.llm.myllm")


class MyLLMClient(LLMPort):
    def __init__(self, settings: MyLLMSettings):
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
            "Authorization": f"Bearer {self.settings.AUTH_TOKEN}",
        }

        last_error = None
        for attempt in range(1, self.settings.MAX_RETRIES + 1):
            try:
                logger.info(f"Calling MyLLM (attempt {attempt}/{self.settings.MAX_RETRIES})")
                resp = requests.post(
                    self.settings.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.TIMEOUT_SECONDS,
                )
                if resp.status_code >= 400:
                    last_error = f"http_{resp.status_code}: {resp.text[:1000]}"
                    logger.warning(f"MyLLM returned error: {last_error}")
                    if 400 <= resp.status_code < 500:
                        break  # client error, retrying won't help
                    if attempt < self.settings.MAX_RETRIES:
                        time.sleep(self.settings.RETRY_DELAY_SECONDS)
                    continue

                resp.raise_for_status()
                result = resp.json()
                return result["choices"][0]["message"]["content"]

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

        logger.error(f"MyLLMClient giving up after {self.settings.MAX_RETRIES} attempts: {last_error}")
        raise LLMClientError(f"MyLLMClient giving up: {last_error}")
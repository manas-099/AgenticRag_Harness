"""
LLMChain: wraps an ordered list of LLMPort backends behind a single LLMPort.
Two independently-configured chains exist (see config/settings.py
LLMChainSettings): one for the agent's tool-decision step, one for answer
generation / claim extraction / section filtering. Same class, different
backend order — set via LLM_AGENT_CHAIN / LLM_GENERATE_CHAIN in .env.

Behaviour:
  * myllm entries consult MyLLMHealthMonitor first. If myllm was already
    marked down within the cooldown window, it's skipped without a network
    call. Otherwise the health monitor's 2-attempt probe runs, and only on
    success is myllm actually called for this request.
  * A backend that doesn't support structured output (response_format) is
    skipped for calls that pass response_format — e.g. the free OpenRouter
    model can't reliably emit constrained JSON.
  * On any failure, the chain logs a warning and moves to the next backend.
  * Log messages use the backend's logical name ("myllm"/"openrouter"/"groq")
    only — never the real underlying model name, which lives only in .env
    and is never printed.
"""

from __future__ import annotations

import logging

from rag_harness.domain.exceptions import LLMClientError
from rag_harness.domain.ports import LLMPort
from rag_harness.infrastructure.llm.health import MyLLMHealthMonitor

logger = logging.getLogger("rag_harness.llm.chain")


class LLMChain(LLMPort):
    def __init__(
        self,
        role: str,
        order: list[str],
        backends: dict[str, LLMPort],
        supports_structured: dict[str, bool],
        myllm_health: MyLLMHealthMonitor | None,
    ):
        """
        role: a label for logging only, e.g. "agent" or "generate".
        order: backend names in try-order, e.g. ["myllm", "openrouter", "groq"].
        backends: name -> constructed LLMPort instance.
        supports_structured: name -> whether it can be trusted with response_format.
        myllm_health: shared health monitor instance (None if "myllm" isn't in order).
        """
        missing = [name for name in order if name not in backends]
        if missing:
            raise ValueError(f"LLMChain '{role}': unknown backend(s) in order: {missing}")

        self.role = role
        self.order = order
        self.backends = backends
        self.supports_structured = supports_structured
        self.myllm_health = myllm_health
        self.last_backend_used: str | None = None

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        response_format: dict | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        errors: list[str] = []

        for name in self.order:
            if response_format is not None and not self.supports_structured.get(name, True):
                logger.info(f"[{self.role}] skipping '{name}' — doesn't support structured output")
                continue

            if name == "myllm" and self.myllm_health is not None:
                if not self.myllm_health.is_available():
                    errors.append("myllm: skipped (health check failed)")
                    continue

            try:
                out = self.backends[name].generate(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if name == "myllm" and self.myllm_health is not None:
                    self.myllm_health.report_success()
                self.last_backend_used = name
                logger.info(f"[{self.role}] backend '{name}' succeeded")
                return out

            except Exception as e:
                if name == "myllm" and self.myllm_health is not None:
                    self.myllm_health.report_failure()
                errors.append(f"{name}: {e}")
                logger.warning(f"[{self.role}] backend '{name}' failed — moving to next in chain: {e}")

        raise LLMClientError(f"LLMChain '{self.role}': all backends failed. Errors: {' | '.join(errors)}")

    def status(self) -> dict:
        """Per-backend availability snapshot, used by GET /v1/health/llm."""
        result = {"role": self.role, "order": self.order, "last_backend_used": self.last_backend_used}
        if self.myllm_health is not None and "myllm" in self.order:
            result["myllm_reachable"] = self.myllm_health.is_available()
        return result

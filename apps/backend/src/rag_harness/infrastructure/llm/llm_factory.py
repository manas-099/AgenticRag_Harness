"""
Factory for constructing the AGENT and GENERATE LLMChains from settings.
Callers (api/dependencies.py) depend on this factory and the LLMPort
interface — never on MyLLMClient/OpenRouterClient/GroqFallbackClient directly.
"""

from __future__ import annotations

import logging

from rag_harness.config.settings import GroqSettings, LLMChainSettings, MyLLMSettings, OpenRouterSettings
from rag_harness.infrastructure.llm.chain import LLMChain
from rag_harness.infrastructure.llm.groq_client import GroqFallbackClient
from rag_harness.infrastructure.llm.health import MyLLMHealthMonitor
from rag_harness.infrastructure.llm.myllm_client import MyLLMClient
from rag_harness.infrastructure.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger("rag_harness.llm.factory")


class LLMClientFactory:
    @staticmethod
    def build_chains(
        chain_settings: LLMChainSettings,
        myllm_settings: MyLLMSettings,
        openrouter_settings: OpenRouterSettings,
        groq_settings: GroqSettings,
    ) -> tuple[LLMChain, LLMChain]:
        """Returns (agent_chain, generate_chain), sharing one MyLLM client
        and one health monitor instance so both chains see the same live
        up/down state instead of probing independently."""
        myllm_client = MyLLMClient(myllm_settings)
        myllm_health = MyLLMHealthMonitor(myllm_settings)
        openrouter_client = OpenRouterClient(openrouter_settings)
        groq_client = GroqFallbackClient(groq_settings)

        backends = {"myllm": myllm_client, "openrouter": openrouter_client, "groq": groq_client}
        supports_structured = {
            "myllm": True,
            "openrouter": openrouter_settings.SUPPORTS_STRUCTURED_OUTPUT,
            "groq": True,
        }

        agent_chain = LLMChain(
            role="agent", order=chain_settings.agent_chain_order, backends=backends,
            supports_structured=supports_structured, myllm_health=myllm_health,
        )
        generate_chain = LLMChain(
            role="generate", order=chain_settings.generate_chain_order, backends=backends,
            supports_structured=supports_structured, myllm_health=myllm_health,
        )
        logger.info(f"agent_chain order={agent_chain.order}  generate_chain order={generate_chain.order}")
        return agent_chain, generate_chain

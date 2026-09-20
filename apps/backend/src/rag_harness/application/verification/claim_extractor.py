"""
Extracts individual claims + their citations from a generated answer via
structured LLM output. Regex parsing of free-form citation text is fragile
(citation punctuation/spacing can vary); structured output enforced at the
model level is not.
"""

from __future__ import annotations

import logging
import re

from rag_harness.domain.models import Claim, ClaimList
from rag_harness.domain.ports import LLMPort
from rag_harness.infrastructure.llm.xgrammar import XGrammar

logger = logging.getLogger("rag_harness.claim_extractor")


class ClaimExtractor:
    def __init__(self, llm_client: LLMPort):
        self.llm_client = llm_client
        self.grammar = XGrammar(ClaimList, name="claim_extraction")

    def extract(self, answer_text: str, available_chunk_ids: list[str]) -> list[Claim]:
        system_prompt = (
            "Extract every distinct factual claim from the given answer text. "
            "For each claim, list the chunk_id(s) that support it, using ONLY ids "
            f"from this available list: {available_chunk_ids}. "
            "If a claim has no clearly supporting chunk_id, give it an empty list rather than guessing one."
        )
        try:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_content=answer_text,
                response_format=self.grammar.build_response_format(),
                max_tokens=1000,
                temperature=0.0,
            )
            parsed = ClaimList.model_validate_json(response)
            logger.info(f"Extracted {len(parsed.claims)} structured claims")
            return parsed.claims
        except Exception as e:
            logger.error(f"Structured claim extraction failed: {e} — falling back to regex")
            return self._regex_fallback(answer_text)

    def _regex_fallback(self, answer_text: str) -> list[Claim]:
        claims = []
        sentences = re.split(r"(?<=[.!?])\s+", answer_text)
        for sentence in sentences:
            cited_ids = re.findall(r"chunk_id:\s*([\w-]+)", sentence)
            if cited_ids:
                claims.append(Claim(claim_text=sentence, cited_chunk_ids=cited_ids))
        return claims
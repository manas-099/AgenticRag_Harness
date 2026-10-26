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
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            logger.warning("Empty or non-string answer_text provided to ClaimExtractor — returning empty claims")
            return []

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
            backend_name = getattr(self.llm_client, "last_backend_used", "unknown")
            parsed = ClaimList.model_validate_json(response)
            logger.info(f"Extracted {len(parsed.claims)} structured claims using backend '{backend_name}'")
            return parsed.claims
        except Exception as e:
            backend_name = getattr(self.llm_client, "last_backend_used", "unknown")
            logger.error(
                f"Structured claim extraction failed on backend '{backend_name}': {e} — falling back to regex"
            )
            return self._regex_fallback(answer_text)

    def _regex_fallback(self, answer_text: str) -> list[Claim]:
        if not answer_text or not isinstance(answer_text, str):
            return []
        claims = []
        sentences = re.split(r"(?<=[.!?])\s+", answer_text)
        for sentence in sentences:
            # Check for chunk_id citations (legacy/test format) or [Doc: <doc_id>, Page: <page_num>]
            chunk_citations = re.findall(r"chunk_id:\s*([\w-]+)", sentence)
            doc_citations = re.findall(r"\[(?:Doc|doc):\s*([^,\]]+)(?:,\s*(?:Page|page):\s*([^\]]+))?\]", sentence)
            cited_ids = chunk_citations + [
                f"{doc.strip()}:p{page.strip()}" if page else doc.strip()
                for doc, page in doc_citations
            ]
            if cited_ids:
                claims.append(Claim(claim_text=sentence.strip(), cited_chunk_ids=cited_ids))
        return claims
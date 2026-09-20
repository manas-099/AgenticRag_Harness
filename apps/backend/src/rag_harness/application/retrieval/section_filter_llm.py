"""
Final section pick via structured LLM output. Validates the LLM's choice
against the real candidate list before trusting it — never applies a
section filter the LLM invented that wasn't actually offered as an option.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from rag_harness.domain.ports import LLMPort
from rag_harness.infrastructure.llm.xgrammar import XGrammar

logger = logging.getLogger("rag_harness.section_filter_llm")


class SectionSelection(BaseModel):
    sections: list[str]


FILTER_SYSTEM_PROMPT_TEMPLATE = """You are a query analyzer for a document retrieval system.

Given a user question, decide which section(s) from this candidate list it clearly relates to:
{sections}

Rules:
- Only include a section if the question is CLEARLY about that topic — do not guess broadly.
- Section names in your response must match EXACTLY (character for character) entries from the list above.
- If the question is general/broad and doesn't clearly match specific sections, return an empty list.
"""


class SectionFilterLLM:
    def __init__(self, llm_client: LLMPort):
        self.llm_client = llm_client
        self.grammar = XGrammar(SectionSelection, name="section_selection")

    def select_sections(self, query: str, candidate_sections: list[str]) -> list[str]:
        if not candidate_sections:
            return []

        system_prompt = FILTER_SYSTEM_PROMPT_TEMPLATE.format(sections=candidate_sections)
        try:
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_content=query,
                response_format=self.grammar.build_response_format(),
                max_tokens=300,
                temperature=0.0,
            )
            parsed = SectionSelection.model_validate_json(response)
            raw_sections = parsed.sections
        except Exception as e:
            logger.error(f"Section filter LLM call/parse failed: {e} — treating as no filter")
            return []

        valid = [s for s in raw_sections if s in candidate_sections]
        invalid = [s for s in raw_sections if s not in candidate_sections]
        if invalid:
            logger.warning(f"LLM returned unknown sections, ignoring: {invalid}")

        logger.info(f"Matched sections: {valid}")
        return valid
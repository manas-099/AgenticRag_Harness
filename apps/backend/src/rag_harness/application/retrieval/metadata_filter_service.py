"""
Orchestrates candidate selection + LLM final pick, builds the actual Qdrant
filter. Degrades to no filter (unfiltered retrieval) if nothing clearly
matches, rather than forcing a possibly-wrong filter on a general query.
"""

from __future__ import annotations

import logging

from qdrant_client.models import FieldCondition, Filter, MatchAny

from rag_harness.application.retrieval.section_candidate_selector import SectionCandidateSelector
from rag_harness.application.retrieval.section_filter_llm import SectionFilterLLM

logger = logging.getLogger("rag_harness.metadata_filter")


class MetadataFilterService:
    def __init__(self, candidate_selector: SectionCandidateSelector, filter_llm: SectionFilterLLM):
        self.candidate_selector = candidate_selector
        self.filter_llm = filter_llm

    def build_filter(self, query: str) -> tuple[dict | Filter | None, list[str]]:
        candidates = self.candidate_selector.get_candidates(query)
        if not candidates:
            return None, []

        matched = self.filter_llm.select_sections(query, candidates)
        if not matched:
            logger.info("No section matched — query treated as general/broad, no filter applied")
            return None, []

        if len(matched) == 1:
            return {"section_title": matched[0]}, matched
        return Filter(must=[FieldCondition(key="section_title", match=MatchAny(any=matched))]), matched
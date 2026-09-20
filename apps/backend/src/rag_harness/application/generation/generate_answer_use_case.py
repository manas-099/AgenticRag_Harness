"""
Context assembly + relevance gate + LLM call. The relevance gate is the
primary hallucination guardrail: if even the top reranked chunk scores below
threshold, the LLM is never called — the system returns "insufficient
information" directly, rather than forcing a confident answer from weak
evidence.
"""

from __future__ import annotations

import logging

from rag_harness.application.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from rag_harness.config.settings import RetrievalSettings
from rag_harness.domain.exceptions import LLMClientError
from rag_harness.domain.models import AnswerResult, RetrievedChunk
from rag_harness.domain.ports import LLMPort
from rag_harness.infrastructure.cache import InMemoryCache

logger = logging.getLogger("rag_harness.generation")

INSUFFICIENT_INFO_ANSWER = "I don't have enough information in the provided documents to answer this."


class GenerateAnswerUseCase:
    def __init__(self, llm_client: LLMPort, cache: InMemoryCache, settings: RetrievalSettings):
        self.llm_client = llm_client
        self.cache = cache
        self.settings = settings

    def _assemble_context(self, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for rc in chunks:
            meta = rc.chunk.metadata
            blocks.append(
                f"[Doc: {meta.doc_id}, Page: {meta.page_num}, chunk_id: {rc.chunk.chunk_id}]\n{rc.chunk.raw_text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _is_relevance_sufficient(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        return max(c.score for c in chunks) >= self.settings.MIN_RELEVANCE_SCORE

    def execute(self, question: str, chunks: list[RetrievedChunk], doc_version_key: str = "v1") -> AnswerResult:
        cached = self.cache.get_query(question, doc_version_key)
        if cached:
            return AnswerResult(answer=cached, sources_used=[], is_insufficient=False, retrieved_chunks=chunks, from_cache=True)

        if not self._is_relevance_sufficient(chunks):
            logger.info("Relevance below threshold — returning insufficient-information response")
            return AnswerResult(
                answer=INSUFFICIENT_INFO_ANSWER, sources_used=[], is_insufficient=True,
                retrieved_chunks=chunks, from_cache=False,
            )

        context_block = self._assemble_context(chunks)
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context_block=context_block, question=question)

        try:
            answer_text = self.llm_client.generate(RAG_SYSTEM_PROMPT, user_prompt, max_tokens=800, temperature=0.0)
        except LLMClientError as e:
            logger.error(f"Generation LLM call failed: {e}")
            answer_text = "I encountered an error generating an answer. Please try again."

        sources_used = [rc.chunk.chunk_id for rc in chunks]
        self.cache.set_query(question, doc_version_key, answer_text)

        return AnswerResult(answer=answer_text, sources_used=sources_used, is_insufficient=False, retrieved_chunks=chunks, from_cache=False)
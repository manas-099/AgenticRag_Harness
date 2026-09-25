"""
Orchestrates citation validation -> groundedness check, in strict order
(cheapest/code-only first), and builds specific, actionable feedback for a
regeneration retry rather than a generic "try again" message.
"""

from __future__ import annotations

import logging

from rag_harness.application.verification.citation_validator import CitationValidator
from rag_harness.application.verification.claim_extractor import ClaimExtractor
from rag_harness.application.verification.groundedness_checker import GroundednessChecker
from rag_harness.domain.models import Chunk, VerificationResult

logger = logging.getLogger("rag_harness.verification")


class VerifyAnswerUseCase:
    def __init__(self, claim_extractor: ClaimExtractor, citation_validator: CitationValidator, groundedness_checker: GroundednessChecker):
        self.claim_extractor = claim_extractor
        self.citation_validator = citation_validator
        self.groundedness_checker = groundedness_checker

    def execute(self, answer_text: str, chunk_registry: dict[str, Chunk]) -> VerificationResult:
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            logger.warning("Empty or non-string answer_text in VerifyAnswerUseCase")
            return VerificationResult(
                passed=False,
                citation_valid=False,
                hallucinated_chunk_ids=[],
                groundedness_results=[],
                feedback_for_retry="No valid draft answer was generated.",
            )

        available_ids = list(chunk_registry.keys())
        claims = self.claim_extractor.extract(answer_text, available_ids)

        if not claims:
            logger.info("No claims extracted — nothing to verify, treating as passed")
            return VerificationResult(passed=True, citation_valid=True, hallucinated_chunk_ids=[], groundedness_results=[])

        citation_valid, hallucinated = self.citation_validator.validate(claims, chunk_registry)
        if not citation_valid:
            feedback = (
                f"Your answer cited chunk_id(s) that were never retrieved: {hallucinated}. "
                f"Only cite chunk_ids from this list: {available_ids}. Regenerate the answer."
            )
            logger.warning(f"Citation validation FAILED: {hallucinated}")
            return VerificationResult(passed=False, citation_valid=False, hallucinated_chunk_ids=hallucinated, groundedness_results=[], feedback_for_retry=feedback)

        groundedness_results = self.groundedness_checker.check_all_claims(claims, chunk_registry)
        ungrounded = [r for r in groundedness_results if not r.is_grounded]

        if ungrounded:
            reasons = "; ".join(f"'{r.claim.claim_text[:80]}...' -> {r.reason}" for r in ungrounded)
            feedback = (
                f"The following claim(s) are not clearly supported by their cited source text: {reasons}. "
                f"Either remove these claims, rephrase them to match the source, or cite a more accurate chunk_id."
            )
            logger.warning(f"Groundedness FAILED for {len(ungrounded)} claim(s)")
            return VerificationResult(passed=False, citation_valid=True, hallucinated_chunk_ids=[], groundedness_results=groundedness_results, feedback_for_retry=feedback)

        logger.info("Verification PASSED")
        return VerificationResult(passed=True, citation_valid=True, hallucinated_chunk_ids=[], groundedness_results=groundedness_results)
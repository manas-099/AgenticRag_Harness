"""
NLI-based groundedness check: for every claim that passed citation
validation, verifies the claim's TEXT is actually entailed by its cited
chunk's text — catching the more dangerous case where a citation is real
but the claim overstates or misstates what the source actually says.
Runs on ALL claims, not scoped to numeric/high-stakes ones only.
"""

from __future__ import annotations

import logging

from rag_harness.application.retrieval.model_registry import CrossEncoderRegistry
from rag_harness.config.settings import NLISettings
from rag_harness.domain.models import Chunk, Claim, GroundednessResult

logger = logging.getLogger("rag_harness.groundedness")


class GroundednessChecker:
    def __init__(self, model_registry: CrossEncoderRegistry, settings: NLISettings):
        self.model_registry = model_registry
        self.settings = settings

    def check_claim(self, claim: Claim, chunk_registry: dict[str, Chunk]) -> GroundednessResult:
        if not claim.cited_chunk_ids:
            return GroundednessResult(claim=claim, entailment_score=0.0, contradiction_score=0.0, is_grounded=False, reason="no_citation_provided")

        cited_texts = []
        for cid in claim.cited_chunk_ids:
            if cid not in chunk_registry:
                logger.error(f"Claim cites unknown chunk_id={cid} — treating as ungrounded")
                return GroundednessResult(claim=claim, entailment_score=0.0, contradiction_score=1.0, is_grounded=False, reason="cited_chunk_not_in_registry")
            cited_texts.append(chunk_registry[cid].raw_text)

        premise = " ".join(cited_texts)
        hypothesis = claim.claim_text

        try:
            nli_model = self.model_registry.nli_model()
            scores = nli_model.predict([(premise, hypothesis)])[0]
            contradiction_score, entailment_score, _neutral_score = float(scores[0]), float(scores[1]), float(scores[2])

            is_grounded = (
                entailment_score >= self.settings.NLI_ENTAILMENT_THRESHOLD
                and contradiction_score <= self.settings.NLI_CONTRADICTION_MAX
            )
            reason = None if is_grounded else (
                f"entailment={entailment_score:.2f} (need >= {self.settings.NLI_ENTAILMENT_THRESHOLD}), "
                f"contradiction={contradiction_score:.2f} (need <= {self.settings.NLI_CONTRADICTION_MAX})"
            )
            return GroundednessResult(claim=claim, entailment_score=entailment_score, contradiction_score=contradiction_score, is_grounded=is_grounded, reason=reason)
        except Exception as e:
            logger.error(f"NLI check failed for claim '{hypothesis[:50]}...': {e}")
            return GroundednessResult(claim=claim, entailment_score=0.0, contradiction_score=1.0, is_grounded=False, reason=f"nli_check_error: {e}")

    def check_all_claims(self, claims: list[Claim], chunk_registry: dict[str, Chunk]) -> list[GroundednessResult]:
        return [self.check_claim(c, chunk_registry) for c in claims]
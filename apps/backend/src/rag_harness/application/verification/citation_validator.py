"""
Code-only, cheapest verification check, runs first. A citation to a chunk_id
that was never retrieved means the model fabricated a source — a hard
failure, not a judgment call, so no model is needed to check it.
"""

from __future__ import annotations

import logging

from rag_harness.domain.models import Chunk, Claim

logger = logging.getLogger("rag_harness.citation_validator")


class CitationValidator:
    def validate(self, claims: list[Claim], chunk_registry: dict[str, Chunk]) -> tuple[bool, list[str]]:
        hallucinated = []
        for claim in claims:
            for cid in claim.cited_chunk_ids:
                if cid not in chunk_registry:
                    hallucinated.append(cid)
                    logger.warning(f"Hallucinated citation detected: {cid}")
        return (len(hallucinated) == 0, hallucinated)
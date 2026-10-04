"""
Orchestrates citation validation → LLM groundedness check.

CHANGED: NLI-based GroundednessChecker replaced with a single myllm call.
WHY: NLI cross-encoder fails on paraphrased, comparative, and multi-chunk
synthesized answers — it checks literal textual entailment, not semantic
grounding. myllm understands paraphrase and inference natively, making it
far more appropriate for verifying RAG-generated answers.

FLOW:
  1. Citation validation (unchanged — pure string match, catches hallucinated
     chunk_ids cheaply before any LLM call)
  2. LLM groundedness check — myllm reads the answer + source chunks and
     returns structured JSON: { passed, ungrounded_claims, feedback }
  3. Build VerificationResult from LLM response
"""

from __future__ import annotations

import json
import logging
import re

from rag_harness.application.verification.citation_validator import CitationValidator
from rag_harness.application.verification.claim_extractor import ClaimExtractor
from rag_harness.domain.models import Chunk, VerificationResult
from rag_harness.domain.models.verification import Claim, GroundednessResult
from rag_harness.domain.ports import LLMPort

logger = logging.getLogger("rag_harness.verification")


# ── LLM verification prompt ────────────────────────────────────────────────────

LLM_VERIFY_SYSTEM_PROMPT = """You are a strict factual verification assistant.

Your job: decide whether an answer is grounded in the provided source chunks.

GROUNDED means: every factual claim in the answer can be traced back to — or
reasonably inferred from — the source chunks. Paraphrasing is fine. Synthesis
across multiple chunks is fine. What is NOT fine: claims that contradict the
sources, or claims that introduce facts not present anywhere in the sources.

You must respond with ONLY valid JSON — no explanation, no markdown, no extra text.

JSON schema:
{
  "passed": true or false,
  "ungrounded_claims": ["claim text that has no support", ...],
  "feedback": "one sentence telling the generator what to fix, or empty string if passed"
}

Rules:
- "passed": true if ALL claims are grounded. false if ANY claim is ungrounded.
- "ungrounded_claims": list the exact problematic claim text. Empty list if passed=true.
- "feedback": specific and actionable. Tell the generator exactly what to remove or fix.
  Example: "The claim 'normal RAG is single-pass' is not stated in the sources — remove it or cite a chunk that supports it."
- Do NOT fail an answer for paraphrasing or reasonable inference from the sources.
- Do NOT fail an answer for good summarization of multiple chunks.
- ONLY fail claims that are genuinely absent from or contradicted by the sources."""


LLM_VERIFY_USER_TEMPLATE = """SOURCE CHUNKS:
{chunks_block}

ANSWER TO VERIFY:
{answer}

Is this answer grounded in the source chunks above?
Respond with JSON only."""


def _build_chunks_block(chunk_registry: dict[str, Chunk]) -> str:
    """Build a readable source block from the chunk registry."""
    blocks = []
    for cid, chunk in chunk_registry.items():
        preview = chunk.raw_text[:600]
        blocks.append(f"[chunk_id: {cid[:12]}]\n{preview}")
    return "\n\n---\n\n".join(blocks)


def _parse_llm_verify_response(response: str) -> dict:
    """Parse LLM JSON response, stripping markdown fences if present."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object from free text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ── Use case ───────────────────────────────────────────────────────────────────

class VerifyAnswerUseCase:
    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        citation_validator: CitationValidator,
        llm_client: LLMPort,
    ):
        self.claim_extractor = claim_extractor
        self.citation_validator = citation_validator
        self.llm_client = llm_client
        # GroundednessChecker (NLI) intentionally NOT injected here anymore.
        # Keep it in dependencies.py if you want to restore NLI later.

    def execute(
        self, answer_text: str, chunk_registry: dict[str, Chunk]
    ) -> VerificationResult:

        # ── Guard: empty answer ───────────────────────────────────────────────
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            logger.warning("Empty or non-string answer_text in VerifyAnswerUseCase")
            return VerificationResult(
                passed=False,
                citation_valid=False,
                hallucinated_chunk_ids=[],
                groundedness_results=[],
                feedback_for_retry="No valid draft answer was generated.",
            )

        # ── Step 1: Citation validation (cheap, code-only) ────────────────────
        available_ids = list(chunk_registry.keys())
        claims = self.claim_extractor.extract(answer_text, available_ids)

        if claims:
            citation_valid, hallucinated = self.citation_validator.validate(
                claims, chunk_registry
            )
            if not citation_valid:
                feedback = (
                    f"Your answer cited chunk_id(s) that were never retrieved: "
                    f"{hallucinated}. Only cite chunk_ids from this list: "
                    f"{available_ids}. Regenerate the answer."
                )
                logger.warning(f"Citation validation FAILED: {hallucinated}")
                return VerificationResult(
                    passed=False,
                    citation_valid=False,
                    hallucinated_chunk_ids=hallucinated,
                    groundedness_results=[],
                    feedback_for_retry=feedback,
                )
        else:
            # No inline citations — still run LLM groundedness check
            logger.info("No inline citations found — proceeding to LLM groundedness check")
            citation_valid = True
            hallucinated = []

        # ── Step 2: LLM groundedness check ───────────────────────────────────
        if not chunk_registry:
            # Nothing to verify against — pass with warning
            logger.warning("chunk_registry empty — skipping LLM verification, treating as passed")
            return VerificationResult(
                passed=True,
                citation_valid=True,
                hallucinated_chunk_ids=[],
                groundedness_results=[],
            )

        chunks_block = _build_chunks_block(chunk_registry)
        user_prompt = LLM_VERIFY_USER_TEMPLATE.format(
            chunks_block=chunks_block,
            answer=answer_text.strip(),
        )

        try:
            response = self.llm_client.generate(
                system_prompt=LLM_VERIFY_SYSTEM_PROMPT,
                user_content=user_prompt,
                max_tokens=300,
                temperature=0.0,
            )
            parsed = _parse_llm_verify_response(response)

            passed = bool(parsed.get("passed", False))
            ungrounded_texts: list[str] = parsed.get("ungrounded_claims", [])
            feedback: str = parsed.get("feedback", "")

            # Build GroundednessResult list from LLM output
            # (keeps VerificationResult shape identical — no downstream changes)
            groundedness_results = []
            for text in ungrounded_texts:
                groundedness_results.append(
                    GroundednessResult(
                        claim=Claim(claim_text=text, cited_chunk_ids=[]),
                        entailment_score=0.0,    # not applicable for LLM verifier
                        contradiction_score=0.0,
                        is_grounded=False,
                        reason="LLM verifier: not supported by source chunks",
                    )
                )

            if passed:
                logger.info("LLM verification PASSED")
                return VerificationResult(
                    passed=True,
                    citation_valid=True,
                    hallucinated_chunk_ids=[],
                    groundedness_results=groundedness_results,
                )
            else:
                logger.warning(
                    f"LLM verification FAILED — {len(ungrounded_texts)} ungrounded claim(s)"
                )
                return VerificationResult(
                    passed=False,
                    citation_valid=True,
                    hallucinated_chunk_ids=[],
                    groundedness_results=groundedness_results,
                    feedback_for_retry=feedback or (
                        f"The following claims are not supported by the source chunks: "
                        f"{ungrounded_texts}. Remove or rephrase them."
                    ),
                )

        except Exception as e:
            # LLM call failed — fail open (treat as passed) so one broken
            # verification call doesn't degrade every answer.
            # Log at ERROR so it's visible in terminal.
            logger.error(
                f"LLM verification call failed ({e}) — treating as passed (fail-open)"
            )
            return VerificationResult(
                passed=True,
                citation_valid=True,
                hallucinated_chunk_ids=[],
                groundedness_results=[],
                feedback_for_retry=None,
            )
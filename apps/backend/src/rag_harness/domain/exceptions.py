"""
Explicit failure taxonomy for the whole harness. Every layer raises one of
these (or lets one propagate) rather than a bare Exception, so callers can
catch precisely what they mean to handle.
"""


class HarnessError(Exception):
    """Base class for all harness-level errors."""


class LLMClientError(HarnessError):
    """Raised when an LLM backend fails after all retries."""


class IngestionError(HarnessError):
    """Raised when a document fails to parse/route/extract during ingestion."""


class RetrievalError(HarnessError):
    """Raised when the retrieval pipeline fails (vector DB down, empty index, etc)."""


class ValidationError(HarnessError):
    """Raised when citation validation hard-fails (hallucinated chunk_id)."""


class LoopBudgetExceeded(HarnessError):
    """Raised (and caught internally) when iteration/token/search caps are hit."""
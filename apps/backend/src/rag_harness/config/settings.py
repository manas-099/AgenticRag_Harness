"""
All runtime configuration, read from environment variables / .env, via
pydantic-settings. This is the ONLY place `os.environ` / `.env` is read —
every other module receives settings objects through dependency injection
(see api/dependencies.py), never reads the environment directly.

NOTE: this file was previously accidentally overwritten with the content of
infrastructure/llm/myllm_client.py (a copy-paste mistake) — nothing in the
project could actually run, since every module importing `get_*_settings`
or `*Settings` from here was importing names that didn't exist. This is the
restored, correct version.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvBase(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class EnvironmentSettings(_EnvBase):
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "rag_harness_chunks"

    @property
    def uses_remote_qdrant(self) -> bool:
        """True when a remote Qdrant URL is configured; False = embedded local storage."""
        return bool(self.QDRANT_URL and self.QDRANT_URL.strip())

    @property
    def qdrant_path(self) -> str:
        """Local filesystem path for embedded Qdrant (dev / no-server mode)."""
        return "./qdrant_local_storage"


# ---------------------------------------------------------------------------
# MyLLM — the self-hosted primary backend. MODEL_NAME/API_URL/AUTH_TOKEN are
# secrets and stay in .env only; they are never hardcoded or logged. Logger
# names and log messages everywhere use "myllm", never the real model name.
# ---------------------------------------------------------------------------
class MyLLMSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="MYLLM_")

    API_URL: str = ""
    AUTH_TOKEN: str = ""
    MODEL_NAME: str = ""
    TIMEOUT_SECONDS: int = 60
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 2.0

    # Health-check behaviour (see infrastructure/llm/health.py):
    # on each generate() call the chain first checks a cached health flag
    # instead of always trying MyLLM cold. If MyLLM was last seen down, it is
    # skipped until COOLDOWN_SECONDS has passed since the last failed check.
    HEALTH_CHECK_ATTEMPTS: int = 2
    HEALTH_CHECK_TIMEOUT_SECONDS: int = 10
    HEALTH_CHECK_COOLDOWN_SECONDS: int = 60


class OpenRouterSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="OPENROUTER_")

    API_KEY: str = ""
    MODEL_NAME: str = "liquid/lfm-2.5-2.6b:free"
    API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 2
    RETRY_DELAY_SECONDS: float = 1.0
    # The free tier model can't reliably emit constrained JSON —
    # skipped automatically for any call that passes response_format.
    SUPPORTS_STRUCTURED_OUTPUT: bool = False


class GroqSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="GROQ_")

    API_KEY: str = ""
    PRIMARY_MODEL: str = "openai/gpt-oss-120b"
    FALLBACK_MODEL: str = "qwen/qwen3-32b"
    TIMEOUT_SECONDS: int = 30
    MAX_RETRIES_PER_MODEL: int = 2
    RETRY_DELAY_SECONDS: float = 3.0

    @property
    def GROQ_API_KEY(self) -> str:
        return self.API_KEY

    @property
    def GROQ_PRIMARY_MODEL(self) -> str:
        return self.PRIMARY_MODEL

    @property
    def GROQ_FALLBACK_MODEL(self) -> str:
        return self.FALLBACK_MODEL

    @property
    def GROQ_TIMEOUT_SECONDS(self) -> int:
        return self.TIMEOUT_SECONDS

    @property
    def GROQ_MAX_RETRIES_PER_MODEL(self) -> int:
        return self.MAX_RETRIES_PER_MODEL

    @property
    def GROQ_RETRY_DELAY_SECONDS(self) -> float:
        return self.RETRY_DELAY_SECONDS


# ---------------------------------------------------------------------------
# LLM chain roles — the harness makes two independent LLM calls with
# different quality/latency needs, so each gets its own configurable order:
#   AGENT   -> the ReAct decision step (agent_decide_node): "what tool next?"
#   GENERATE -> answer generation, claim extraction, section filtering.
# Order is a comma-separated list of backend names in .env, e.g.
#   LLM_AGENT_CHAIN=myllm,groq
#   LLM_GENERATE_CHAIN=myllm,openrouter,groq
# Valid names: "myllm", "openrouter", "groq".
# ---------------------------------------------------------------------------
class LLMChainSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="LLM_")

    AGENT_CHAIN: str = "myllm,groq"
    GENERATE_CHAIN: str = "myllm,openrouter,groq"

    @property
    def agent_chain_order(self) -> list[str]:
        return [b.strip() for b in self.AGENT_CHAIN.split(",") if b.strip()]

    @property
    def generate_chain_order(self) -> list[str]:
        return [b.strip() for b in self.GENERATE_CHAIN.split(",") if b.strip()]


class HarnessLoopSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="HARNESS_")

    MAX_ITERATIONS: int = 8
    MAX_TOKEN_BUDGET: int = 12000
    MAX_SEARCH_ATTEMPTS: int = 4
    MAX_VALIDATION_RETRIES: int = 2
    STUCK_LOOP_WINDOW: int = 3


class ChunkingSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="CHUNKING_")

    CHUNK_SIZE_CHARS: int = 1800
    CHUNK_OVERLAP_CHARS: int = 200
    CHUNKING_MAX_WORKERS: int = 4
    WHOLE_DOC_CONTEXT_THRESHOLD_CHARS: int = 20000
    DEDUP_SIMILARITY_THRESHOLD: float = 0.92


class IngestionSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="INGESTION_")

    MIN_EXTRACTED_CHARS: int = 40
    MIN_TEXT_CHARS_PER_PAGE: int = 20
    MAX_JUNK_CHAR_RATIO: float = 0.3
    MAX_REPLACEMENT_CHAR_RATIO: float = 0.02
    IMAGE_COVERAGE_SCAN_THRESHOLD: float = 0.6


class EmbeddingSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="EMBEDDING_")

    PRIMARY_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    FALLBACK_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 768
    QDRANT_COLLECTION: str = "rag_harness_chunks"


class RetrievalSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="RETRIEVAL_")

    TOP_K_DENSE: int = 12
    TOP_K_SPARSE: int = 12
    TOP_K_FINAL_RERANK: int = 8
    RRF_K: int = 60
    MIN_RELEVANCE_SCORE: float = 0.15
    DEDUP_SIMILARITY_THRESHOLD: float = 0.92


class RerankSettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="RERANK_")

    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"


class NLISettings(_EnvBase):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="NLI_")

    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    NLI_ENTAILMENT_THRESHOLD: float = 0.6
    NLI_CONTRADICTION_MAX: float = 0.3


# ---------------------------------------------------------------------------
# Cached singleton accessors — every consumer calls these instead of
# constructing settings objects directly, so env is parsed once per process.
# ---------------------------------------------------------------------------
@lru_cache
def get_environment_settings() -> EnvironmentSettings:
    return EnvironmentSettings()


@lru_cache
def get_myllm_settings() -> MyLLMSettings:
    return MyLLMSettings()


@lru_cache
def get_openrouter_settings() -> OpenRouterSettings:
    return OpenRouterSettings()


@lru_cache
def get_groq_settings() -> GroqSettings:
    return GroqSettings()


@lru_cache
def get_llm_chain_settings() -> LLMChainSettings:
    return LLMChainSettings()


@lru_cache
def get_harness_loop_settings() -> HarnessLoopSettings:
    return HarnessLoopSettings()


@lru_cache
def get_chunking_settings() -> ChunkingSettings:
    return ChunkingSettings()


@lru_cache
def get_ingestion_settings() -> IngestionSettings:
    return IngestionSettings()


@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings()


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()


@lru_cache
def get_rerank_settings() -> RerankSettings:
    return RerankSettings()


@lru_cache
def get_nli_settings() -> NLISettings:
    return NLISettings()

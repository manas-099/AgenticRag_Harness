
"""
Dependency-injection wiring: builds the full object graph (infrastructure ->
application -> RAGPipeline) once, from settings, for FastAPI's Depends()
system. This is the ONLY place concrete infrastructure classes get
instantiated and wired to application use cases.
"""

from __future__ import annotations

from functools import lru_cache

from rag_harness.application.agent.decision import AgentDecisionEngine
from rag_harness.application.agent.graph_builder import RunAgenticQueryUseCase, build_harness_graph
from rag_harness.application.agent.tools import AgentTools
from rag_harness.application.generation.generate_answer_use_case import GenerateAnswerUseCase
from rag_harness.application.ingestion.contextual_chunker import ContextualChunker
from rag_harness.application.ingestion.file_router import FileRouter
from rag_harness.application.ingestion.ingest_document_use_case import IngestDocumentUseCase
from rag_harness.application.ingestion.structure_aware_splitter import StructureAwareSplitter
from rag_harness.application.rag_pipeline import RAGPipeline
from rag_harness.application.retrieval.fusion_service import RetrievalFusionService
from rag_harness.application.retrieval.metadata_filter_service import MetadataFilterService
from rag_harness.application.retrieval.model_registry import CrossEncoderRegistry
from rag_harness.application.retrieval.retrieve_use_case import RetrieveUseCase
from rag_harness.application.retrieval.section_candidate_selector import SectionCandidateSelector
from rag_harness.application.retrieval.section_filter_llm import SectionFilterLLM
from rag_harness.application.verification.citation_validator import CitationValidator
from rag_harness.application.verification.claim_extractor import ClaimExtractor
from rag_harness.application.verification.groundedness_checker import GroundednessChecker
from rag_harness.application.verification.verify_answer_use_case import VerifyAnswerUseCase
from rag_harness.config.settings import (
    get_chunking_settings,
    get_embedding_settings,
    get_environment_settings,
    get_groq_settings,
    get_harness_loop_settings,
    get_ingestion_settings,
    get_llm_chain_settings,
    get_myllm_settings,
    get_nli_settings,
    get_openrouter_settings,
    get_rerank_settings,
    get_retrieval_settings,
)
from rag_harness.infrastructure.cache import InMemoryCache
from rag_harness.infrastructure.document_parsers import AnyDocParser, DoclingParser, ExtractionQualityGate, PDFStructuralClassifier
from rag_harness.infrastructure.embeddings import SentenceTransformerEmbedder
from rag_harness.infrastructure.llm import LLMChain, LLMClientFactory
from rag_harness.infrastructure.sparse_search import BM25Retriever
from rag_harness.infrastructure.vector_stores import QdrantChunkStore, QdrantSectionStore


@lru_cache
def get_llm_chains() -> tuple[LLMChain, LLMChain]:
    """(agent_chain, generate_chain) — cached so both chains (and their
    shared MyLLM health monitor) are built exactly once per process, and so
    GET /v1/health/llm can reuse the same live instances the pipeline uses."""
    return LLMClientFactory.build_chains(
        get_llm_chain_settings(), get_myllm_settings(), get_openrouter_settings(), get_groq_settings(),
    )


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    embedding_settings = get_embedding_settings()
    environment_settings = get_environment_settings()

    agent_chain, generate_chain = get_llm_chains()
    embedder = SentenceTransformerEmbedder(embedding_settings)
    model_registry = CrossEncoderRegistry(get_rerank_settings(), get_nli_settings())

    vector_store = QdrantChunkStore(embedding_settings, environment_settings)
    section_store = QdrantSectionStore(environment_settings, embedding_settings, embedder)
    sparse_retriever = BM25Retriever()
    # Qdrant is the durable store; BM25Retriever's index is process-memory
    # only and starts empty on every restart (including `uvicorn --reload`
    # triggering on a source save). Rebuild it from Qdrant here so sparse
    # search doesn't silently go dark for anything ingested before the
    # most recent restart.
    sparse_retriever.rebuild_from_vector_store(vector_store)
    cache = InMemoryCache()

    classifier = PDFStructuralClassifier(get_ingestion_settings())
    file_router = FileRouter(classifier, AnyDocParser(), DoclingParser(), ExtractionQualityGate(get_ingestion_settings()))
    splitter = StructureAwareSplitter(get_chunking_settings())
    # Contextual chunking preamble: cheap, high-volume calls -> generate_chain
    # (never the agent chain, which should stay reserved for tool decisions).
    chunker = ContextualChunker(generate_chain, get_chunking_settings(), splitter)
    ingest_use_case = IngestDocumentUseCase(file_router, chunker, embedder, vector_store, sparse_retriever, section_store)

    fusion_service = RetrievalFusionService(get_retrieval_settings(), embedder, model_registry)
    retrieve_use_case = RetrieveUseCase(vector_store, sparse_retriever, fusion_service, embedder, get_retrieval_settings())

    section_selector = SectionCandidateSelector(section_store, model_registry, get_retrieval_settings())
    section_filter_llm = SectionFilterLLM(generate_chain)
    metadata_filter_service = MetadataFilterService(section_selector, section_filter_llm)

    generate_use_case = GenerateAnswerUseCase(generate_chain, cache, get_retrieval_settings())

    claim_extractor = ClaimExtractor(generate_chain)
    citation_validator = CitationValidator()
    groundedness_checker = GroundednessChecker(model_registry, get_nli_settings())
    verify_use_case = VerifyAnswerUseCase(claim_extractor, citation_validator, groundedness_checker)

    # The ReAct "what tool next?" decision uses the AGENT chain — kept
    # separately configurable from generation (see LLM_AGENT_CHAIN in .env).
    agent_tools = AgentTools(retrieve_use_case, vector_store)
    decision_engine = AgentDecisionEngine(agent_chain)
    loop_settings = get_harness_loop_settings()
    compiled_graph = build_harness_graph(agent_tools, decision_engine, generate_use_case, verify_use_case, generate_chain, loop_settings)
    agentic_use_case = RunAgenticQueryUseCase(compiled_graph, loop_settings)

    return RAGPipeline(
        ingest_use_case, retrieve_use_case, generate_use_case, cache,
        vector_store, sparse_retriever, agentic_use_case, metadata_filter_service,
    )
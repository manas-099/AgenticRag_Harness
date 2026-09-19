"""
Orchestrates: structure-aware split -> section title extraction (sequential,
since section inheritance depends on chunk order) -> concurrent LLM calls
generating a 1-2 sentence contextual preamble per chunk -> final Chunk
objects with full metadata.

Concurrency: preamble generation is I/O-bound (waiting on an LLM API), so
ThreadPoolExecutor gives a real speed win. Order is preserved via a
pre-sized result list indexed by submission order, not append-as-completed,
since downstream windowed-context lookups depend on stable chunk ordering.
"""

from __future__ import annotations

import logging
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from rag_harness.application.ingestion.section_title_extractor import extract_section_title
from rag_harness.application.ingestion.structure_aware_splitter import StructureAwareSplitter
from rag_harness.config.settings import ChunkingSettings
from rag_harness.domain.models import Chunk, ChunkMetadata
from rag_harness.domain.ports import LLMPort

logger = logging.getLogger("rag_harness.chunking")


class ContextualChunker:
    def __init__(self, llm_client: LLMPort, settings: ChunkingSettings, splitter: StructureAwareSplitter):
        self.llm_client = llm_client
        self.settings = settings
        self.splitter = splitter
        self._log_lock = threading.Lock()

    def chunk_document(
        self, doc_id: str, doc_version: str, source_type: str, page_texts: list[tuple[Optional[int], str, str]]
    ) -> list[Chunk]:
        """page_texts: list of (page_num, text, ingestion_route)."""
        full_document_text = "\n\n".join(text for _, text, _ in page_texts)

        all_chunks_flat: list[str] = []
        chunk_page_map: list[tuple[Optional[int], str]] = []  # (page_num, route)
        for page_num, text, route in page_texts:
            for base_chunk in self.splitter.split(text):
                all_chunks_flat.append(base_chunk)
                chunk_page_map.append((page_num, route))

        # Section titles must be computed sequentially — inheritance depends
        # on chunk order — BEFORE the concurrent LLM step runs.
        section_titles: list[Optional[str]] = []
        running_section: Optional[str] = None
        for raw_chunk in all_chunks_flat:
            running_section = extract_section_title(raw_chunk, running_section)
            section_titles.append(running_section)

        result_chunks: list[Optional[Chunk]] = [None] * len(all_chunks_flat)

        with ThreadPoolExecutor(max_workers=self.settings.CHUNKING_MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(
                    self._build_single_chunk,
                    idx, raw_chunk, chunk_page_map[idx], full_document_text, all_chunks_flat,
                    doc_id, doc_version, source_type, section_titles[idx],
                ): idx
                for idx, raw_chunk in enumerate(all_chunks_flat)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_chunks[idx] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {idx} failed entirely: {e} — building without preamble as last resort")
                    page_num, route = chunk_page_map[idx]
                    result_chunks[idx] = Chunk(
                        raw_text=all_chunks_flat[idx],
                        contextual_text=all_chunks_flat[idx],
                        metadata=ChunkMetadata(
                            doc_id=doc_id, doc_version=doc_version, source_type=source_type,
                            page_num=page_num, section_title=section_titles[idx], ingestion_route=route,
                        ),
                    )

        logger.info(f"Chunked doc={doc_id} into {len(result_chunks)} contextual chunks")
        return result_chunks

    def _build_single_chunk(
        self, idx: int, raw_chunk: str, page_route: tuple[Optional[int], str],
        full_document_text: str, all_chunks: list[str],
        doc_id: str, doc_version: str, source_type: str, section_title: Optional[str],
    ) -> Chunk:
        page_num, route = page_route
        preamble = self._generate_preamble(full_document_text, raw_chunk, idx, all_chunks)
        contextual_text = f"{preamble}\n{raw_chunk}" if preamble else raw_chunk

        with self._log_lock:
            logger.info(f"Chunk {idx + 1}/{len(all_chunks)} built (doc={doc_id}, page={page_num}, section={section_title})")

        return Chunk(
            raw_text=raw_chunk,
            contextual_text=contextual_text,
            metadata=ChunkMetadata(
                doc_id=doc_id, doc_version=doc_version, source_type=source_type,
                page_num=page_num, section_title=section_title, ingestion_route=route,
            ),
        )

    def _generate_preamble(self, full_document_text: str, target_chunk: str, chunk_index: int, all_chunks: list[str]) -> str:
        use_whole_doc = len(full_document_text) <= self.settings.WHOLE_DOC_CONTEXT_THRESHOLD_CHARS

        if use_whole_doc:
            context_source = full_document_text
        else:
            prev_chunk = all_chunks[chunk_index - 1] if chunk_index > 0 else ""
            next_chunk = all_chunks[chunk_index + 1] if chunk_index < len(all_chunks) - 1 else ""
            context_source = f"...{prev_chunk}\n[TARGET CHUNK HERE]\n{next_chunk}..."

        system_prompt = "You situate a text chunk within its document. Be concise: 1-2 sentences only."
        user_content = textwrap.dedent(f"""
            Document context:
            {context_source[:8000]}

            Target chunk:
            {target_chunk}

            Give a short 1-2 sentence context to situate this chunk within the overall document.
            Do not repeat the chunk text. Do not include document names, page numbers, or IDs —
            those are tracked separately as metadata.
        """).strip()

        try:
            preamble = self.llm_client.generate(system_prompt, user_content, max_tokens=300, temperature=0.0)
            if not preamble or not preamble.strip():
                with self._log_lock:
                    logger.error(f"Chunk {chunk_index}: LLM returned empty preamble")
                return ""
            return preamble.strip()
        except Exception as e:
            with self._log_lock:
                logger.error(f"Chunk {chunk_index}: preamble generation raised {type(e).__name__}: {e}")
            return ""
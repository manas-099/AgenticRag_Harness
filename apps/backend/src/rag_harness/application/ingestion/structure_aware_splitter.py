"""
Structure-aware text splitting: tries markdown headers first (so a chunk
never crosses a section boundary), then falls back to a paragraph/sentence/
word-aware recursive split within each section. Deliberately avoids raw
character-count slicing, which can sever a sentence mid-clause.
"""

from __future__ import annotations

import logging

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag_harness.config.settings import ChunkingSettings

logger = logging.getLogger("rag_harness.chunking.splitter")


class StructureAwareSplitter:
    def __init__(self, settings: ChunkingSettings):
        self.settings = settings
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE_CHARS,
            chunk_overlap=settings.CHUNK_OVERLAP_CHARS,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        try:
            sections = self.header_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Markdown header split failed ({e}) — falling back to plain recursive split")
            sections = None

        chunks: list[str] = []
        if sections:
            for section in sections:
                section_text = section.page_content
                if len(section_text) <= self.settings.CHUNK_SIZE_CHARS:
                    chunks.append(section_text)
                else:
                    chunks.extend(self.recursive_splitter.split_text(section_text))
        else:
            chunks = self.recursive_splitter.split_text(text)

        chunks = [c for c in chunks if c.strip()]
        logger.info(f"Structure-aware split produced {len(chunks)} chunks")
        return chunks
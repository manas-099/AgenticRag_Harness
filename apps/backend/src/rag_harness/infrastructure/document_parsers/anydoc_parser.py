"""
Concrete DocumentParserPort implementation for clean, uniform-text PDF pages.
Uses PyMuPDF's own structural text extraction directly — this IS the same
category of signal AnyDoc's PDF path relies on internally, and avoids an
unnecessary extra conversion step for an already-isolated page's text.
"""

from __future__ import annotations

import logging

import fitz

from rag_harness.domain.exceptions import IngestionError
from rag_harness.domain.ports import DocumentParserPort

logger = logging.getLogger("rag_harness.ingestion.anydoc")


class AnyDocParser(DocumentParserPort):
    def extract_page_text(self, file_path: str, page_num: int) -> str:
        try:
            with fitz.open(file_path) as doc:
                return doc[page_num].get_text("text")
        except Exception as e:
            logger.error(f"AnyDoc/PyMuPDF extraction failed on page {page_num}: {e}")
            raise IngestionError(f"Text extraction failed: {e}")
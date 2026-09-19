"""
Concrete DocumentParserPort implementation for scanned/mixed PDF pages.
Runs Docling's OCR conversion ONCE per document (not once per flagged page —
that was a real, measured slowness bug from repeated model reloads), then
groups extracted items by page using Docling's own provenance data
(item.prov[].page_no), which is the only reliable way to get per-page text
out of a DoclingDocument.
"""

from __future__ import annotations

import logging

from docling.document_converter import DocumentConverter

from rag_harness.domain.exceptions import IngestionError
from rag_harness.domain.ports import DocumentParserPort

logger = logging.getLogger("rag_harness.ingestion.docling")


class DoclingParser(DocumentParserPort):
    def __init__(self):
        self._converter = DocumentConverter()
        self._cache: dict[str, dict[int, str]] = {}  # file_path -> {page_num: text}, avoids re-converting the same doc

    def extract_page_text(self, file_path: str, page_num: int) -> str:
        if file_path not in self._cache:
            self._cache[file_path] = self._convert_whole_document(file_path)
        return self._cache[file_path].get(page_num, "")

    def _convert_whole_document(self, file_path: str) -> dict[int, str]:
        logger.info(f"Running batched Docling conversion for {file_path}")
        try:
            result = self._converter.convert(file_path)
            doc = result.document

            page_texts: dict[int, list[str]] = {}
            for item, _level in doc.iterate_items():
                if not hasattr(item, "prov") or not item.prov:
                    continue
                for prov in item.prov:
                    page_no = prov.page_no - 1  # Docling is 1-indexed, rest of pipeline is 0-indexed
                    page_texts.setdefault(page_no, [])
                    if hasattr(item, "text") and item.text:
                        page_texts[page_no].append(item.text)
                    elif hasattr(item, "export_to_markdown"):
                        try:
                            page_texts[page_no].append(item.export_to_markdown(doc))
                        except Exception:
                            page_texts[page_no].append(str(item))

            result_dict = {page_no: "\n\n".join(parts) for page_no, parts in page_texts.items()}

            if not result_dict:
                logger.warning(f"Per-page grouping produced nothing for {file_path} — falling back to full-doc markdown")
                full_markdown = doc.export_to_markdown()
                result_dict = {p_no - 1: full_markdown for p_no in doc.pages.keys()}

            logger.info(f"Docling conversion complete: {len(result_dict)} pages extracted from {file_path}")
            return result_dict

        except Exception as e:
            logger.error(f"Docling conversion failed for {file_path}: {e}")
            raise IngestionError(f"Docling extraction failed: {e}")

    def clear_cache(self, file_path: str | None = None) -> None:
        """Call after finishing ingestion of a document to free memory —
        the cache exists only to avoid re-converting within one ingestion run."""
        if file_path:
            self._cache.pop(file_path, None)
        else:
            self._cache.clear()
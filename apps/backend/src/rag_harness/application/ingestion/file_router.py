"""
Orchestrates per-page routing decisions: classifies each page, batches
Docling once per document if any page needs it, extracts each page via the
chosen parser, and reroutes AnyDoc output through Docling if the quality
gate flags it as suspicious.

Depends only on the DocumentParserPort implementations and the classifier/
quality-gate use-case helpers — no direct PyMuPDF/Docling imports here.
"""

from __future__ import annotations

import logging

import fitz

from rag_harness.domain.exceptions import IngestionError
from rag_harness.domain.models import PageStrategy, PageStrategyDecision
from rag_harness.infrastructure.document_parsers import (
    AnyDocParser,
    DoclingParser,
    ExtractionQualityGate,
    PDFStructuralClassifier,
)

logger = logging.getLogger("rag_harness.ingestion.file_router")


class PageExtractionResult:
    def __init__(self, page_num: int | None, text: str, route: str):
        self.page_num = page_num
        self.text = text
        self.route = route


class FileRouter:
    def __init__(
        self,
        classifier: PDFStructuralClassifier,
        anydoc_parser: AnyDocParser,
        docling_parser: DoclingParser,
        quality_gate: ExtractionQualityGate,
    ):
        self.classifier = classifier
        self.anydoc_parser = anydoc_parser
        self.docling_parser = docling_parser
        self.quality_gate = quality_gate

    def route_file(self, file_path: str, doc_id: str) -> list[PageExtractionResult]:
        if file_path.lower().endswith(".pdf"):
            return self._route_pdf(file_path, doc_id)
        elif file_path.lower().endswith(".md"):
            return self._route_md(file_path)
        raise IngestionError(f"Unsupported file type for {file_path} — only PDF and MD supported")

    def _route_pdf(self, file_path: str, doc_id: str) -> list[PageExtractionResult]:
        try:
            pdf_doc = fitz.open(file_path)
        except Exception as e:
            raise IngestionError(f"PDF open failed: {e}")

        try:
            decisions: list[PageStrategyDecision] = [self.classifier.classify_page(p) for p in pdf_doc]
        finally:
            pdf_doc.close()

        results: list[PageExtractionResult] = []
        for decision in decisions:
            logger.info(f"doc={doc_id} page={decision.page_num}: strategy={decision.strategy.value} ({decision.reason})")

            if decision.strategy == PageStrategy.WHOLE_PAGE_ANYDOC:
                text = self.anydoc_parser.extract_page_text(file_path, decision.page_num)
                suspicious, reason = self.quality_gate.is_suspicious(text)
                if suspicious:
                    logger.warning(f"Page {decision.page_num}: extraction suspicious ({reason}) — rerouting to Docling")
                    text = self.docling_parser.extract_page_text(file_path, decision.page_num)
                    results.append(PageExtractionResult(decision.page_num, text, "docling_fallback"))
                    continue
                results.append(PageExtractionResult(decision.page_num, text, "anydoc"))

            else:  # WHOLE_PAGE_DOCLING — covers scanned and mixed pages
                text = self.docling_parser.extract_page_text(file_path, decision.page_num)
                results.append(PageExtractionResult(decision.page_num, text, "docling"))

        self.docling_parser.clear_cache(file_path)
        return results

    def _route_md(self, file_path: str) -> list[PageExtractionResult]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise IngestionError(f"MD file read failed: {e}")
        logger.info("MD file ingested directly, no classification needed")
        return [PageExtractionResult(None, content, "direct_md")]
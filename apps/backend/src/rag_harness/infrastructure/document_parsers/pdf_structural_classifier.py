"""
Real, structural PDF page classifier using PyMuPDF — reads actual embedded
text operators, image coverage, and table structure rather than guessing
from rendered pixels. This is what decides, per page, whether AnyDoc (fast,
native text) or Docling (OCR, handles scans/mixed/tables) should run.
"""

from __future__ import annotations

import logging

import fitz  # PyMuPDF

from rag_harness.config.settings import IngestionSettings
from rag_harness.domain.models import PageStrategy, PageStrategyDecision

logger = logging.getLogger("rag_harness.ingestion.pdf_classifier")


class PDFStructuralClassifier:
    """Two-way decision only: uniform clean text -> AnyDoc; anything scanned
    OR mixed (table/scan/text together) -> Docling, whole page. No region
    splitting — deliberately simpler, per design decision."""

    def __init__(self, settings: IngestionSettings):
        self.settings = settings

    def classify_page(self, page: fitz.Page) -> PageStrategyDecision:
        text = page.get_text("text")
        text_char_count = len(text.strip())
        has_text_layer = text_char_count >= self.settings.MIN_TEXT_CHARS_PER_PAGE

        image_coverage = self._compute_image_coverage(page)
        has_table = self._has_table(page)

        if has_text_layer and image_coverage < 0.15 and not has_table:
            return PageStrategyDecision(
                page_num=page.number, strategy=PageStrategy.WHOLE_PAGE_ANYDOC, reason="uniform_text_page"
            )

        if not has_text_layer and image_coverage >= self.settings.IMAGE_COVERAGE_SCAN_THRESHOLD:
            reason = "uniform_scan_page"
        elif has_table:
            reason = "mixed_page_has_table"
        else:
            reason = "mixed_signals_text_and_image"

        return PageStrategyDecision(page_num=page.number, strategy=PageStrategy.WHOLE_PAGE_DOCLING, reason=reason)

    def _compute_image_coverage(self, page: fitz.Page) -> float:
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            return 0.0
        image_area = 0.0
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                for r in page.get_image_rects(xref):
                    image_area += r.width * r.height
            except Exception:
                continue
        return min(image_area / page_area, 1.0)

    def _has_table(self, page: fitz.Page) -> bool:
        try:
            tables = page.find_tables()
            return len(tables.tables) > 0
        except Exception as e:
            logger.warning(f"Table detection failed on page {page.number}: {e}")
            return False
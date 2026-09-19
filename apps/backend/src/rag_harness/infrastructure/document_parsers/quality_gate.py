"""
Post-extraction quality gate — catches cases the upfront structural
classifier missed: garbled font encodings, pre-existing bad OCR text, or
suspiciously short extraction results. Runs on AnyDoc's output specifically;
if it fails, the caller should reroute that page to Docling.
"""

from __future__ import annotations

from rag_harness.config.settings import IngestionSettings


class ExtractionQualityGate:
    def __init__(self, settings: IngestionSettings):
        self.settings = settings

    def is_suspicious(self, extracted_text: str) -> tuple[bool, str]:
        stripped = extracted_text.strip()
        if len(stripped) < self.settings.MIN_EXTRACTED_CHARS:
            return True, "extracted_text_too_short"

        replacement_count = extracted_text.count("\ufffd")
        ratio = replacement_count / max(len(extracted_text), 1)
        if ratio > self.settings.MAX_REPLACEMENT_CHAR_RATIO:
            return True, f"high_replacement_char_ratio_{ratio:.3f}"

        junk_chars = sum(1 for c in extracted_text if not (c.isalnum() or c.isspace() or c in ".,;:!?()-'\""))
        junk_ratio = junk_chars / max(len(extracted_text), 1)
        if junk_ratio > self.settings.MAX_JUNK_CHAR_RATIO:
            return True, f"high_junk_char_ratio_{junk_ratio:.3f}"

        return False, ""
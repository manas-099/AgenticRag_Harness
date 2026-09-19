"""
Extracts a section title for a chunk from markdown header syntax, inheriting
the previous chunk's section if this chunk doesn't start a new header.

Known limitation (tracked, not yet fixed): plain PyMuPDF text extraction
(AnyDoc-routed pages) does not produce '#'-style markdown headers, so this
only reliably finds sections on Docling-routed pages whose SectionHeaderItem
text happens to carry markdown syntax. A more robust version should read
Docling's item TYPE directly rather than regex-matching for '#', and add a
heuristic for plain-text pages (e.g. short, capitalized standalone lines).
"""

from __future__ import annotations

import re
from typing import Optional


def extract_section_title(chunk_text: str, previous_section: Optional[str]) -> Optional[str]:
    first_line = chunk_text.strip().split("\n")[0] if chunk_text.strip() else ""
    match = re.match(r"^#{1,6}\s+(.+)$", first_line)
    if match:
        return match.group(1).strip()
    return previous_section
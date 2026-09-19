"""
Abstract contract for extracting text from a source file. AnyDoc and Docling
are the two current implementations, chosen per-page by application-layer
routing logic — this port is what they both implement.
"""

from abc import ABC, abstractmethod


class DocumentParserPort(ABC):
    @abstractmethod
    def extract_page_text(self, file_path: str, page_num: int) -> str:
        raise NotImplementedError
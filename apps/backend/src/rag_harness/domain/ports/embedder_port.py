"""
Abstract contract for turning text into vectors.
"""

from abc import ABC, abstractmethod


class EmbedderPort(ABC):
    @abstractmethod
    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError
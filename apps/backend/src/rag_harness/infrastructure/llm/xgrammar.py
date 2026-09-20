"""
Wraps a Pydantic model into a strict JSON-schema response_format for LLM
backends that support structured output. Used by any use case that needs
reliable, parseable LLM output (agent decisions, claim extraction, section
selection) instead of free-text/regex parsing.
"""

from __future__ import annotations

from pydantic import BaseModel


class XGrammar:
    def __init__(self, pydantic_model: type[BaseModel], name: str) -> None:
        self.pydantic_model = pydantic_model
        self.name = name

    def build_response_format(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "strict": True,
                "schema": self.pydantic_model.model_json_schema(),
            },
        }